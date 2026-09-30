"""Severity models. All expose predict_proba(descriptions, subjects) -> (n, 4) array."""

import json
import os

import numpy as np

from sia import PRIORITY_LEVELS, URGENT_THRESHOLD
from sia import rubric

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DISTILBERT_DIR = os.path.join(MODELS_DIR, "distilbert")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.joblib")
EMBEDDING_DIR = os.path.join(MODELS_DIR, "embedding")
RESOLUTION_REF_PATH = os.path.join(MODELS_DIR, "resolution_reference.json")
N_LEVELS = len(PRIORITY_LEVELS)


def model_text(description, subject=None):
    """Text the models see: the description, plus the subject only when the body is too short."""
    desc = description if isinstance(description, str) else ""
    body = rubric.GREETING.sub("", desc)
    if len(body.split()) < rubric.MIN_DESCRIPTION_WORDS and isinstance(subject, str) and subject:
        return f"{subject}. {desc}".strip()
    return desc


def _texts(descriptions, subjects):
    subjects = subjects if subjects is not None else [None] * len(descriptions)
    return [model_text(d, s) for d, s in zip(descriptions, subjects)]


class RubricModel:
    """The rule-based rubric wrapped as a model; always available, no training needed."""

    name = "rubric"

    def predict_proba(self, descriptions, subjects=None, batch_size=None, progress=None):
        subjects = subjects if subjects is not None else [None] * len(descriptions)
        out = np.zeros((len(descriptions), N_LEVELS))
        for i, (d, s) in enumerate(zip(descriptions, subjects)):
            a = rubric.assess(d, s)
            # The rubric is a hard rule: leftover probability stays on the same side of the
            # urgency line as its verdict, so no threshold can flip what the rule decided.
            side = range(URGENT_THRESHOLD, N_LEVELS) if a.level >= URGENT_THRESHOLD else range(URGENT_THRESHOLD)
            others = [lvl for lvl in side if lvl != a.level]
            out[i, a.level] = a.confidence if others else 1.0
            for lvl in others:
                out[i, lvl] = (1 - a.confidence) / len(others)
        if progress:
            progress(len(descriptions), len(descriptions))
        return out


class TfidfSeverityModel:
    """Word + character n-gram TF-IDF with logistic regression. Fast CPU baseline."""

    name = "tfidf"

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    @staticmethod
    def build():
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer

        features = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
        ])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0)
        return TfidfSeverityModel(Pipeline([("features", features), ("clf", clf)]))

    def fit(self, descriptions, subjects, labels, sample_weight=None):
        self.pipeline.fit(_texts(descriptions, subjects), labels, clf__sample_weight=sample_weight)
        return self

    def predict_proba(self, descriptions, subjects=None, batch_size=None, progress=None):
        raw = self.pipeline.predict_proba(_texts(descriptions, subjects))
        out = np.zeros((len(descriptions), N_LEVELS))
        out[:, self.pipeline.classes_] = raw
        if progress:
            progress(len(descriptions), len(descriptions))
        return out

    def save(self, path=TFIDF_PATH):
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path=TFIDF_PATH):
        import joblib
        return cls(joblib.load(path))


class EmbeddingSeverityModel:
    """Frozen sentence encoder (all-MiniLM-L6-v2, mean pooled) + logistic regression.

    The encoder is never fine-tuned, so it keeps its general notion of sentence
    meaning; only a linear layer is learned from the pseudo-labels.
    """

    name = "embedding"
    ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, encoder, tokenizer, clf=None):
        import torch
        self.torch = torch
        self.encoder = encoder.eval()
        self.tokenizer = tokenizer
        self.clf = clf

    @classmethod
    def build(cls):
        from transformers import AutoModel, AutoTokenizer
        return cls(AutoModel.from_pretrained(cls.ENCODER), AutoTokenizer.from_pretrained(cls.ENCODER))

    def embed(self, texts, batch_size=128, progress=None):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), self.encoder.config.hidden_size), dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                     max_length=128, return_tensors="pt")
                hidden = self.encoder(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out[idx] = self.torch.nn.functional.normalize(pooled, dim=-1).numpy()
                if progress:
                    progress(min(start + batch_size, len(texts)), len(texts))
        return out

    def fit(self, descriptions, subjects, labels, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        X = self.embed(_texts(descriptions, subjects))
        self.clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0)
        self.clf.fit(X, labels, sample_weight=sample_weight)
        return self

    def predict_proba(self, descriptions, subjects=None, batch_size=128, progress=None):
        raw = self.clf.predict_proba(self.embed(_texts(descriptions, subjects), batch_size, progress))
        out = np.zeros((len(descriptions), N_LEVELS))
        out[:, self.clf.classes_] = raw
        return out

    def save(self, path=None):
        import joblib
        path = path or EMBEDDING_DIR
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        joblib.dump(self.clf, os.path.join(path, "classifier.joblib"))

    @classmethod
    def load(cls, path=None):
        import joblib
        from sia.onnx_runtime import onnx_enabled
        path = path or EMBEDDING_DIR
        clf = joblib.load(os.path.join(path, "classifier.joblib"))
        if onnx_enabled(path):
            return OnnxEmbeddingModel(path, clf)
        from transformers import AutoTokenizer, BertModel
        return cls(BertModel.from_pretrained(path), AutoTokenizer.from_pretrained(path), clf)


class OnnxEmbeddingModel:
    """EmbeddingSeverityModel served by onnxruntime (no torch import)."""

    name = "embedding"

    def __init__(self, path, clf):
        from sia.onnx_runtime import OnnxEncoder
        self.encoder = OnnxEncoder(path, max_length=128)
        self.clf = clf

    def embed(self, texts, batch_size=128, progress=None):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            hidden, mask = self.encoder([texts[i] for i in idx])
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
            out[idx] = pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return out

    def predict_proba(self, descriptions, subjects=None, batch_size=128, progress=None):
        raw = self.clf.predict_proba(self.embed(_texts(descriptions, subjects), batch_size, progress))
        out = np.zeros((len(descriptions), N_LEVELS))
        out[:, self.clf.classes_] = raw
        return out


class DistilBertSeverityModel:
    """Fine-tuned distilbert-base-uncased predicting the 4-level severity."""

    name = "distilbert"

    def __init__(self, model, tokenizer, max_length=96):
        import torch
        self.torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def load(cls, path=DISTILBERT_DIR):
        from sia.onnx_runtime import onnx_enabled
        meta_path = os.path.join(path, "sia_meta.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        if onnx_enabled(path):
            return OnnxDistilBertModel(path, meta.get("max_length", 96))
        # Concrete classes: the Auto* resolvers add ~10s of module scanning on load.
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
        return cls(DistilBertForSequenceClassification.from_pretrained(path),
                   DistilBertTokenizerFast.from_pretrained(path),
                   meta.get("max_length", 96))

    def predict_proba(self, descriptions, subjects=None, batch_size=64, progress=None):
        texts = _texts(descriptions, subjects)
        # Sort by length so each batch pads to a similar size.
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), N_LEVELS))
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                idx = order[start:start + batch_size]
                enc = self.tokenizer([texts[i] for i in idx], truncation=True, padding=True,
                                     max_length=self.max_length, return_tensors="pt")
                logits = self.model(**enc).logits
                out[idx] = self.torch.softmax(logits, dim=-1).numpy()
                if progress:
                    progress(min(start + batch_size, len(texts)), len(texts))
        return out


class HybridModel:
    """Rubric where it has a problem cue, learned model everywhere else.

    The learned models are trained on rubric labels, so where a Medium+ cue
    fires the model can at best reproduce the rubric (and on unseen phrasing it
    is often confidently wrong). Where the rubric is silent, or only sees a
    routine-request cue, it has no real evidence and the learned model decides.
    """

    def __init__(self, learned):
        self.learned = learned
        self.rules = RubricModel()
        self.name = f"hybrid({learned.name})"

    def predict_proba(self, descriptions, subjects=None, batch_size=64, progress=None):
        learned = self.learned.predict_proba(descriptions, subjects, batch_size=batch_size, progress=progress)
        rules = self.rules.predict_proba(descriptions, subjects)
        subjects = subjects if subjects is not None else [None] * len(descriptions)
        rule_decides = np.array([rubric.assess(d, s).level > rubric.LOW for d, s in zip(descriptions, subjects)])
        return np.where(rule_decides[:, None], rules, learned)


class OnnxDistilBertModel:
    """DistilBertSeverityModel served by onnxruntime (no torch import)."""

    name = "distilbert"

    def __init__(self, path, max_length=96):
        from sia.onnx_runtime import OnnxEncoder
        self.encoder = OnnxEncoder(path, max_length=max_length)

    def predict_proba(self, descriptions, subjects=None, batch_size=64, progress=None):
        from sia.onnx_runtime import softmax
        texts = _texts(descriptions, subjects)
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), N_LEVELS))
        for start in range(0, len(texts), batch_size):
            idx = order[start:start + batch_size]
            logits, _ = self.encoder([texts[i] for i in idx])
            out[idx] = softmax(logits)
            if progress:
                progress(min(start + batch_size, len(texts)), len(texts))
        return out


def available_backends():
    backends = []
    has_bert = os.path.exists(os.path.join(DISTILBERT_DIR, "config.json"))
    has_emb = os.path.exists(os.path.join(EMBEDDING_DIR, "classifier.joblib"))
    if has_bert or has_emb:
        backends.append("hybrid")
    if has_emb:
        backends.append("embedding")
    if has_bert:
        backends.append("distilbert")
    if os.path.exists(TFIDF_PATH):
        backends.append("tfidf")
    backends.append("rubric")
    return backends


def _load_learned(backend):
    from sia.calibration import CalibratedModel, temperature_for
    loaders = {"distilbert": DistilBertSeverityModel.load, "tfidf": TfidfSeverityModel.load,
               "embedding": EmbeddingSeverityModel.load}
    model = loaders[backend]()
    t = temperature_for(backend)
    return CalibratedModel(model, t) if t != 1.0 else model


def hybrid_base():
    """Learned model the hybrid falls back on: set in models/calibration.json, else the best available."""
    from sia.calibration import load_config
    trained = available_backends()
    choice = load_config().get("hybrid_base")
    if choice in trained:
        return choice
    return next(b for b in ("embedding", "distilbert", "tfidf") if b in trained)


def load_model(backend="auto"):
    if backend == "auto":
        backend = available_backends()[0]
    if backend == "hybrid":
        return HybridModel(_load_learned(hybrid_base()))
    if backend in ("distilbert", "tfidf", "embedding"):
        return _load_learned(backend)
    if backend == "rubric":
        return RubricModel()
    raise ValueError(f"Unknown backend {backend!r}; choose from {available_backends()}")


def load_resolution_reference():
    from sia.audit import ResolutionReference
    if os.path.exists(RESOLUTION_REF_PATH):
        return ResolutionReference.from_json(RESOLUTION_REF_PATH)
    return None
