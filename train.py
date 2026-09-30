"""Train SIA severity models on rubric pseudo-labels.

Also writes the rubric pseudo-labels to data/pseudo_labeled_tickets.csv.

Examples:
    # final tfidf, embedding and distilbert models
    python train.py --augment 0.7
    # generalisation test: every template held out once (no models saved)
    python train.py --backend embedding --augment 0.7 --split template-cv --tag aug07
    # also learn from reviewer labels
    python train.py --augment 0.7 --feedback
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from sia import PRIORITY_LEVELS, URGENT_THRESHOLD
from sia.audit import ResolutionReference, classify
from sia.labeling import generate_labels
from sia import feedback
from sia.augment import augment, load_paraphrases
from sia.models import (DISTILBERT_DIR, EMBEDDING_DIR, MODELS_DIR, RESOLUTION_REF_PATH, DistilBertSeverityModel,
                        EmbeddingSeverityModel, TfidfSeverityModel, model_text)

REPORTS_DIR = "reports"


def template_folds(df, k, seed):
    """Assign every template to one of k folds, spreading each severity level across folds,
    so that across the k runs every template is held out exactly once."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    offset = 0
    for _, g in df.groupby("severity_level"):
        templates = list(rng.permutation(sorted(g["template"].unique())))
        for i, t in enumerate(templates):
            fold_of[t] = (i + offset) % k
        offset += len(templates)
    return df["template"].map(fold_of).to_numpy()


def evaluate(proba, val):
    y_true = val["severity_level"].to_numpy()
    y_pred = proba.argmax(axis=1)
    urgent_true = y_true >= URGENT_THRESHOLD
    urgent_pred = proba[:, URGENT_THRESHOLD:].sum(axis=1) >= 0.5
    mism_true = val["mismatch_type"].to_numpy()
    mism_pred = np.array([classify(a, p) for a, p in zip(val["assigned_level"], proba[:, URGENT_THRESHOLD:].sum(axis=1))])
    labels = list(range(len(PRIORITY_LEVELS)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=PRIORITY_LEVELS,
                                   output_dict=True, zero_division=0)
    return {
        "n_val": int(len(val)),
        "severity_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "severity_macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "urgent_vs_not_accuracy": round(accuracy_score(urgent_true, urgent_pred), 4),
        "mismatch_accuracy": round(accuracy_score(mism_true, mism_pred), 4),
        "mismatch_macro_f1": round(f1_score(mism_true, mism_pred, average="macro", zero_division=0), 4),
        "per_level_recall": {k: round(report[k]["recall"], 4) for k in PRIORITY_LEVELS},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_template_accuracy": {t: round(float(np.mean(y_pred[m] == y_true[m])), 3)
                                  for t in sorted(val["template"].unique())
                                  for m in [val["template"].to_numpy() == t]},
    }


def train_tfidf(train):
    model = TfidfSeverityModel.build()
    model.fit(train["Ticket_Description"].tolist(), train["Ticket_Subject"].tolist(),
              train["severity_level"].to_numpy(), sample_weight=train["label_confidence"].to_numpy())
    return model


def train_embedding(train):
    model = EmbeddingSeverityModel.build()
    model.fit(train["Ticket_Description"].tolist(), train["Ticket_Subject"].tolist(),
              train["severity_level"].to_numpy(), sample_weight=train["label_confidence"].to_numpy())
    return model


def train_distilbert(train, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sklearn.utils.class_weight import compute_class_weight

    torch.manual_seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    base = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(PRIORITY_LEVELS),
        id2label=dict(enumerate(PRIORITY_LEVELS)), label2id={p: i for i, p in enumerate(PRIORITY_LEVELS)})

    texts = [model_text(d, s) for d, s in zip(train["Ticket_Description"], train["Ticket_Subject"])]
    labels = train["severity_level"].to_numpy()
    enc = tokenizer(texts, truncation=True, max_length=args.max_length)
    rows = [{"input_ids": ids, "attention_mask": am, "labels": int(y)}
            for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels)]
    loader = DataLoader(rows, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: tokenizer.pad(b, return_tensors="pt"))

    present = np.unique(labels)
    weights = np.ones(len(PRIORITY_LEVELS))
    weights[present] = compute_class_weight("balanced", classes=present, y=labels)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    steps = args.epochs * len(loader)
    warmup = max(1, int(0.1 * steps))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min((s + 1) / warmup, max(0.0, (steps - s) / max(1, steps - warmup))))

    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            y = batch.pop("labels")
            loss = loss_fn(model(**batch).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{steps} loss {running / 50:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                running = 0.0
    model.eval()
    return DistilBertSeverityModel(model, tokenizer, args.max_length)


def fit_backend(backend, train, args):
    if backend == "tfidf":
        return train_tfidf(train)
    if backend == "embedding":
        return train_embedding(train)
    human = train[train["label_source"] == "reviewer"] if "label_source" in train else train.iloc[:0]
    pseudo = train.drop(human.index)
    sub = pseudo
    if len(pseudo) > args.train_size:
        sub, _ = train_test_split(pseudo, train_size=args.train_size, random_state=args.seed,
                                  stratify=pseudo["severity_level"])
    # Reviewer labels are never subsampled away; repeating them gives them their weight,
    # since the DistilBERT loss does not take per-sample weights.
    return train_distilbert(pd.concat([sub] + [human] * int(feedback.HUMAN_LABEL_WEIGHT)), args)


def prepare_train(train, args, paraphrases, human=None):
    # Only train tickets are rewritten, and only with paraphrases of their own template,
    # so held-out templates stay unseen in the template splits.
    if args.augment > 0:
        train = augment(train, paraphrases, prob=args.augment, seed=args.seed)
    if human is not None and len(human):
        # Reviewer labels replace the rubric's label for the same ticket.
        train = pd.concat([train[~train["Ticket_ID"].isin(human["Ticket_ID"])], human], ignore_index=True)
    return train


def save_model(backend, model, train, args):
    from sia.onnx_runtime import ONNX_FILE
    stale = os.path.join(DISTILBERT_DIR if backend == "distilbert" else EMBEDDING_DIR, ONNX_FILE)
    if backend != "tfidf" and os.path.exists(stale):
        os.remove(stale)  # an old export must never be served with new weights; rerun export_onnx.py
        print(f"Removed stale {stale}")
    if backend in ("tfidf", "embedding"):
        model.save()
    else:
        model.model.save_pretrained(DISTILBERT_DIR)
        model.tokenizer.save_pretrained(DISTILBERT_DIR)
        with open(os.path.join(DISTILBERT_DIR, "sia_meta.json"), "w") as f:
            json.dump({"max_length": args.max_length, "labels": PRIORITY_LEVELS,
                       "train_size": int(min(len(train), args.train_size)),
                       "epochs": args.epochs, "augment": args.augment}, f, indent=2)
    print(f"Saved {backend} model to {MODELS_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/customer_support_tickets.csv")
    ap.add_argument("--backend", choices=["tfidf", "embedding", "distilbert", "all"], default="all")
    ap.add_argument("--split", choices=["random", "template-cv"], default="random",
                    help="random: final models (saved). template-cv: every template held out once, "
                         "to measure generalisation (nothing saved).")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--train-size", type=int, default=4000,
                    help="DistilBERT training subsample (CPU budget). TF-IDF and embedding use the full train split.")
    ap.add_argument("--val-size", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--augment", type=float, default=0.0,
                    help="Probability of replacing a training ticket's template sentence with a paraphrase "
                         "(data/paraphrases.json). Validation tickets are never augmented.")
    ap.add_argument("--feedback", action="store_true",
                    help="Add reviewer labels from data/feedback.csv to training (never to validation).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="", help="Suffix for the metrics file, to keep experiment results apart.")
    ap.add_argument("--no-save", action="store_true", help="Evaluate only; do not overwrite saved models.")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    labeled = generate_labels(df)
    labeled.to_csv("data/pseudo_labeled_tickets.csv", index=False)
    paraphrases = load_paraphrases() if args.augment > 0 else {}
    human = feedback.as_training_rows(feedback.load()) if args.feedback else None
    if human is not None:
        print(f"Using {len(human)} reviewer-labelled tickets")
        labeled = labeled[~labeled["Ticket_ID"].isin(human["Ticket_ID"])]  # keep them out of validation

    # Each "run" is one (train, val) pair; template-cv has one per fold.
    if args.split == "template-cv":
        fold = template_folds(labeled, args.folds, args.seed)
        per_fold_val = max(1, args.val_size // args.folds)
        runs = []
        for k in range(args.folds):
            val = labeled[fold == k]
            runs.append((labeled[fold != k], val.sample(min(len(val), per_fold_val), random_state=args.seed)))
    else:
        train, val = train_test_split(labeled, test_size=0.2, random_state=args.seed,
                                      stratify=labeled["severity_level"])
        if len(val) > args.val_size:
            val = val.sample(args.val_size, random_state=args.seed)
        runs = [(train, val)]
    for i, (tr, va) in enumerate(runs):
        print(f"Split '{args.split}' run {i + 1}/{len(runs)}: {len(tr)} train / {len(va)} val tickets, "
              f"{tr['template'].nunique()} / {va['template'].nunique()} templates")

    save = not args.no_save and args.split == "random"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if save:
        ResolutionReference.from_frame(runs[0][0]).to_json(RESOLUTION_REF_PATH)

    backends = ["tfidf", "embedding", "distilbert"] if args.backend == "all" else [args.backend]
    for backend in backends:
        print(f"\n=== {backend} ===", flush=True)
        probas, vals, train_secs, infer_secs = [], [], 0.0, 0.0
        for i, (tr, va) in enumerate(runs):
            t0 = time.time()
            train_i = prepare_train(tr, args, paraphrases, human)
            model = fit_backend(backend, train_i, args)
            train_secs += time.time() - t0
            t0 = time.time()
            probas.append(model.predict_proba(va["Ticket_Description"].tolist(), va["Ticket_Subject"].tolist()))
            infer_secs += time.time() - t0
            vals.append(va)
            if len(runs) > 1:
                print(f"  fold {i + 1}/{len(runs)} done ({train_secs:.0f}s training so far)", flush=True)
        val_all = pd.concat(vals)
        metrics = evaluate(np.vstack(probas), val_all)
        config = {"epochs": args.epochs, "train_size": args.train_size} if backend == "distilbert" else {}
        config["augment"] = args.augment
        config["reviewer_labels"] = 0 if human is None else int(len(human))
        metrics.update({"backend": backend, "split": args.split, "tag": args.tag, "config": config,
                        "train_seconds": round(train_secs, 1),
                        "val_inference_tickets_per_sec": round(len(val_all) / max(infer_secs, 1e-9), 1)})
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_template_accuracy")},
                         indent=2), flush=True)

        suffix = f"_{args.tag}" if args.tag else ""
        with open(os.path.join(REPORTS_DIR, f"train_metrics_{backend}_{args.split}{suffix}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if args.split == "template-cv":
            # Out-of-fold predictions: every ticket scored by a model that never saw its template.
            # calibrate.py fits confidence calibration on these.
            np.savez(os.path.join(REPORTS_DIR, f"oof_{backend}{suffix}.npz"), proba=np.vstack(probas),
                     y=val_all["severity_level"].to_numpy(), assigned=val_all["assigned_level"].to_numpy())

        if save:
            save_model(backend, model, runs[0][0], args)


if __name__ == "__main__":
    main()
