"""Paraphrase augmentation.

The dataset has only 30 description templates, so a model can memorise one
sentence per label and learn shortcuts (e.g. "transaction" -> fraud). Here
each ticket's template sentence is swapped, with some probability, for a
hand-written paraphrase that keeps the same meaning and severity. The
greeting and random filler text around it are kept.

Paraphrases that are near-copies of any evaluation ticket are dropped
before use, so the test sets stay unseen.
"""

import difflib
import json
import os
import re

import numpy as np
import pandas as pd

from sia import rubric
from sia.labeling import SENTENCE, template_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAPHRASES_PATH = os.path.join(ROOT, "data", "paraphrases.json")
EVAL_SETS = [os.path.join(ROOT, "data", "challenge_set.csv"), os.path.join(ROOT, "data", "fresh_test_set.csv")]

# A paraphrase this close to an evaluation ticket counts as leaked.
MAX_CHAR_SIMILARITY = 0.65
MAX_WORD_OVERLAP = 0.45
_STOP = {"i", "my", "the", "a", "to", "and", "is", "it", "on", "of", "for", "me", "you", "your",
         "in", "this", "that", "was", "have", "be", "can", "do"}


def _words(text):
    return set(re.findall(r"[a-z0-9']+", text.lower())) - _STOP


def _too_close(text, eval_texts):
    words = _words(text)
    for other in eval_texts:
        if difflib.SequenceMatcher(None, text.lower(), other.lower()).ratio() >= MAX_CHAR_SIMILARITY:
            return True
        union = words | _words(other)
        if union and len(words & _words(other)) / len(union) >= MAX_WORD_OVERLAP:
            return True
    return False


def load_paraphrases(path=PARAPHRASES_PATH, eval_paths=EVAL_SETS, verbose=True):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    eval_texts = []
    for p in eval_paths:
        if os.path.exists(p):
            eval_texts += pd.read_csv(p)["Ticket_Description"].tolist()
    kept, dropped = {}, 0
    for key, options in raw.items():
        ok = [o for o in options if not _too_close(o, eval_texts)]
        dropped += len(options) - len(ok)
        if ok:
            kept[key] = ok
    if verbose:
        total = sum(len(v) for v in raw.values())
        print(f"Paraphrases: kept {total - dropped}/{total} (dropped {dropped} near-copies of evaluation tickets)")
    return kept


def augment(df, paraphrases, prob=0.7, seed=0):
    """Return a copy of df where each description's template sentence is replaced
    by a random paraphrase with probability `prob`. Labels are left unchanged."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    new_desc = []
    for desc in out["Ticket_Description"]:
        key = template_key(desc)
        options = paraphrases.get(key)
        if not options or rng.random() >= prob:
            new_desc.append(desc)
            continue
        greeting = rubric.GREETING.match(desc)
        prefix = greeting.group(0) if greeting else ""
        body = desc[len(prefix):]
        m = SENTENCE.match(body.strip())
        rest = body.strip()[len(m.group(1)):] if m else ""
        new_desc.append(f"{prefix}{options[rng.integers(len(options))]}{rest}")
    out["Ticket_Description"] = new_desc
    return out
