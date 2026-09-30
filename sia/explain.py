"""Word-level evidence for learned-model decisions, by occlusion.

When the rubric has no cue for a ticket, the verdict comes from a learned
model and the dossier has no quoted phrase to show. Occlusion fills that gap:
delete one word at a time, re-score, and measure how much the model's support
for its own verdict falls. It needs no gradients, so it works for every backend,
and it measures the real model rather than a proxy.
"""

import re

import numpy as np

from sia import URGENT_THRESHOLD

WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def occlusion(model, description, subject=None, urgent_threshold=0.5, top_k=3, min_importance=0.02):
    """Return the words whose removal most weakens the model's verdict.

    importance = support(full text) - support(text without the word), where
    support = P(High or Critical) if the verdict is urgent, else 1 - that.
    Adjacent important words are merged into one phrase.
    """
    words = list(WORD.finditer(description or ""))
    if not words:
        return []
    variants = [description] + [description[:m.start()] + description[m.end():] for m in words]
    proba = model.predict_proba(variants, [subject] * len(variants))
    p_urgent = proba[:, URGENT_THRESHOLD:].sum(axis=1)
    urgent = p_urgent[0] >= urgent_threshold
    support = p_urgent if urgent else 1 - p_urgent
    importance = support[0] - support[1:]

    ranked = [i for i in np.argsort(-importance) if importance[i] >= min_importance][:top_k]
    spans = sorted((words[i].start(), words[i].end(), float(importance[i])) for i in ranked)

    # Merge words separated only by whitespace/punctuation into one phrase.
    merged = []
    for start, end, imp in spans:
        if merged and re.fullmatch(r"[\s,;:'\-]*", description[merged[-1][1]:start]):
            s, _, i = merged[-1]
            merged[-1] = (s, end, i + imp)
        else:
            merged.append((start, end, imp))

    return [{
        "signal": "model_attribution",
        "method": "occlusion",
        "matched_text": description[s:e],
        "start": s,
        "end": e,
        "importance": round(imp, 3),
        "meaning": f"Removing this lowers the model's confidence in its verdict by {imp:.2f}",
    } for s, e, imp in sorted(merged, key=lambda x: -x[2])]
