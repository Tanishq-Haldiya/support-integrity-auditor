"""Reviewer feedback: the path from pseudo-labels to real labels.

Every review in the app appends one row to data/feedback.csv. Training can then
use those rows as human labels (they override the rubric's label for the same
ticket and carry more weight), and evaluation can report how often reviewers
agree with the auditor's flags.
"""

import csv
import os
from datetime import datetime, timezone

import pandas as pd

from sia import LEVEL_OF, PRIORITY_LEVELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_PATH = os.path.join(ROOT, "data", "feedback.csv")
FIELDS = ["reviewed_at", "Ticket_ID", "Ticket_Subject", "Ticket_Description", "Priority_Level",
          "model_backend", "model_verdict", "model_severity", "reviewer_severity", "reviewer_note"]
HUMAN_LABEL_WEIGHT = 3.0


def record(ticket, model_backend, model_verdict, model_severity, reviewer_severity, note="", path=FEEDBACK_PATH):
    if reviewer_severity not in PRIORITY_LEVELS:
        raise ValueError(f"reviewer_severity must be one of {PRIORITY_LEVELS}")
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "Ticket_ID": ticket.get("Ticket_ID"), "Ticket_Subject": ticket.get("Ticket_Subject") or "",
            "Ticket_Description": ticket.get("Ticket_Description"), "Priority_Level": ticket.get("Priority_Level"),
            "model_backend": model_backend, "model_verdict": model_verdict, "model_severity": model_severity,
            "reviewer_severity": reviewer_severity, "reviewer_note": note,
        })


def load(path=FEEDBACK_PATH):
    """Latest review per ticket, or an empty frame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=FIELDS)
    df = pd.read_csv(path)
    return df.sort_values("reviewed_at").drop_duplicates("Ticket_ID", keep="last").reset_index(drop=True)


def as_training_rows(fb):
    """Feedback in the shape train.py expects, labelled by the reviewer."""
    if fb.empty:
        return fb
    out = fb[["Ticket_ID", "Ticket_Subject", "Ticket_Description", "Priority_Level"]].copy()
    out["severity_level"] = fb["reviewer_severity"].map(LEVEL_OF)
    out["label_confidence"] = HUMAN_LABEL_WEIGHT
    out["label_source"] = "reviewer"
    return out


def agreement(fb):
    """How often reviewers upheld the auditor's verdict (a precision estimate for flags)."""
    if fb.empty:
        return {}
    urgent = lambda s: LEVEL_OF[s] >= 2  # noqa: E731
    same_side = [urgent(m) == urgent(r) for m, r in zip(fb["model_severity"], fb["reviewer_severity"])]
    flagged = fb["model_verdict"] != "Consistent"
    upheld = pd.Series(same_side)[flagged.to_numpy()]
    return {
        "reviews": int(len(fb)),
        "flags_reviewed": int(flagged.sum()),
        "flags_upheld": int(upheld.sum()),
        "flag_precision": round(float(upheld.mean()), 3) if len(upheld) else None,
        "exact_severity_agreement": round(float((fb["model_severity"] == fb["reviewer_severity"]).mean()), 3),
    }
