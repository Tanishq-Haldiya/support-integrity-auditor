"""Mismatch decision and evidence dossier generation."""

import json

import numpy as np
import pandas as pd

from sia import LEVEL_OF, PRIORITY_LEVELS, REQUIRED_COLUMNS, URGENT_THRESHOLD
from sia import rubric


class ResolutionReference:
    """Resolution-time distribution per assigned priority.

    Resolution time is an outcome of the *assigned* priority (agents work
    High tickets faster), so on its own it says little about true severity.
    We only use it as handling evidence: how fast was this ticket resolved
    compared with other tickets that carried the same assigned priority?
    """

    def __init__(self, samples):
        self.samples = {p: np.sort(np.asarray(v, dtype=float)) for p, v in samples.items() if len(v)}

    @classmethod
    def from_frame(cls, df):
        if "Resolution_Time_Hours" not in df:
            return cls({})
        clean = df.dropna(subset=["Resolution_Time_Hours", "Priority_Level"])
        return cls({p: g["Resolution_Time_Hours"].to_numpy() for p, g in clean.groupby("Priority_Level")})

    def to_json(self, path):
        # Store percentiles rather than raw rows; enough for lookups and small on disk.
        qs = np.linspace(0, 100, 101)
        data = {p: np.percentile(v, qs).round(3).tolist() for p, v in self.samples.items()}
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def percentile(self, priority, hours):
        """Share (0-100) of same-priority tickets that took longer than this one."""
        ref = self.samples.get(priority)
        if ref is None or hours is None or pd.isna(hours):
            return None
        slower = len(ref) - np.searchsorted(ref, float(hours), side="right")
        return round(100.0 * slower / len(ref), 1)


def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required column(s): {', '.join(missing)}")
    bad = sorted(set(df["Priority_Level"].dropna()) - set(PRIORITY_LEVELS))
    if bad:
        raise ValueError(f"Unknown Priority_Level value(s): {bad}. Expected one of {PRIORITY_LEVELS}.")


def classify(assigned_level, p_urgent, threshold=0.5):
    inferred_urgent = p_urgent >= threshold
    assigned_urgent = assigned_level >= URGENT_THRESHOLD
    if inferred_urgent and not assigned_urgent:
        return "Hidden Crisis"
    if assigned_urgent and not inferred_urgent:
        return "False Alarm"
    return "Consistent"


def inferred_level_from_proba(proba, threshold=0.5):
    """Most likely level, constrained to the side of the urgency boundary the model favours."""
    p_urgent = proba[URGENT_THRESHOLD:].sum()
    if p_urgent >= threshold:
        return URGENT_THRESHOLD + int(np.argmax(proba[URGENT_THRESHOLD:]))
    return int(np.argmax(proba[:URGENT_THRESHOLD]))


def _resolution_evidence(row, ref):
    hours = row.get("Resolution_Time_Hours")
    if ref is None or hours is None or pd.isna(hours):
        return None
    pct = ref.percentile(row["Priority_Level"], hours)
    if pct is None:
        return None
    if pct >= 80:
        note = f"Resolved faster than {pct:.0f}% of other {row['Priority_Level']} tickets: agents appear to have treated it as more urgent than its label."
    elif pct <= 20:
        note = f"Slower than {100 - pct:.0f}% of other {row['Priority_Level']} tickets: handled with less urgency than its label suggests."
    else:
        note = f"Resolution time is typical for {row['Priority_Level']} tickets; timing adds no extra evidence."
    return {
        "signal": "resolution_time",
        "value": f"{hours:g} hours",
        "faster_than_pct_of_same_priority": pct,
        "interpretation": note,
        "used_in_decision": False,
    }


def build_dossier(row, proba, assessment, backend, ref=None, threshold=0.5):
    assigned = row["Priority_Level"]
    assigned_level = LEVEL_OF[assigned]
    p_urgent = float(proba[URGENT_THRESHOLD:].sum())
    mismatch = classify(assigned_level, p_urgent, threshold)
    if mismatch == "Consistent":
        return None

    inferred = PRIORITY_LEVELS[inferred_level_from_proba(proba, threshold)]
    rule_urgent = assessment.level >= URGENT_THRESHOLD
    model_urgent = p_urgent >= threshold
    evidence = rubric.evidence(assessment)
    res = _resolution_evidence(row, ref)
    if res:
        evidence.append(res)

    cue_text = ", ".join(f'"{e["matched_text"]}" ({e["indicates"]})' for e in evidence if e["signal"] == "text_cue")
    analysis = (
        f"Assigned {assigned}; the {backend} model estimates P(High or Critical) = {p_urgent:.2f} "
        f"(urgent threshold {threshold:.2f}), so the inferred severity is {inferred}. "
        + (f"Text cues found: {cue_text}. " if cue_text else "No rubric cues were found in the text. ")
        + ("The rule-based rubric agrees." if rule_urgent == model_urgent
           else f"The rule-based rubric disagrees (rubric says {assessment.severity}); flagged for manual review.")
    )

    return {
        "ticket_id": str(row["Ticket_ID"]),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mismatch,
        "severity_delta": f"{assigned} -> {inferred}",
        "severity_gap": LEVEL_OF[inferred] - assigned_level,
        "confidence": round(p_urgent if model_urgent else 1 - p_urgent, 3),
        "urgent_threshold": round(threshold, 3),
        "needs_review": rule_urgent != model_urgent,
        "model": {"backend": backend,
                  "probabilities": {lvl: round(float(p), 3) for lvl, p in zip(PRIORITY_LEVELS, proba)}},
        "rubric_severity": assessment.severity,
        "feature_evidence": evidence,
        "constraint_analysis": analysis,
    }


def needs_attribution(dossier):
    """True when no rubric cue explains the verdict's side of the boundary."""
    urgent = LEVEL_OF[dossier["inferred_severity"]] >= URGENT_THRESHOLD
    return not any(e["signal"] == "text_cue" and (LEVEL_OF[e["indicates"]] >= URGENT_THRESHOLD) == urgent
                   for e in dossier["feature_evidence"])


def add_attributions(result, model, threshold=None, only_unexplained=True):
    """Attach occlusion evidence to flagged tickets whose verdict no rubric cue explains."""
    from sia.calibration import urgent_threshold
    from sia.explain import occlusion
    threshold = urgent_threshold() if threshold is None else threshold
    out = result.copy()
    for i, row in out[out["dossier"].notna()].iterrows():
        d = json.loads(row["dossier"])
        if only_unexplained and not needs_attribution(d):
            continue
        subject = row.get("Ticket_Subject") if "Ticket_Subject" in out else None
        d["feature_evidence"] += occlusion(model, row["Ticket_Description"], subject, threshold)
        out.at[i, "dossier"] = json.dumps(d)
    return out


def audit_frame(df, model, ref=None, batch_size=64, progress=None, threshold=None):
    """Run the model over a ticket frame and attach verdicts and dossiers."""
    from sia.calibration import urgent_threshold
    threshold = urgent_threshold() if threshold is None else threshold
    validate_columns(df)
    out = df.reset_index(drop=True).copy()
    subjects = out["Ticket_Subject"] if "Ticket_Subject" in out else pd.Series([None] * len(out))
    proba = model.predict_proba(out["Ticket_Description"].tolist(), subjects.tolist(),
                                batch_size=batch_size, progress=progress)

    assessments = [rubric.assess(d, s) for d, s in zip(out["Ticket_Description"], subjects)]
    p_urgent = proba[:, URGENT_THRESHOLD:].sum(axis=1)
    assigned = out["Priority_Level"].map(LEVEL_OF)

    out["inferred_severity"] = [PRIORITY_LEVELS[inferred_level_from_proba(p, threshold)] for p in proba]
    out["p_high_or_critical"] = p_urgent.round(4)
    out["mismatch_type"] = [classify(a, p, threshold) for a, p in zip(assigned, p_urgent)]
    out["confidence"] = np.where(p_urgent >= threshold, p_urgent, 1 - p_urgent).round(4)
    out["rubric_severity"] = [a.severity for a in assessments]
    out["needs_review"] = [(a.level >= URGENT_THRESHOLD) != (p >= threshold) for a, p in zip(assessments, p_urgent)]
    dossiers = [build_dossier(row, pr, a, model.name, ref, threshold)
                for row, pr, a in zip(out.to_dict("records"), proba, assessments)]
    out["dossier"] = [json.dumps(d) if d else None for d in dossiers]
    return out
