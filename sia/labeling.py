"""Pseudo-label generation and signal diagnostics."""

import re

from scipy import stats

from sia import LEVEL_OF, PRIORITY_LEVELS, URGENT_THRESHOLD
from sia import rubric
from sia.audit import ResolutionReference

SENTENCE = re.compile(r"^(.*?[.?!])(\s|$)")


def template_key(description):
    """First sentence of the body. Used to keep near-duplicate tickets on one side of a split."""
    body = rubric.GREETING.sub("", str(description)).strip()
    m = SENTENCE.match(body)
    return (m.group(1) if m else body).strip().lower()


def mismatch_type(assigned_level, inferred_level):
    if inferred_level >= URGENT_THRESHOLD > assigned_level:
        return "Hidden Crisis"
    if assigned_level >= URGENT_THRESHOLD > inferred_level:
        return "False Alarm"
    return "Consistent"


def generate_labels(df):
    out = df.copy()
    subjects = out["Ticket_Subject"] if "Ticket_Subject" in out else [None] * len(out)
    assessments = [rubric.assess(d, s) for d, s in zip(out["Ticket_Description"], subjects)]

    out["severity_level"] = [a.level for a in assessments]
    out["inferred_severity"] = [a.severity for a in assessments]
    out["label_confidence"] = [a.confidence for a in assessments]
    out["matched_cues"] = ["; ".join(sorted({m.cue for m in a.matches})) for a in assessments]
    out["assigned_level"] = out["Priority_Level"].map(LEVEL_OF)
    out["severity_gap"] = out["severity_level"] - out["assigned_level"]
    out["mismatch_type"] = [mismatch_type(a, i) for a, i in zip(out["assigned_level"], out["severity_level"])]
    out["mismatch_label"] = (out["mismatch_type"] != "Consistent").astype(int)
    out["template"] = out["Ticket_Description"].map(template_key)

    if "Resolution_Time_Hours" in out:
        ref = ResolutionReference.from_frame(out)
        out["faster_than_pct_same_priority"] = [
            ref.percentile(p, h) for p, h in zip(out["Priority_Level"], out["Resolution_Time_Hours"])
        ]
    return out


def resolution_signal_report(labeled):
    """Does resolution time carry information about severity beyond the assigned priority?

    Pooled across priorities, fast resolution correlates with severity only
    because agents resolve tickets *labelled* High faster. The honest test is
    within each assigned priority.
    """
    rows = []
    pooled = stats.spearmanr(labeled["Resolution_Time_Hours"], labeled["assigned_level"])
    for prio in PRIORITY_LEVELS:
        g = labeled[labeled["Priority_Level"] == prio]
        groups = [x["Resolution_Time_Hours"].to_numpy() for _, x in g.groupby("severity_level") if len(x) > 5]
        if len(groups) < 2:
            rows.append({"assigned_priority": prio, "n": len(g), "note": "only one inferred level present"})
            continue
        h, p = stats.kruskal(*groups)
        k, n = len(groups), sum(len(x) for x in groups)
        rho = stats.spearmanr(g["Resolution_Time_Hours"], g["severity_level"])
        rows.append({
            "assigned_priority": prio,
            "n": n,
            "median_hours_by_inferred_level": {
                PRIORITY_LEVELS[lvl]: float(x["Resolution_Time_Hours"].median())
                for lvl, x in g.groupby("severity_level")
            },
            "spearman_rho": round(float(rho.statistic), 4),
            "kruskal_p": round(float(p), 4),
            "eta_squared": round(max(0.0, (h - k + 1) / (n - k)), 5),
        })
    return {
        "spearman_resolution_vs_assigned_priority": round(float(pooled.statistic), 4),
        "within_priority": rows,
    }


# --- Naive baselines, for comparison ------------------------------------------
# The obvious first approach: count urgent-sounding keywords, and treat fast
# resolution as a sign of severity. Kept to show why the rubric replaces both.
KEYWORD_BASELINE_TERMS = ["crash", "down", "fail", "immediate", "urgent", "asap", "broken",
                   "critical", "emergency", "unacceptable", "manager"]


def keyword_fusion_labels(df):
    """Baseline labels: keyword count and inverted resolution time, averaged; top 30% = urgent."""
    text = df["Ticket_Description"].fillna("").str.lower()
    score = sum(text.str.contains(r"\b" + w + r"\w*\b", regex=True).astype(int) for w in KEYWORD_BASELINE_TERMS)
    score = score + (text.str.contains("not") & (text.str.contains("work") | text.str.contains("load"))).astype(int)
    nlp = score / max(score.max(), 1)
    time = 1 - df["Resolution_Time_Hours"] / df["Resolution_Time_Hours"].max()
    fused = 0.5 * nlp + 0.5 * time
    return (fused >= fused.quantile(0.70)).to_numpy()


def keyword_filler_hit_rate(df):
    """Share of baseline keyword hits that occur outside the ticket's first sentence (i.e. in filler)."""
    hits = outside = 0
    for desc in df["Ticket_Description"].fillna(""):
        body = rubric.GREETING.sub("", desc).lower()
        first = len(template_key(desc))
        for w in KEYWORD_BASELINE_TERMS:
            for m in re.finditer(r"\b" + w + r"\w*\b", body):
                hits += 1
                outside += m.start() >= first
    return {"keyword_hits": hits, "hits_in_filler_text": outside,
            "filler_share": round(outside / hits, 4) if hits else 0.0}
