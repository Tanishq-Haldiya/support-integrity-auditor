import json

import numpy as np
import pandas as pd
import pytest

from sia.audit import ResolutionReference, audit_frame, classify, validate_columns
from sia.models import RubricModel


def frame(**overrides):
    row = {"Ticket_ID": "T1", "Ticket_Subject": "x", "Priority_Level": "Low",
           "Resolution_Time_Hours": 5, "Ticket_Description": "Hi Support, The application crashes on launch."}
    row.update(overrides)
    return pd.DataFrame([row])


@pytest.mark.parametrize("assigned, p_urgent, expected", [
    (0, 0.9, "Hidden Crisis"),
    (1, 0.9, "Hidden Crisis"),
    (2, 0.9, "Consistent"),   # an urgent ticket correctly labelled High must not be flagged
    (3, 0.1, "False Alarm"),
    (0, 0.1, "Consistent"),
])
def test_classify(assigned, p_urgent, expected):
    assert classify(assigned, p_urgent) == expected


def test_hidden_crisis_dossier_is_grounded():
    out = audit_frame(frame(), RubricModel()).iloc[0]
    assert out["mismatch_type"] == "Hidden Crisis"
    dossier = json.loads(out["dossier"])
    text = out["Ticket_Description"]
    cues = [e for e in dossier["feature_evidence"] if e["signal"] == "text_cue"]
    assert cues and all(e["matched_text"] in text for e in cues)
    assert dossier["severity_delta"] == "Low -> High"


def test_high_priority_urgent_ticket_is_consistent():
    out = audit_frame(frame(Priority_Level="High"), RubricModel()).iloc[0]
    assert out["mismatch_type"] == "Consistent"
    assert pd.isna(out["dossier"])


def test_false_alarm():
    out = audit_frame(frame(Priority_Level="Critical",
                            Ticket_Description="Hi Support, What are your support operating hours?"),
                      RubricModel()).iloc[0]
    assert out["mismatch_type"] == "False Alarm"


def test_non_range_index_is_handled():
    df = pd.concat([frame(), frame(Ticket_ID="T2", Priority_Level="High")])
    df.index = [10, 3]
    out = audit_frame(df, RubricModel())
    assert list(out["mismatch_type"]) == ["Hidden Crisis", "Consistent"]


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="Priority_Level"):
        validate_columns(pd.DataFrame({"Ticket_ID": [1], "Ticket_Description": ["x"]}))


def test_hybrid_uses_model_only_when_rubric_is_silent():
    from sia.models import HybridModel

    class AlwaysHigh:
        name = "fake"

        def predict_proba(self, descriptions, subjects=None, **kw):
            return np.tile([0.0, 0.0, 1.0, 0.0], (len(descriptions), 1))

    proba = HybridModel(AlwaysHigh()).predict_proba(
        ["Our checkout page is down.", "I noticed a double charge on my statement."])
    assert proba[0].argmax() == 2          # no problem cue -> learned model decides
    assert proba[1].argmax() == 1          # billing cue -> rubric decides (Medium)
    assert np.allclose(proba.sum(axis=1), 1)


def test_attributions_added_only_when_no_cue_explains_the_verdict():
    from sia.audit import add_attributions

    class OutageModel:
        name = "fake"

        def predict_proba(self, descriptions, subjects=None, **kw):
            return np.array([[0.05, 0.05, 0.85, 0.05] if "kaput" in d or "crashes" in d else [0.8, 0.1, 0.05, 0.05]
                             for d in descriptions])

    # "kaput" is not a rubric cue, so only the model can explain the first verdict.
    df = pd.concat([frame(Ticket_Description="Hi Support, the payments page is kaput for everyone."),
                    frame(Ticket_ID="T2")])  # T2 has the rubric cue "crashes"
    out = add_attributions(audit_frame(df, OutageModel(), threshold=0.5), OutageModel(), threshold=0.5)
    d1 = json.loads(out.loc[0, "dossier"])
    assert [e["matched_text"] for e in d1["feature_evidence"] if e["signal"] == "model_attribution"] == ["kaput"]
    d2 = out.loc[1, "dossier"]
    assert d2 is None or not any(e["signal"] == "model_attribution" for e in json.loads(d2)["feature_evidence"])


def test_threshold_changes_the_verdict():
    assert classify(0, 0.3, threshold=0.25) == "Hidden Crisis"
    assert classify(0, 0.3, threshold=0.5) == "Consistent"
    assert classify(2, 0.3, threshold=0.25) == "Consistent"


def test_resolution_percentile():
    ref = ResolutionReference({"Low": np.array([10, 20, 30, 40])})
    assert ref.percentile("Low", 15) == 75.0
    assert ref.percentile("High", 15) is None


def test_rubric_probabilities_never_cross_the_urgency_line():
    # A cue-less ticket defaults to Low; at a 0.25 threshold it must still read as not urgent.
    proba = RubricModel().predict_proba(["Where is your office?", "The app crashes on start."])
    assert proba[0, 2:].sum() == 0 and proba[1, :2].sum() == 0
    assert np.allclose(proba.sum(axis=1), 1)
