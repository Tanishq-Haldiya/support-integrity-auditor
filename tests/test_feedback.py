import pytest

from sia import feedback


def test_record_load_and_agreement(tmp_path):
    path = tmp_path / "feedback.csv"
    t = {"Ticket_ID": "T1", "Ticket_Description": "App crashes", "Priority_Level": "Low"}
    feedback.record(t, "hybrid", "Hidden Crisis", "High", "Critical", path=str(path))
    feedback.record(dict(t, Ticket_ID="T2"), "hybrid", "False Alarm", "Low", "High", path=str(path))
    # A later review of the same ticket replaces the earlier one.
    feedback.record(t, "hybrid", "Hidden Crisis", "High", "High", path=str(path))

    fb = feedback.load(str(path))
    assert len(fb) == 2
    assert fb.set_index("Ticket_ID").loc["T1", "reviewer_severity"] == "High"

    stats = feedback.agreement(fb)
    assert stats["flags_reviewed"] == 2 and stats["flags_upheld"] == 1 and stats["flag_precision"] == 0.5

    rows = feedback.as_training_rows(fb)
    assert set(rows["severity_level"]) == {2} and (rows["label_confidence"] == feedback.HUMAN_LABEL_WEIGHT).all()


def test_invalid_severity_rejected(tmp_path):
    with pytest.raises(ValueError):
        feedback.record({"Ticket_ID": "T"}, "x", "Consistent", "Low", "Urgent", path=str(tmp_path / "f.csv"))
