import os

import pytest

pytest.importorskip("fastapi")
os.environ["SIA_BACKEND"] = "rubric"  # no trained model needed

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["backend"] == "rubric"


def test_audit_flags_hidden_crisis_with_grounded_evidence():
    text = "Hi Support, the application crashes every time I open settings."
    r = client.post("/audit", json={"tickets": [{"ticket_id": "T1", "description": text, "priority": "Low"}]})
    assert r.status_code == 200
    v = r.json()[0]
    assert v["mismatch_type"] == "Hidden Crisis"
    assert all(e["matched_text"] in text for e in v["dossier"]["feature_evidence"] if e["signal"] == "text_cue")


def test_consistent_ticket_has_no_dossier():
    r = client.post("/audit", json={"tickets": [{"ticket_id": "T2", "description": "Where is your office?",
                                                 "priority": "Low"}]})
    assert r.json()[0]["mismatch_type"] == "Consistent" and r.json()[0]["dossier"] is None


def test_invalid_priority_is_rejected():
    r = client.post("/audit", json={"tickets": [{"ticket_id": "T3", "description": "x", "priority": "Urgent"}]})
    assert r.status_code == 422
