"""HTTP scoring service.

    uvicorn api:app --port 8000
    SIA_BACKEND=rubric uvicorn api:app        # choose a backend (default: best available)

POST /audit with {"tickets": [{"ticket_id": "T1", "description": "...", "priority": "Low"}]}
"""

import json
import os
from functools import lru_cache
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sia import PRIORITY_LEVELS
from sia.audit import add_attributions, audit_frame
from sia.calibration import urgent_threshold
from sia.models import load_model, load_resolution_reference

app = FastAPI(title="Support Integrity Auditor", version="2.0")


class Ticket(BaseModel):
    ticket_id: str
    description: str = Field(min_length=1)
    priority: Literal["Low", "Medium", "High", "Critical"]
    subject: Optional[str] = None
    resolution_time_hours: Optional[float] = Field(default=None, ge=0)


class AuditRequest(BaseModel):
    tickets: list[Ticket] = Field(min_length=1, max_length=1000)
    explain: bool = Field(default=False, description="Add word-level evidence where no rubric cue explains a flag")


class Verdict(BaseModel):
    ticket_id: str
    mismatch_type: str
    assigned_priority: str
    inferred_severity: str
    p_high_or_critical: float
    confidence: float
    needs_review: bool
    dossier: Optional[dict] = None


@lru_cache(maxsize=1)
def model():
    return load_model(os.environ.get("SIA_BACKEND", "auto"))


@app.get("/health")
def health():
    m = model()
    return {"status": "ok", "backend": m.name, "urgent_threshold": urgent_threshold(), "levels": PRIORITY_LEVELS}


@app.post("/audit", response_model=list[Verdict])
def audit(req: AuditRequest):
    df = pd.DataFrame([{
        "Ticket_ID": t.ticket_id, "Ticket_Description": t.description, "Priority_Level": t.priority,
        "Ticket_Subject": t.subject, "Resolution_Time_Hours": t.resolution_time_hours,
    } for t in req.tickets])
    try:
        result = audit_frame(df, model(), load_resolution_reference())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if req.explain:
        result = add_attributions(result, model())
    return [Verdict(
        ticket_id=str(r.Ticket_ID), mismatch_type=r.mismatch_type, assigned_priority=r.Priority_Level,
        inferred_severity=r.inferred_severity, p_high_or_critical=float(r.p_high_or_critical),
        confidence=float(r.confidence), needs_review=bool(r.needs_review),
        dossier=json.loads(r.dossier) if isinstance(r.dossier, str) else None,
    ) for r in result.itertuples()]
