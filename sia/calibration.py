"""Confidence calibration and the cost-based urgency threshold.

Temperature scaling: p_calibrated = softmax(log p / T). T > 1 softens an
over-confident model. T is fitted on out-of-fold predictions from template
cross-validation, i.e. on tickets whose wording the model never saw, which is
the situation where over-confidence hurts.

Threshold: if missing an urgent ticket costs `cost_ratio` times more than a
wrong urgent flag, flag urgent when P(urgent) >= 1 / (1 + cost_ratio).
That rule is only meaningful once probabilities are calibrated.
"""

import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

from sia import URGENT_THRESHOLD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIBRATION_PATH = os.path.join(ROOT, "models", "calibration.json")
DEFAULT_COST_RATIO = 3.0


def apply_temperature(proba, temperature):
    logits = np.log(np.clip(proba, 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def urgent_nll(proba, y):
    """Log loss of P(High or Critical), the one probability the decision thresholds."""
    p = np.clip(proba[:, URGENT_THRESHOLD:].sum(axis=1), 1e-12, 1 - 1e-12)
    truth = np.asarray(y) >= URGENT_THRESHOLD
    return float(-np.mean(np.where(truth, np.log(p), np.log(1 - p))))


def fit_temperature(proba, y):
    """Temperature that makes P(High or Critical) best calibrated (minimum urgent log loss)."""
    res = minimize_scalar(lambda t: urgent_nll(apply_temperature(proba, t), y), bounds=(0.05, 50.0),
                          method="bounded")
    return float(res.x)


def expected_calibration_error(proba, y, bins=10):
    """ECE of the urgent-vs-not probability: the average gap between stated
    confidence and observed frequency, weighted by bin size."""
    p = proba[:, URGENT_THRESHOLD:].sum(axis=1)
    truth = (np.asarray(y) >= URGENT_THRESHOLD).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - truth[m].mean())
    return float(ece)


def threshold_from_cost(cost_ratio):
    return 1.0 / (1.0 + cost_ratio)


def load_config():
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return {}


def urgent_threshold():
    """Decision threshold on P(High or Critical). 0.5 until calibrate.py has run."""
    return float(load_config().get("urgent_threshold", 0.5))


def temperature_for(backend):
    return float(load_config().get("temperature", {}).get(backend, 1.0))


class CalibratedModel:
    """Wraps a backend and applies its fitted temperature."""

    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.name = model.name

    def predict_proba(self, descriptions, subjects=None, **kw):
        return apply_temperature(self.model.predict_proba(descriptions, subjects, **kw), self.temperature)
