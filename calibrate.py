"""Fit per-backend temperature and set the cost-based urgency threshold.

    python calibrate.py [--cost-ratio 3] [--tag aug07]

Reads reports/oof_<backend>[_tag].npz written by
`train.py --split template-cv`, and writes models/calibration.json.
"""

import argparse
import glob
import json
import os

import numpy as np

from sia.calibration import (CALIBRATION_PATH, DEFAULT_COST_RATIO, apply_temperature, expected_calibration_error,
                             fit_temperature, threshold_from_cost, urgent_nll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cost-ratio", type=float, default=DEFAULT_COST_RATIO,
                    help="How many times worse a missed urgent ticket is than a wrong urgent flag.")
    ap.add_argument("--tag", default="", help="Which template-cv run's out-of-fold predictions to use.")
    args = ap.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    config = {"cost_ratio": args.cost_ratio, "urgent_threshold": round(threshold_from_cost(args.cost_ratio), 4),
              "temperature": {}, "fit_on": {}, "diagnostics": {}}
    for path in sorted(glob.glob(os.path.join("reports", f"oof_*{suffix}.npz"))):
        backend = os.path.basename(path)[len("oof_"):-len(f"{suffix}.npz")]
        if not backend or "_" in backend:
            continue
        data = np.load(path)
        proba, y = data["proba"], data["y"]
        t = fit_temperature(proba, y)
        cal = apply_temperature(proba, t)
        config["temperature"][backend] = round(t, 4)
        config["fit_on"][backend] = os.path.basename(path)
        config["diagnostics"][backend] = {
            "n": int(len(y)),
            "nll_before": round(urgent_nll(proba, y), 4), "nll_after": round(urgent_nll(cal, y), 4),
            "ece_before": round(expected_calibration_error(proba, y), 4),
            "ece_after": round(expected_calibration_error(cal, y), 4),
        }
        print(f"{backend}: T={t:.3f}  {config['diagnostics'][backend]}")

    # The hybrid falls back on the learned model that generalised best to unseen templates
    # (template-CV mismatch macro-F1). Chosen from CV, never from a test set.
    cv = {}
    for path in glob.glob(os.path.join("reports", f"train_metrics_*_template-cv{suffix}.json")):
        m = json.load(open(path))
        cv[m["backend"]] = m["mismatch_macro_f1"]
    if cv:
        config["hybrid_base"] = max(cv, key=cv.get)
        config["hybrid_base_reason"] = {"template_cv_mismatch_macro_f1": cv}
        print(f"Hybrid fallback: {config['hybrid_base']} (template-CV mismatch macro-F1 {cv})")

    os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Urgent threshold {config['urgent_threshold']} (cost ratio {args.cost_ratio}). Wrote {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
