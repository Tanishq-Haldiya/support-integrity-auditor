"""Audit a CSV of tickets for priority mismatch.

    python predict.py tickets.csv [output.csv] [--backend auto|distilbert|tfidf|rubric]

Required columns: Ticket_ID, Ticket_Description, Priority_Level.
Optional: Ticket_Subject, Resolution_Time_Hours (used as handling evidence only).
"""

import argparse
import sys

import pandas as pd

from sia.audit import add_attributions, audit_frame
from sia.models import available_backends, load_model, load_resolution_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="data/customer_support_tickets.csv")
    ap.add_argument("output", nargs="?", default="predictions_output.csv")
    ap.add_argument("--backend", default="auto", choices=["auto", "hybrid", "embedding", "distilbert", "tfidf", "rubric"])
    ap.add_argument("--limit", type=int, help="Only audit the first N rows")
    ap.add_argument("--explain", action="store_true",
                    help="Add word-level occlusion evidence to flagged tickets that no rubric cue explains")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    if args.backend != "auto" and args.backend not in available_backends():
        sys.exit(f"Backend '{args.backend}' is not trained yet. Run: python train.py --backend {args.backend}")

    model = load_model(args.backend)
    print(f"Auditing {len(df)} tickets with the '{model.name}' backend...")
    try:
        result = audit_frame(df, model, load_resolution_reference(),
                             progress=lambda done, total: print(f"\r  {done}/{total}", end="", flush=True))
    except ValueError as e:
        sys.exit(f"\nError: {e}")
    print()
    if args.explain:
        print("Computing word-level evidence for model-only flags...")
        result = add_attributions(result, model)

    result.to_csv(args.output, index=False)
    print(result["mismatch_type"].value_counts().to_string())
    print(f"{int(result['needs_review'].sum())} ticket(s) flagged for manual review (model and rubric disagree).")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
