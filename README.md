# Support Integrity Auditor (SIA)

SIA audits customer-support tickets for **priority mismatch**: tickets whose assigned priority does not match how severe they actually are. Every flag comes with evidence taken from the ticket itself.

- **Hidden Crisis**: an urgent issue (outage, lock-out, fraud) filed as Low or Medium.
- **False Alarm**: a routine request (pricing question, invoice) filed as High or Critical.

## How it works

1. A phrase-level rubric reads each ticket and infers its severity (Low to Critical), producing training labels where none exist.
2. A classifier learns severity from the ticket text alone, so it generalises to wording the rubric has no pattern for.
3. The inferred severity is compared with the assigned priority; a mismatch on the urgent-vs-routine line is flagged.
4. Each flag gets an evidence dossier: the quoted phrases or the words the model relied on, the model's confidence, and how the ticket was handled.
5. Reviewers confirm or correct flags in a web app, and their decisions become labels for the next round of training.

On a sealed set of 80 tickets written in wording the models never saw, SIA catches **80.6% of hidden crises**, against 19.4% for a keyword baseline.

## Quick start

```bash
pip install -r requirements.txt
python train.py --augment 0.7            # train the models
streamlit run app.py                     # open the review app
```

The app, a batch CLI (`predict.py`) and an HTTP API (`api.py`) are included.
