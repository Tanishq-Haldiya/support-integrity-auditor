"""Export trained transformer backends to int8 ONNX and check they still agree with torch.

    python export_onnx.py                        # every trained transformer backend
    python export_onnx.py --backend embedding    # just one

Writes models/<backend>/model.onnx: int8 when quantisation changes no verdict on 1,000 tickets, else fp32. Inference then runs on onnxruntime with no
torch import (set SIA_RUNTIME=torch to switch back).
"""

import json
import os
import shutil
import sys
import time

os.environ["SIA_RUNTIME"] = "torch"  # load the torch models as the reference

import numpy as np
import pandas as pd
import torch

from sia import URGENT_THRESHOLD
from sia.models import (DISTILBERT_DIR, EMBEDDING_DIR, DistilBertSeverityModel, EmbeddingSeverityModel)
from sia.onnx_runtime import ONNX_FILE

THRESHOLDS = [0.25, 0.5]  # the cost-based threshold and the plain one


def export(module, input_names, output_name, path, with_token_types):
    dummy = {"input_ids": torch.ones(2, 16, dtype=torch.long), "attention_mask": torch.ones(2, 16, dtype=torch.long)}
    if with_token_types:
        dummy["token_type_ids"] = torch.zeros(2, 16, dtype=torch.long)
    axes = {n: {0: "batch", 1: "seq"} for n in input_names}
    axes[output_name] = {0: "batch"} if output_name == "logits" else {0: "batch", 1: "seq"}
    torch.onnx.export(module, tuple(dummy[n] for n in input_names), path, input_names=input_names,
                      output_names=[output_name], dynamic_axes=axes, opset_version=17)


class _Logits(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


class _Hidden(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids).last_hidden_state


def main():
    import argparse
    from onnxruntime.quantization import QuantType, quantize_dynamic

    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["distilbert", "embedding"])
    only = ap.parse_args().backend

    sample = pd.read_csv("data/customer_support_tickets.csv").sample(1000, random_state=0)
    texts, subjects = sample["Ticket_Description"].tolist(), sample["Ticket_Subject"].tolist()
    jobs = []
    if os.path.exists(os.path.join(DISTILBERT_DIR, "config.json")):
        jobs.append(("distilbert", DISTILBERT_DIR, DistilBertSeverityModel))
    if os.path.exists(os.path.join(EMBEDDING_DIR, "classifier.joblib")):
        jobs.append(("embedding", EMBEDDING_DIR, EmbeddingSeverityModel))
    jobs = [j for j in jobs if only in (None, j[0])]
    if not jobs:
        sys.exit("No trained transformer backends found. Run train.py first.")

    for name, path, cls in jobs:
        ref = cls.load(path)
        fp32 = os.path.join(path, "model.fp32.onnx")
        if name == "distilbert":
            export(_Logits(ref.model).eval(), ["input_ids", "attention_mask"], "logits", fp32, False)
        else:
            export(_Hidden(ref.encoder).eval(), ["input_ids", "attention_mask", "token_type_ids"],
                   "last_hidden_state", fp32, True)
        int8 = os.path.join(path, "model.int8.tmp.onnx")
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)

        t0 = time.time()
        p_ref = ref.predict_proba(texts, subjects)
        t_ref = time.time() - t0
        u_ref = p_ref[:, URGENT_THRESHOLD:].sum(1)

        def score(candidate):
            shutil.copyfile(candidate, os.path.join(path, ONNX_FILE))
            os.environ["SIA_RUNTIME"] = "onnx"
            m = cls.load(path)
            t0 = time.time()
            p = m.predict_proba(texts, subjects)
            secs = time.time() - t0
            os.environ["SIA_RUNTIME"] = "torch"
            u = p[:, URGENT_THRESHOLD:].sum(1)
            flips = int(sum(((u_ref >= t) != (u >= t)).sum() for t in THRESHOLDS))
            return p, secs, flips

        # Keep int8 only if it never changes a verdict at any threshold we might use; otherwise fp32.
        p_onnx, t_onnx, flips = score(int8)
        precision = "int8"
        if flips:
            print(f"{name}: int8 changed {flips} verdict(s) across thresholds {THRESHOLDS}; keeping fp32")
            p_onnx, t_onnx, flips = score(fp32)
            precision = "fp32"
        os.remove(fp32)
        os.remove(int8)
        with open(os.path.join(path, "onnx_meta.json"), "w") as f:
            json.dump({"precision": precision, "checked_tickets": len(texts), "thresholds": THRESHOLDS,
                       "verdict_flips": flips}, f, indent=2)

        size = os.path.getsize(os.path.join(path, ONNX_FILE)) / 1e6
        torch_size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
                         if f.endswith(".safetensors")) / 1e6
        print(f"{name}: {precision}, {torch_size:.0f} MB -> {size:.0f} MB | {len(texts) / t_ref:.0f} -> "
              f"{len(texts) / t_onnx:.0f} tickets/s | same level {(p_ref.argmax(1) == p_onnx.argmax(1)).mean():.1%} | "
              f"verdict flips {flips} | max prob diff {np.abs(p_ref - p_onnx).max():.3f}")

if __name__ == "__main__":
    main()
