"""Torch-free inference with onnxruntime + the `tokenizers` library.

`python export_onnx.py` writes model.onnx next to each trained transformer.
When that file exists, the backends load through here instead of torch/transformers,
which cuts start-up from tens of seconds to about a second and shrinks the model
up to 4x when int8 is safe. Set SIA_RUNTIME=torch to force the torch path.
"""

import os

import numpy as np

ONNX_FILE = "model.onnx"


def onnx_enabled(model_dir):
    return os.environ.get("SIA_RUNTIME", "onnx") != "torch" and os.path.exists(os.path.join(model_dir, ONNX_FILE))


class OnnxEncoder:
    """Tokenise + run an exported transformer. Returns (first output, attention mask)."""

    def __init__(self, model_dir, max_length):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(os.path.join(model_dir, ONNX_FILE), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def __call__(self, texts):
        enc = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        return self.session.run(None, feeds)[0], mask


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
