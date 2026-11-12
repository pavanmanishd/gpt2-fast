#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import GPT2LMHeadModel


class GPT2Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = model.config.n_layer

    def forward(self, input_ids, attention_mask, *past):
        past_key_values = tuple((past[2 * i], past[2 * i + 1]) for i in range(self.layers))
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        flat = []
        for key, value in outputs.past_key_values:
            flat.append(key)
            flat.append(value)
        return (outputs.logits, *flat)


def last_logits(value):
    array = np.asarray(value)
    return array.reshape(-1, array.shape[-1])[-1]


def max_diff(left, right):
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def export_f32(model, out_path):
    config = model.config
    layers = config.n_layer
    heads = config.n_head
    head_dim = config.n_embd // heads
    past_len = 3
    seq_len = 2

    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.ones((1, past_len + seq_len), dtype=torch.long)
    past = []
    for _ in range(layers * 2):
        past.append(torch.zeros((1, heads, past_len, head_dim), dtype=torch.float32))

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {"input_ids": {1: "seq"}, "attention_mask": {1: "total"}, "logits": {1: "seq"}}
    for layer in range(layers):
        for kind in ("key", "value"):
            input_names.append(f"past_key_values.{layer}.{kind}")
            output_names.append(f"present.{layer}.{kind}")
            dynamic_axes[f"past_key_values.{layer}.{kind}"] = {2: "past"}
            dynamic_axes[f"present.{layer}.{kind}"] = {2: "total"}

    with torch.no_grad():
        torch.onnx.export(
            GPT2Step(model),
            (input_ids, attention_mask, *past),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    model.eval()
    return heads, head_dim


def check(path, model, prompt, heads, head_dim):
    model.eval()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    layers = (len(inputs) - 2) // 2

    def run(ids, past_values, past_len):
        feed = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, past_len + len(ids)), dtype=np.int64),
        }
        for layer in range(layers):
            for index, kind in enumerate(("key", "value")):
                feed[f"past_key_values.{layer}.{kind}"] = past_values[layer * 2 + index]
        return session.run(None, feed)

    with torch.no_grad():
        reference = model(torch.tensor([prompt], dtype=torch.long)).logits[0, -1].numpy()

    empty = [np.zeros((1, heads, 0, head_dim), dtype=np.float32) for _ in range(layers * 2)]
    results = run(prompt, empty, 0)
    logits = last_logits(results[0])
    prefill_diff = max_diff(logits, reference)

    first = int(np.argmax(logits))
    with torch.no_grad():
        extended = model(torch.tensor([prompt + [first]], dtype=torch.long)).logits[0, -1].numpy()
    results = run([first], list(results[1:]), len(prompt))
    step_logits = last_logits(results[0])
    step_diff = max_diff(step_logits, extended)

    size_mb = path.stat().st_size / 1e6
    print(
        f"{path.name}: {size_mb:.1f} MB, {len(inputs)} inputs, {len(outputs)} outputs, "
        f"prefill logit diff {prefill_diff:.3e}, cache step logit diff {step_diff:.3e}",
        file=sys.stderr,
    )
    if results[0].shape[-1] != reference.shape[-1]:
        raise SystemExit(f"{path}: vocabulary mismatch")
    if not np.isfinite(step_logits).all():
        raise SystemExit(f"{path}: non-finite logits")
    return prefill_diff, step_diff


def main():
    parser = argparse.ArgumentParser(description="Export GPT-2 with a past key/value cache to ONNX.")
    parser.add_argument("--model-dir", default="models/gpt2")
    parser.add_argument("--out-dir", default="models/gpt2-onnx")
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT2LMHeadModel.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()
    prompt = [1, 2, 3]

    f32_path = out_dir / "model.onnx"
    heads, head_dim = export_f32(model, f32_path)
    prefill_diff, step_diff = check(f32_path, model, prompt, heads, head_dim)

    shutil.copyfile(model_dir / "config.json", out_dir / "config.json")
    summary = {
        "source": str(model_dir),
        "f32": f32_path.name,
        "f32_prefill_logit_max_abs_diff": prefill_diff,
        "f32_cache_step_logit_max_abs_diff": step_diff,
        "layers": model.config.n_layer,
        "heads": heads,
        "head_dim": head_dim,
        "vocab": int(model.config.vocab_size),
    }

    if not args.skip_int8:
        int8_path = out_dir / "model-int8.onnx"
        quantize_dynamic(
            str(f32_path),
            str(int8_path),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        int8_prefill_diff, int8_step_diff = check(int8_path, model, prompt, heads, head_dim)
        summary["int8"] = int8_path.name
        summary["int8_prefill_logit_max_abs_diff"] = int8_prefill_diff
        summary["int8_cache_step_logit_max_abs_diff"] = int8_step_diff

    (out_dir / "export.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), file=sys.stderr)


if __name__ == "__main__":
    main()
