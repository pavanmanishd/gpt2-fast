#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import torch
from transformers import GPT2LMHeadModel


def argmax_lowest(logits):
    maximum = torch.max(logits)
    return int(torch.nonzero(logits == maximum, as_tuple=False)[0])


def main():
    parser = argparse.ArgumentParser(description="GPT-2 worker backed by transformers and PyTorch.")
    parser.add_argument("--model-dir", default="models/gpt2")
    args = parser.parse_args()

    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    torch.set_num_threads(threads)
    print(f"torch threads: {threads}", file=sys.stderr, flush=True)

    started = time.perf_counter()
    model = GPT2LMHeadModel.from_pretrained(args.model_dir)
    model.eval()
    load_ms = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"ready": True, "load_ms": load_ms}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        prompt_ids = request["ids"]
        max_new_tokens = int(request["max_new_tokens"])
        return_logits = bool(request.get("return_logits", False))

        with torch.inference_mode():
            output = model(torch.tensor([prompt_ids], dtype=torch.long), use_cache=True)
            logits = output.logits[0, -1]
            past_key_values = output.past_key_values

            if return_logits:
                print(json.dumps({"logits": logits.tolist()}), flush=True)

            for index in range(max_new_tokens):
                token = argmax_lowest(logits)
                print(json.dumps({"token": token}), flush=True)
                if index + 1 < max_new_tokens:
                    output = model(
                        torch.tensor([[token]], dtype=torch.long),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits = output.logits[0, -1]
                    past_key_values = output.past_key_values

        print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
