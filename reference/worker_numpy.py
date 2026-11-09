#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

# BLAS fixes its thread pool at import time, so these have to be set before numpy loads.
threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
os.environ.setdefault("OMP_NUM_THREADS", str(threads))

import numpy as np

from gpt2_numpy import GPT2


def main():
    parser = argparse.ArgumentParser(description="GPT-2 worker backed by the NumPy implementation.")
    parser.add_argument("--model-dir", default="models/gpt2")
    args = parser.parse_args()

    print(f"numpy threads: {threads}", file=sys.stderr, flush=True)

    started = time.perf_counter()
    model = GPT2(args.model_dir)
    load_ms = (time.perf_counter() - started) * 1000.0
    print(json.dumps({"ready": True, "load_ms": load_ms}, separators=(",", ":")), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        prompt_ids = request["ids"]
        max_new_tokens = int(request["max_new_tokens"])
        return_logits = bool(request.get("return_logits", False))

        logits, caches = model.prefill(prompt_ids)
        if return_logits:
            print(json.dumps({"logits": logits.tolist()}, separators=(",", ":")), flush=True)

        for index in range(max_new_tokens):
            token = int(np.argmax(logits))
            print(json.dumps({"token": token}, separators=(",", ":")), flush=True)
            if index + 1 < max_new_tokens:
                logits, caches = model.step(token, len(prompt_ids) + index, caches)

        print(json.dumps({"done": True}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
