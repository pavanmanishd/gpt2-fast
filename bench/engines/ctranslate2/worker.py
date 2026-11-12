#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def load_vocabulary(model_dir):
    path = model_dir / "vocabulary.json"
    if not path.exists():
        path = model_dir / "shared_vocabulary.json"
    if not path.exists():
        raise SystemExit(f"ct2-worker: no vocabulary file in {model_dir}")
    vocabulary = json.loads(path.read_text())
    if isinstance(vocabulary, list):
        return vocabulary
    tokens = [None] * len(vocabulary)
    for token, index in vocabulary.items():
        tokens[index] = token
    return tokens


def main():
    parser = argparse.ArgumentParser(description="GPT-2 worker backed by CTranslate2.")
    parser.add_argument("--model-dir", default="models/gpt2-ct2/float32")
    parser.add_argument("--compute-type", default="default")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    threads = args.threads
    if threads <= 0:
        threads = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))

    import ctranslate2
    import numpy as np

    ctranslate2.set_log_level(logging.WARNING)

    model_dir = Path(args.model_dir)
    tokens = load_vocabulary(model_dir)

    started = time.perf_counter()
    generator = ctranslate2.Generator(
        str(model_dir),
        device="cpu",
        compute_type=args.compute_type,
        inter_threads=1,
        intra_threads=threads,
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    print(
        f"ctranslate2 worker: {threads} threads, {generator.compute_type}, "
        f"supported {sorted(ctranslate2.get_supported_compute_types('cpu'))}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps({"ready": True, "load_ms": load_ms}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        prompt_ids = [int(token) for token in request["ids"]]
        max_new_tokens = int(request["max_new_tokens"])
        return_logits = bool(request.get("return_logits", False))
        prompt = [tokens[token] for token in prompt_ids]

        if return_logits:
            view = generator.forward_batch([prompt_ids])
            rows = np.asarray(view).reshape(-1, view.shape[-1])
            print(json.dumps({"logits": rows[-1].tolist()}), flush=True)

        emitted = 0

        def callback(step):
            nonlocal emitted
            emitted += 1
            os.write(1, b'{"token": %d}\n' % int(step.token_id))
            return False

        generator.generate_batch(
            [prompt],
            max_length=max_new_tokens,
            sampling_topk=1,
            include_prompt_in_result=False,
            end_token=[],
            callback=callback,
        )
        if emitted != max_new_tokens:
            print(
                f"ct2-worker: emitted {emitted} of {max_new_tokens} tokens",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
