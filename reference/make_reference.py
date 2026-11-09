#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(model_dir):
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.eval()
    return model


def argmax_lowest(logits):
    maximum = torch.max(logits)
    return int(torch.nonzero(logits == maximum, as_tuple=False)[0])


def greedy(model, prompt_ids, max_new_tokens):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        output = model(input_ids, use_cache=True)
        logits = output.logits[0, -1]
        first_logits = logits.clone()
        past_key_values = output.past_key_values

        generated_ids = []
        min_gap = float("inf")
        min_gap_step = 0
        for step in range(max_new_tokens):
            top_two = torch.topk(logits, 2).values
            gap = float(top_two[0] - top_two[1])
            if gap < min_gap:
                min_gap = gap
                min_gap_step = step

            token = argmax_lowest(logits)
            generated_ids.append(token)

            if step + 1 < max_new_tokens:
                output = model(
                    torch.tensor([[token]], dtype=torch.long),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = output.logits[0, -1]
                past_key_values = output.past_key_values

    return {
        "generated_ids": generated_ids,
        "first_logits": first_logits,
        "min_gap": min_gap,
        "min_gap_step": min_gap_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the GPT-2 greedy reference outputs.")
    parser.add_argument("--model-dir", default="models/gpt2")
    parser.add_argument("--prompts", default="bench/prompts.json")
    parser.add_argument("--out", default="reference/expected.json")
    parser.add_argument("--logits-dir", default="reference/expected")
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text())
    max_new_tokens = int(prompts["max_new_tokens"])
    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    torch.set_num_threads(threads)

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_dir)
    model = load_model(args.model_dir)

    logits_dir = Path(args.logits_dir)
    logits_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    for prompt in prompts["prompts"]:
        prompt_ids = tokenizer.encode(prompt["text"])
        outcome = greedy(model, prompt_ids, max_new_tokens)

        logits_path = logits_dir / f"{prompt['id']}.first_logits.f32"
        outcome["first_logits"].cpu().numpy().astype("<f4", copy=True).tofile(logits_path)

        generated_text = tokenizer.decode(outcome["generated_ids"])
        min_gap = round(outcome["min_gap"], 6)
        print(
            f"{prompt['id']}: {len(prompt_ids)} prompt tokens, min_gap {min_gap:.4f} "
            f"at step {outcome['min_gap_step']}",
            file=sys.stderr,
        )
        if outcome["min_gap"] < 0.05:
            failures.append(prompt["id"])

        results.append(
            {
                "id": prompt["id"],
                "prompt_ids": prompt_ids,
                "generated_ids": outcome["generated_ids"],
                "generated_text": generated_text,
                "min_gap": min_gap,
                "min_gap_step": outcome["min_gap_step"],
                "first_logits": logits_path.as_posix(),
            }
        )

    expected = {
        "model": prompts["model"],
        "generator": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": f"{platform.system().lower()}-{platform.machine()}",
        },
        "max_new_tokens": max_new_tokens,
        "prompts": results,
    }
    Path(args.out).write_text(json.dumps(expected, indent=2) + "\n")

    if failures:
        print(f"prompts below the 0.05 minimum gap: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
