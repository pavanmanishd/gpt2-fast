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

NEAR_TIE_GAP = 0.05


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


def output_paths(args):
    suffix = "" if args.model == "gpt2" else f"-{args.model}"
    base = Path(args.out_dir) if args.out_dir else Path("reference")
    out = Path(args.out) if args.out else base / f"expected{suffix}.json"
    logits_dir = Path(args.logits_dir) if args.logits_dir else base / f"expected{suffix}"
    return out, logits_dir


def main():
    parser = argparse.ArgumentParser(description="Build the GPT-2 greedy reference outputs.")
    parser.add_argument("--model", default="gpt2", help="model name, for example gpt2-medium")
    parser.add_argument("--model-dir", default=None, help="defaults to models/<model>")
    parser.add_argument("--prompts", default="bench/prompts.json")
    parser.add_argument("--out", default=None, help="defaults to reference/expected[-<model>].json")
    parser.add_argument(
        "--logits-dir", default=None, help="defaults to reference/expected[-<model>]/"
    )
    parser.add_argument("--out-dir", default=None, help="base directory for the default outputs")
    args = parser.parse_args()

    model_dir = args.model_dir or f"models/{args.model}"
    out, logits_dir = output_paths(args)

    prompts = json.loads(Path(args.prompts).read_text())
    max_new_tokens = int(prompts["max_new_tokens"])
    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    torch.set_num_threads(threads)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    model = load_model(model_dir)

    logits_dir.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    near_ties = []
    for prompt in prompts["prompts"]:
        prompt_ids = tokenizer.encode(prompt["text"])
        outcome = greedy(model, prompt_ids, max_new_tokens)

        logits_path = logits_dir / f"{prompt['id']}.first_logits.f32"
        outcome["first_logits"].cpu().numpy().astype("<f4", copy=True).tofile(logits_path)

        generated_text = tokenizer.decode(outcome["generated_ids"])
        min_gap = round(outcome["min_gap"], 6)
        near_tie = outcome["min_gap"] < NEAR_TIE_GAP
        print(
            f"{prompt['id']}: {len(prompt_ids)} prompt tokens, min_gap {min_gap:.4f} "
            f"at step {outcome['min_gap_step']}",
            file=sys.stderr,
        )
        if near_tie:
            if args.model == "gpt2":
                failures.append(prompt["id"])
            else:
                near_ties.append(prompt["id"])
                print(
                    f"warning: {prompt['id']}: min_gap {min_gap:.4f} is below {NEAR_TIE_GAP}; "
                    "marking it near_tie, a greedy mismatch on this prompt will not fail the session",
                    file=sys.stderr,
                )

        entry = {
            "id": prompt["id"],
            "prompt_ids": prompt_ids,
            "generated_ids": outcome["generated_ids"],
            "generated_text": generated_text,
            "min_gap": min_gap,
            "min_gap_step": outcome["min_gap_step"],
            "first_logits": logits_path.as_posix(),
        }
        if near_tie:
            entry["near_tie"] = True
        results.append(entry)

    expected = {
        "model": args.model,
        "generator": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": f"{platform.system().lower()}-{platform.machine()}",
        },
        "max_new_tokens": max_new_tokens,
        "prompts": results,
    }
    out.write_text(json.dumps(expected, indent=2) + "\n")
    print(f"expected written to {out}", file=sys.stderr)

    if failures:
        print(f"prompts below the 0.05 minimum gap: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    if near_ties:
        print(f"prompts marked near_tie: {', '.join(near_ties)}", file=sys.stderr)


if __name__ == "__main__":
    main()
