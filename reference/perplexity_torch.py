#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def main():
    parser = argparse.ArgumentParser(description="Measure transformers perplexity on a fixed text.")
    parser.add_argument("--model", default="gpt2", help="model name, for example gpt2-medium")
    parser.add_argument("--model-dir", default=None, help="defaults to models/<model>")
    parser.add_argument("--text", default="reference/perplexity.txt")
    parser.add_argument("--out", default=None, help="defaults to reference/perplexity[-<model>].json")
    args = parser.parse_args()

    model_dir = args.model_dir or f"models/{args.model}"
    suffix = "" if args.model == "gpt2" else f"-{args.model}"
    out = args.out or f"reference/perplexity{suffix}.json"

    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    torch.set_num_threads(threads)

    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir)
    ids = tokenizer.encode(Path(args.text).read_text())
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        logits = model(input_ids, use_cache=True).logits[0, :-1]
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = input_ids[0, 1:]
        negative_log_likelihood = -logprobs.gather(1, targets[:, None]).mean().item()
    perplexity = math.exp(negative_log_likelihood)

    result = {
        "text": args.text,
        "tokens": len(ids),
        "perplexity": perplexity,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
