# N1: naive Rust

**Result:** `results/log.jsonl`, milestone `N1`. Raw output in `results/runs/N1/`. Code at tag `n1`.

## What changed

The first version of the engine, written from scratch in Rust with no machine-learning or maths libraries. It has four parts:

- **A weight loader** for the safetensors file: an 8-byte header length, a JSON header saying where each tensor lives, then raw 32-bit floats.
- **A tokenizer**: GPT-2's byte-level BPE. Text is split into chunks (words, numbers, punctuation, spaces), each chunk becomes bytes, and the 50,000 merge rules from `merges.txt` glue bytes back into tokens. It was checked against Hugging Face's tokenizer on 13,048 strings in many scripts, with zero differences.
- **The model**: 12 transformer layers, each with layer norm, attention over 12 heads, and a two-layer MLP with GELU. About 300 lines.
- **Greedy generation**: score all 50,257 tokens, take the best, append it, repeat.

Everything is deliberately the simplest thing that works:

- **No memory between tokens.** To produce token 129, the engine runs the whole model over all 128 previous tokens plus the prompt again, from scratch.
- **Textbook matrix multiply.** For each output row, for each output column, add up `x[i][k] * w[k][j]` over k. The weights are stored input-major (`w[k][j]`), so the inner loop jumps 2,304 to 3,072 floats (9 to 12 KB) through memory on every step.
- One thread, 32-bit floats, no vector instructions written by hand.

## Results

One session, one timed run per prompt, no warm-up. The full protocol (warm-up plus 5 runs, 5 sessions) would have taken about 80 hours; this run took 110 minutes.

| prompt | prompt tokens | time to first token | decode tok/s | time for 128 tokens |
|---|---|---|---|---|
| p016 | 16 | 1.1 s | 0.199 | 10.7 min |
| p032 | 32 | 2.0 s | 0.164 | 12.9 min |
| p064 | 64 | 3.9 s | 0.123 | 17.2 min |
| p128 | 128 | 7.8 s | 0.082 | 25.9 min |
| p256 | 255 | 15.9 s | 0.049 | 43.2 min |

Median across prompts: **0.12 tokens per second** decode, 16 prompt tokens per second prefill, 3.9 s to first token, 499 MB peak memory.

Every one of the 640 generated tokens matches the PyTorch reference. The largest difference in the first-token scores is 0.00014, against a 0.053 safety margin.

For comparison, PyTorch on the same 4-core container does 48 tokens per second, about 390 times faster.

## Why it is slow

Two causes, both visible in the numbers:

1. **Decode speed falls as the text grows.** It goes from 0.199 tok/s with a 16-token prompt to 0.049 with a 255-token prompt. Without a memory of past tokens, each new token costs a full pass over everything so far, so generating 128 tokens from the long prompt means processing roughly 41,000 token-positions instead of 382. N2 removes this.
2. **The matrix multiply is the whole program.** The flame graph (`results/runs/N1/flame.svg`, 64-token prompt) puts 98.2% of CPU time inside one function, `linear`, the triple loop. Attention, layer norm, GELU and the final scoring step together are under 2%. The 64-token forward pass does about 5.4 billion multiply-adds (11 billion floating-point operations) in 3.9 s, roughly 2.8 GFLOP/s. We think the cause is the memory jump in the inner loop: each step lands on a new cache line. N3 tests that by changing only the loop order.

## What went wrong along the way

- The hand-written pre-tokenizer first treated any whitespace as the optional leading space in GPT-2's pattern. Fuzzing found one mismatch (`"\n'ta"`); the pattern only allows a literal space there. Fixed before any measurement.
- Nothing failed in the measured run.

## Post draft

> Step 1: GPT-2 in plain Rust, no libraries. Correct on every token, and 0.12 tokens per second: one word every 8 seconds, about 390x slower than PyTorch. 98% of the time sits in one triple loop. Two reasons: the engine re-reads the whole text for every new word, and the loop walks memory the wrong way. Next: fix the first one.
