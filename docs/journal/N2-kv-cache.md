# N2: KV cache

**Result:** `results/log.jsonl`, milestone `N2`. Raw output in `results/runs/N2/`. Code at tag `n2`.

## What changed

N1 had no memory between tokens: to produce each new token it ran all 12 layers over the whole text again.

Attention is the only part of the model where one position looks at the others. Each layer turns every position into a query, a key and a value. A position's output is a weighted mix of the values of itself and every earlier position, with weights taken from how well its query matches their keys. Keys and values for a position depend only on that position and the ones before it, so once computed they never change.

N2 stores them. Generation now runs the prompt through the model once, keeps each layer's keys and values (the "KV cache"), and then runs the model for one new position per token, looking up past positions in the cache instead of recomputing them.

Nothing else changed: same triple-loop matrix multiply, same arithmetic in the same order. The logits are bit-identical to N1's: 804,112 values compared byte for byte.

## Results

5 sessions × 5 timed runs per prompt, median of session medians.

| | N1 | N2 |
|---|---|---|
| decode tok/s | 0.123 | **11.73** |
| prefill tok/s | 16.1 | 16.0 |
| time to first token | 3.94 s | 4.08 s |
| peak memory | 499 MB | 559 MB |

Decode is **95× faster**. All 640 tokens match the reference in every run. Session medians ranged from 11.45 to 11.78 tok/s.

## Why it got faster, and by exactly how much

Per prompt, decode speed in N1 fell as the text grew. In N2 it is flat:

| prompt | N1 decode tok/s | N2 decode tok/s |
|---|---|---|
| p016 | 0.199 | 11.97 |
| p064 | 0.123 | 11.58 |
| p256 | 0.049 | 11.67 |

The size of the jump can be predicted. Almost all the work is matrix multiplies, which cost one multiply-add per weight per position. The 12 layers hold about 85M weights; the final scoring step (50,257 × 768) about 39M, and both versions run it for the last position only. For the 64-token prompt, generation averages about 128 positions of text, so:

- N1 per token: 85M × 128 + 39M ≈ 10.9 billion multiply-adds
- N2 per token: 85M × 1 + 39M ≈ 124 million multiply-adds

That is 88× less work. The measured speedup on that prompt is 94×. Work done per second barely moved, from about 2.7 to 2.9 GFLOP/s, so the speedup comes from doing less work, not from doing it faster.

The flame graph (`results/runs/N2/flame.svg`) shows the same picture: 88% of the time is still in `linear`, the triple loop, and the scoring step `last_logits` is 6.5%.

## What it did not change

- **Prefill.** The prompt still has to go through the model once, with the same slow loop, so time to first token is unchanged (4.1 s for 64 tokens).
- **Memory.** Peak memory went up 60 MB. The cache holds 12 layers × 1,024 positions × 768 numbers for keys and the same for values, about 75 MB, allocated up front.

## What went wrong along the way

Nothing failed. The cache was written to keep the exact operation order of N1, and a test checks that generating with the cache produces the same tokens as recomputing from scratch.

## Post draft

> Step 2: remember instead of recompute. GPT-2's attention keys and values for past words never change, so keep them (the "KV cache"). Same maths, bit-identical output, 95x faster: 0.12 → 11.7 tokens/s. The predicted saving on the 64-token prompt was 88x less work; measured 94x. Still 4x slower than PyTorch. Next: the loop that does 88% of the work reads memory in the wrong order.
