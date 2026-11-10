# N7: 8-bit weights

**Result:** `results/log.jsonl`, milestone `N7`. Raw output in `results/runs/N7/`, perplexity in `results/runs/N7/perplexity-*.json`. Code at tag `n7`. This is a quantized result: it is shown separately and judged by token agreement and perplexity, not exact match.

## What changed

N4-N6 showed that decoding is waiting on memory: every token reads all ~500 MB of f32 weights. N7 stores the weights as 8-bit integers instead, one byte per weight instead of four.

- **Format.** Weights are split into groups of 64 consecutive inputs. Each group gets one f32 scale = (largest absolute value in the group) / 127, and each weight becomes `round(w / scale)`, an integer from −127 to 127. Applied to the four matrices in every layer and to the token embedding, which doubles as the final scoring matrix. Biases, layer norm and position embeddings stay f32.
- **Kernels.** The NEON kernels load int8 weights, widen them to f32 in registers, multiply-add as before, and apply each group's scale once at the end of the group. Activations stay f32.
- Quantization happens when the model loads (or ahead of time with `bench quantize`, for the browser). The f32 path is untouched and still bit-identical to N6.

## Results

5 sessions, 4 threads, median of session medians.

| | N6 (f32) | N7 (int8) | change |
|---|---|---|---|
| decode tok/s | 132.6 | **254.6** | 1.92× |
| prefill tok/s | 1,107 | 961 | 0.87× |
| time to first token (64-token prompt) | 56 ms | 65 ms | |
| peak memory | 560 MB | **207 MB** | −63% |
| perplexity, Moby-Dick passage (1,005 tokens) | 76.83 | 78.00 | +1.5% |
| tokens matching the f32 reference | 640 / 640 | 297 / 640 | 46% |

PyTorch's perplexity on the same passage is 76.84; the f32 engine is within 0.009 of it.

Token agreement per prompt: two prompts (p032, p128) match all 128 tokens; p064 matches the first 35, then drifts; p016 drifts at token 2 and p256 at token 0. Once one greedy choice differs, everything after it is a different continuation, so "46% of tokens" overstates how different the model is. Perplexity is the better measure of quality, and it moved by 1.5%.

## Why decode got faster but prefill didn't

- **Decode** reads every weight once per token, so its speed tracks bytes moved. Weights went from 474 MB to 119 MB (plus 7 MB of scales), about 3.8× fewer bytes, and decode got 1.9× faster. It doesn't get the full 3.8×: in host tests, int8 on one thread was only 1.2-1.3× faster than f32, because widening int8 to f32 costs about five instructions per eight weights. One core can't do that conversion fast enough to use the bandwidth saved. With four cores sharing the work, the bandwidth saving shows.
- **Prefill** reuses each weight for every prompt token, so it was never waiting on memory, and the widening is now extra arithmetic: 13% slower.

## The first attempt lost too much quality

The first version used one scale per output column (all 768 or 3,072 inputs share a scale). Host measurements:

| scheme | perplexity | tokens matching |
|---|---|---|
| f32 | 76.83 | 640 / 640 |
| one scale per column | 81.13 (+5.6%) | 74 / 640 |
| one scale per 32 inputs | 78.00 | 307 / 640 |
| **one scale per 64 inputs** | **78.00** | **297 / 640** |
| one scale per 128 inputs | 78.26 | 213 / 640 |

To find the cause, each group of matrices was quantized on its own with everything else in f32:

| only this part in int8 | perplexity |
|---|---|
| token embedding / scoring matrix | 81.23 (+4.39) |
| attention input (all layers) | 76.65 |
| attention output | 76.85 |
| MLP up | 76.86 |
| MLP down | 76.82 |

The token embedding alone accounts for the whole loss; quantizing all four layer matrices together barely moves perplexity. Its rows don't have one huge outlier; the spread is moderate everywhere (largest value per row is 1-3× the 99.9th percentile), and smaller groups track those variations better. Groups of 64 were picked over 32: same perplexity, 5% faster, half the scale memory.

## What didn't work

- One scale per output column: +5.6% perplexity, as above.
- Wider tiles in the int8 kernels: 16 columns per block is fastest for decode (91 tok/s single thread, against 84 at 32 and 66 at 64 columns). For prompts the grouped kernel had to drop from 16 to 8 columns to fit the extra accumulators in registers, which costs about 7% of prefill.
- The flame graph couldn't be recorded in the same run: the profiling script stopped on the expected token mismatches. It was fixed to accept quantized runs and the profile was recorded afterwards from commit `0ef867e`, whose int8 code is the same as `n7`.

## Post draft

> Step 7: 8-bit weights. Decoding reads every weight for every token, so store them in 1 byte instead of 4: 132 → 255 tokens/s, memory 560 → 207 MB. The cost: perplexity +1.5%. The first try (one scale per column) cost +5.6%. Quantizing one matrix at a time found the culprit: the token embedding alone. A scale per 64 weights fixed most of it. Shown separately from the exact-match results, because it no longer gives the same tokens.
