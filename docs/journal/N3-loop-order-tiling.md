# N3: loop order and tiling

**Result:** `results/log.jsonl`, milestone `N3`. Raw output in `results/runs/N3/`. Code at tag `n3`.

## What changed

Same maths, same additions in the same order, different path through memory.

GPT-2 stores each weight matrix input-major: row `k` holds the weights from input `k` to every output. The N1/N2 multiply computed one output at a time:

```
for each output column j:
    sum over k of x[k] * w[k][j]      ← walks down a column: jumps 9-12 KB per step
```

Memory reaches the CPU in 64-byte cache lines. Walking down a column uses 4 bytes of each line and throws the rest away. N3 swaps the two inner loops:

```
for each input k:
    a = x[k]
    for each output column j:
        y[j] += a * w[k][j]           ← walks along a row: every byte of every line is used
```

Each output still gets its additions in the same order (k = 0, 1, 2, …, bias last), so the result is bit-identical to N2. That was checked byte for byte on the logits of all five prompts.

On top of that, **tiling**: when processing the prompt, 8 input rows are handled together, so each weight row is loaded once and used 8 times while it is still in the fastest cache. The attention step got the same loop-order fix for its weighted sum of values.

## Results

5 sessions, median of session medians.

| | N2 | N3 | change |
|---|---|---|---|
| decode tok/s | 11.73 | **43.16** | 3.7× |
| prefill tok/s | 16.0 | **185.2** | 11.6× |
| time to first token (64-token prompt) | 4.08 s | 0.33 s | 12.3× |
| peak memory | 559 MB | 560 MB | |

All 640 tokens match the reference. N3 now decodes at 89% of PyTorch's speed (48.4 tok/s).

## Why it got faster

Microbenchmarks of the three matrix shapes in a GPT-2 layer, single thread, 256 rows (host, not the container):

| matrix | N2 loop order | swapped loops | + 8-row tiles |
|---|---|---|---|
| attention in, 768 → 2,304 | 3.0 GFLOP/s | 31.9 | 39.5 |
| MLP up, 768 → 3,072 | 2.4 | 31.9 | 37.4 |
| MLP down, 3,072 → 768 | 3.0 | 31.7 | 42.8 |

Swapping the loops is a 10× gain on its own; tiling adds another 20-25%.

**The tile size test** is the evidence that this is about cache. Tiles of 2 to 8 rows were all faster than no tiling. At 12 and 16 rows the MLP-up multiply got 36-39% slower than with 8-row tiles (44.0 and 44.7 ms against 32.3 ms): 12 or 16 output rows of 3,072 floats are 147-192 KB, more than the M4 core's 128 KB L1 data cache, so the weight row gets evicted between uses.

**A side effect worth stating plainly:** once the inner loop walks memory in a straight line, the Rust compiler turned it into vector instructions on its own. The release binary contains NEON `fmul.4s` and `fadd.4s` instructions in `linear`, four floats at a time. There is no SIMD in the source; that is LLVM's loop vectorizer. Part of N3's gain is therefore "vectorized by accident", which is only possible because of the new loop order. It does not fuse the multiply and the add into one instruction, because Rust doesn't let the compiler change rounding.

**Why decode gained less than prefill (3.7× vs 11.6×).** When decoding, each weight is used once per token, so tiling has nothing to reuse. The flame graph (`results/runs/N3/flame.svg`) also shows a new bottleneck: the final scoring step, a plain dot product of the hidden state with each of 50,257 vocabulary rows, is now 23.5% of the time (it was 6.5% in N2). That loop adds into one running sum, which the compiler may not vectorize without changing the result. The matrix multiply (`linear` and the inlined `mul`) is 61%.

## Noise in this run

Some individual runs were about half the speed of the rest: on the 255-token prompt, 4 of 25 runs decoded at 18-29 tok/s instead of about 40. The other runs in the same sessions were normal. Other development work (builds, a headless browser) was running on the host at the time. The medians absorb these runs: session medians ranged from 39.5 to 45.0 tok/s. A re-run of the same code on an idle machine (`results/runs/N3-quiet/`, 5 sessions) gave 43.29 tok/s decode and 190.4 tok/s prefill, within 3% of the numbers above, so they stand.

## What didn't work

- Tiles of 12 and 16 rows: slower, see above.
- Also splitting the columns into 1,024-wide blocks: prefill 5% faster, but decode 13% slower. Rejected, since decode is the headline.
- Tiles of 6 were about 1% faster than 8 in total but slower on two of three shapes; kept 8.

## Post draft

> Step 3: same maths, different order. The multiply loop was walking down columns of the weight matrix, using 4 bytes of every 64-byte cache line. Swap two loops and it reads memory in a straight line: 10x faster on its own, bit-identical output. Decode 11.7 → 43 tokens/s, prompt processing 12x faster. Bonus: the compiler started using vector instructions by itself. Tiles bigger than the 128 KB L1 cache made it up to 39% slower again, which is how we know it's the cache.
