# N10: output-major weights

**Result:** `results/log.jsonl`, milestones `N10` (f32), `N10-int8`, and `N10-threads1..3` for the scaling curve. Raw output in `results/runs/N10*/`. Code at tag `n10`.

## What changed

The race in N9 ended with a clear loss on decode speed: llama.cpp read weights at about 155 GB/s where gpt2-fast managed about 66 GB/s, on the same four cores. Its profile was one function, a dot product of a contiguous row of weights with the input. N10 tests whether that layout is the whole difference.

GPT-2's checkpoint stores each matrix input-major: row `k` holds the weights from input `k` to every output. Since N4, decoding one token walked that layout in strips: for 32 outputs at a time, step through all inputs, reading 32 weights (128 bytes) from each row. Rows are 9 to 12 KB apart, so every step lands somewhere new.

N10 transposes the four matrices of every layer when the model loads, so each output's weights are one contiguous row (output-major), and changes the kernels to match:

- **Decode** is now one dot product per output: a straight read of 768 to 3,072 floats. Two outputs are computed at once, each with four vector accumulators. Threads split the outputs.
- **Prompts** use a 4 × 4 tile: four prompt rows against four weight rows, 16 dot products sharing their loads.
- **int8** keeps groups of 64 inputs, now along each output's row, with the scales stored per row.

The numbers change slightly because sums are added in a different order: logits move by at most 0.00011 and all 640 reference tokens still match.

## Evidence before the rewrite

Before touching the engine, a microbenchmark compared the two layouts on a single-token multiply, cycling through 128 MB of copies so the weights really come from memory (host, GB/s of weights read):

| matrix | cores | input-major strips (N6) | output-major rows | |
|---|---|---|---|---|
| 768 → 3,072 | 1 | 38.0 | 56.1 | 1.5× |
| 768 → 3,072 | 4 | 30.4 | 159.7 | 5.3× |
| 3,072 → 768 | 1 | 14.4 | 54.9 | 3.8× |
| 3,072 → 768 | 4 | 40.6 | 160.7 | 4.0× |

Two things stand out. The strip pattern is slower on one core, and it does not scale: on the first shape four cores read *less* per second than one. Output-major rows scale 2.9× from one core to four.

## Results

5 sessions, 4 threads, median of session medians. These were collected while the Mac was in normal use, so each session is kept only if a 6-second control run of the N6 engine, right before and right after it, decodes at 127 tok/s or more (132.6 on a quiet machine). Control values are saved next to the sessions in `controls.tsv`; a control run of N6 in the same batch gave 128.3 tok/s.

| | N6 (f32) | N10 (f32) | change |
|---|---|---|---|
| decode tok/s | 132.6 | **306.5** | 2.31× |
| prefill tok/s | 1,107 | 1,265 | +14% |
| time to first token (64-token prompt) | 56 ms | 46 ms | |
| peak memory | 560 MB | 553 MB | |

| | N7 (int8) | N10-int8 | change |
|---|---|---|---|
| decode tok/s | 254.6 | **410.2** | 1.61× |
| prefill tok/s | 961 | 976 | |

All 640 tokens still match in f32; int8 keeps the same 297 of 640 and its perplexity is unchanged (77.996 against 78.00).

On gpt2-medium the f32 decode goes from 29.6 tok/s (N8) to **110.2** (3.7×), and prompt processing from 123 to 368 tok/s.

gpt2-fast's f32 now decodes as fast as llama.cpp's f32 (300.7 tok/s, re-measured in the same batch; see N12).

## Why it got faster

Excluding model loading, the N6 flame graph had 46% of samples on vector load instructions and 26% in the matrix kernel's loop. In N10 (`results/runs/N10/flame.svg`) loads are 16%, and the dot-product kernel itself is the largest item (36%). The same weights now arrive fast enough that the arithmetic shows up again.

The scoring step (50,257 dot products) was already output-major in N6 and did not change; it now takes 19% of decode time, because everything around it got faster.

This also explains two earlier puzzles:

- **N4** concluded that decoding was limited by memory, because 42% of the profile sat on instructions loading weights. It was limited by how the weights were being read. The same memory delivers 2.3 times more bytes per second when read in straight lines.
- **N5** could not explain why three threads were no faster than two. The strip pattern does not scale with cores (table above). N10 scales steadily: 192.6 tok/s on 2 threads, 255.9 on 3 and 306.5 on 4 (the 1-thread run is still being collected).

## What it cost

- **Load time.** Transposing while loading takes the model from 0.24 s to 1.0 s to load (median of the five sessions). llama.cpp memory-maps its file in 0.04 s. Storing the file already transposed would remove this; not done yet.

## What didn't work

- Four outputs at a time for decode: fine on two shapes, but 111 GB/s instead of 161 on the 3,072 → 768 shape.
- Two outputs one after the other in the same loop: no gain. Interleaving them was worth 4% end to end.
- Prompt tiles of 4 × 8 and 8 × 4 ran out of vector registers (45-51 GFLOP/s against 79 for 4 × 4).
- int8 gains less than f32 (1.61× against 2.31×): its kernel is limited by converting int8 to float, not by memory. That is what N13 goes after.

## Post draft

> Step 10: steal llama.cpp's weight layout. Its profile was one loop: a dot product over a contiguous row of weights. Ours read 128 bytes from each of 768 rows spread 9 KB apart. Transpose the weights at load time so each output's weights are one row: decode 133 → 307 tokens/s, and our f32 now matches llama.cpp's f32. The earlier "we're limited by memory" conclusion was wrong: we were limited by how we read it.
