# N8: bigger models

**Result:** `results/log.jsonl`, milestones `N8-<model>-f32`, `N8-<model>-int8` and `N8-<model>-torch` for gpt2-medium (355M), gpt2-large (774M) and gpt2-xl (1.5B). Raw output in `results/runs/N8-*/`. Chart: `results/charts/sizes.png`.

## What this step is

No code changed. The same engine (N7 kernels with N6's f32 path) and PyTorch ran on the three larger GPT-2 models, to see which optimizations still hold up when the weights no longer fit in any cache.

| model | layers | width | parameters | f32 weights | int8 weights |
|---|---|---|---|---|---|
| gpt2 | 12 | 768 | 124M | 0.5 GB | 0.13 GB |
| gpt2-medium | 24 | 1,024 | 355M | 1.4 GB | 0.36 GB |
| gpt2-large | 36 | 1,280 | 774M | 3.1 GB | 0.78 GB |
| gpt2-xl | 48 | 1,600 | 1.5B | 6.2 GB | 1.6 GB |

## Correctness on bigger models

The five prompts were chosen so GPT-2 124M never has a near-tie between its top two candidates. The larger models do: 5 of 5 prompts on medium, 4 of 5 on large and xl have at least one step where the top two scores are within 0.05 (on xl's 64-token prompt, 0.0003). The bench flags those prompts and reports a mismatch on them separately instead of failing the run. In practice the f32 engine matched **every** token on every model anyway, and its perplexity on the Moby-Dick passage is within 0.007 of PyTorch's on all four sizes.

## Results

4 cores, 5 sessions each, median of session medians.

| model | engine | decode tok/s | prompt tok/s | peak memory | perplexity | tokens matching |
|---|---|---|---|---|---|---|
| 124M | PyTorch | 48.4 | 1,010 | 1,236 MB | 76.84 | 640 / 640 |
| | gpt2-fast f32 | 132.6 | 1,107 | 560 MB | 76.83 | 640 / 640 |
| | gpt2-fast int8 | 254.6 | 961 | 207 MB | 78.00 | 297 / 640 |
| 355M | PyTorch | 23.1 | 406 | 3,018 MB | 51.70 | 640 / 640 |
| | gpt2-fast f32 | 29.6 | 123 | 1,580 MB | 51.69 | 640 / 640 |
| | gpt2-fast int8 | 100.9 | 298 | 563 MB | 52.93 | 60 / 640 |
| 774M | PyTorch | 12.5 | 199 | 6,241 MB | 43.06 | 640 / 640 |
| | gpt2-fast f32 | 16.0 | 72 | 3,336 MB | 43.05 | 640 / 640 |
| | gpt2-fast int8 | 43.9 | 137 | 1,159 MB | 43.06 | 640 / 640 |
| 1.5B | PyTorch | did not fit | | | 27.89 | |
| | gpt2-fast f32 | 7.8 | 37 | 6,633 MB | 27.88 | 640 / 640 |
| | gpt2-fast int8 | 15.1 | 43 | 2,195 MB | 27.86 | 638 / 640 |

PyTorch perplexities are the reference values (`reference/perplexity-*.json`); xl's was computed on the Mac because PyTorch needed 8.2 GB.

## What held up and what didn't

**Decode still beats PyTorch, by less.** gpt2-fast f32 is 2.7× PyTorch on 124M but only 1.28× on medium and large. On 124M every weight matrix fits in the M4's caches between tokens; on the bigger models every token streams the weights from main memory. gpt2-fast's f32 decode moves about 42-50 GB/s on the bigger models (66 GB/s on 124M), and that rate, not arithmetic, sets the speed.

**int8 matters more as models grow.** It cuts the bytes per token by about 3.8×. Decode speedup over f32: 1.9× (124M), 3.4× (medium), 2.7× (large), 1.9× (xl). And quality: perplexity cost is +1.5% on 124M and +2.4% on medium, but +0.003% on large and −0.08% on xl (noise). On large, int8 matched all 640 tokens. Bigger models lose less to rounding their weights to 8 bits; we didn't measure why.

**Prompt processing falls behind PyTorch.** On 124M gpt2-fast beats PyTorch at processing prompts (1,107 against 1,010 tok/s). On medium PyTorch is 3.3× faster (406 against 123) and on large 2.8× (199 against 72). One clue: on medium, gpt2-fast's int8 prefill (298) is 2.4× faster than its f32 prefill (123), while on 124M int8 prefill was the slower one. That fits a memory problem rather than an arithmetic one: the prompt kernel from N4 re-reads the whole weight matrix for every block of 4 prompt rows. On 124M the biggest matrix (9.4 MB) fits in the M4 Pro's 16 MB L2 cache; medium's MLP matrices are 16.8 MB and don't, so for a 256-token prompt they would be read from main memory 64 times per layer. PyTorch's matrix library splits the work into cache-sized blocks. This is a hypothesis; it will be tested by adding that blocking.

**Memory.** gpt2-fast f32 uses about half of PyTorch's memory at every size (it keeps one copy of the weights and a fixed KV cache). PyTorch couldn't run gpt2-xl in the 8 GB container: its worker was killed during the first request. gpt2-fast runs gpt2-xl in f32 in 6.6 GB, and in int8 in 2.2 GB.

## Measurement notes

- gpt2-xl's later sessions were slower: f32 session medians 7.8, 7.8, 7.8, 7.9, 6.4; int8 16.0, 15.1, 15.3, 13.3, 10.7. No other container ran. The macOS power log (`pmset -g log`) shows the Mac was unplugged at 11:43, during the xl runs, so the last sessions ran on battery with the power management that comes with it. The median absorbs most of it; xl will be re-measured on AC power.
- The first attempt at this run stopped when the Mac went to sleep and the Docker VM hung. All N8 numbers come from the second run, which kept the machine awake.

## Post draft

> Step 8: bigger models. Same engine on GPT-2 355M, 774M and 1.5B. Two surprises. 8-bit weights get better with size: +1.5% perplexity on 124M, but no measurable loss on 774M and 1.5B, at 2-3x the speed. And our prompt processing falls behind PyTorch (3x slower on 355M) once a weight matrix stops fitting in the 16 MB cache. PyTorch couldn't load the 1.5B model in 8 GB; gpt2-fast runs it in 2.2 GB at 15 tokens/s.
