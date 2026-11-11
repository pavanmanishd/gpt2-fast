# N12: rematch against llama.cpp

**Result:** `results/log.jsonl`, milestones `N12-llamacpp-f32`, `N12-llamacpp-f16`, `N12-llamacpp-q8_0`, compared with `N11` and `N11-int8` from the same batch. Raw output in `results/runs/N12-llamacpp-*/`. Chart: `results/charts/race-rematch.png`.

## What this is

N9 lost decode speed to llama.cpp by 2.3× and showed why. N10 and N11 acted on it. This re-runs llama.cpp (same build, `b11912`, same adapter, same GGUF files) in the same evening and under the same session gate as N11, so the comparison is between runs made under the same conditions.

## Results

GPT-2 124M, 4 cores, 5 gated sessions each, median of session medians.

| engine | decode tok/s | prompt tok/s | first token | memory | tokens matching reference |
|---|---|---|---|---|---|
| **gpt2-fast f32 (N11)** | **323.6** | **1,397** | **46 ms** | 553 MB | 640 / 640 |
| llama.cpp f32 | 300.7 | 895 | 71 ms | 543 MB | 640 / 640 |
| llama.cpp f16 | 386.4 | 839 | 76 ms | 309 MB | 640 / 640 |
| gpt2-fast int8 (N11-int8) | 413.0 | 995 | 61 ms | 208 MB | 297 / 640 |
| llama.cpp q8_0 | 679.6 | 4,316 | 15 ms | 281 MB | 182 / 640 |

- **f32:** gpt2-fast now decodes 1.08× faster than llama.cpp and processes prompts 1.56× faster. In N9 it was 2.3× slower at decoding.
- **f16:** llama.cpp's 16-bit weights read half the bytes and stay ahead; gpt2-fast has no 16-bit mode.
- **8-bit:** llama.cpp's q8_0 is still 1.65× faster at decoding and 4.3× faster at prompts. gpt2-fast's int8 widens every weight to a float before multiplying; q8_0 rounds the activations too and uses the CPU's integer dot-product instruction. N13 does the same.

## A correction to N9

N9 reported llama.cpp q8_0 at 541 tok/s. Tonight it measured 679.6, with session medians between 676 and 693. The N9 q8_0 sessions ran around 07:30 that morning, when the Mac briefly lost power and went to sleep, and the f16 sessions just before them were already slowing (331-400 tok/s against a steady 382-399 tonight). The 541 was most likely depressed by that. The N9 journal and log line are left as measured, with a note pointing here; 679.6 is the number to use for llama.cpp q8_0.

## Post draft

> Rematch, same 4 cores, same night, same gate: gpt2-fast f32 323 tok/s vs llama.cpp f32 301, and prompts 1,397 vs 895. Two steps did it: steal their weight layout (N10), block prompts for the cache (N11). llama.cpp's 8-bit q8_0 still wins (680 tok/s), because it multiplies 8-bit by 8-bit with integer dot products. That's next.
