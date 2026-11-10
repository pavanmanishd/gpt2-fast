# N9: the race against llama.cpp

**Result:** `results/log.jsonl`, milestones `N9-llamacpp-f32`, `N9-llamacpp-f16`, `N9-llamacpp-q8_0`, compared with `R0-torch`, `R0-numpy`, `N6` and `N7`. Raw output in `results/runs/N9-llamacpp-*/`. Chart: `results/charts/race.png`.

## Setup

llama.cpp is the standard for running language models on CPUs. It was built from release `b11912` inside the same bench image, for `armv8.6-a` with the dot-product and int8 matrix-multiply extensions the M4 has, and given the same 4-core container.

To time it the same way as everything else, a 200-line C++ program (`bench/llamacpp/worker.cpp`) links llama.cpp's library and speaks the bench's protocol: it takes token ids, runs the prompt in one batch, then picks each next token with its own argmax (same tie rule as ours), printing each token the moment it's chosen. No llama.cpp sampler is involved, so any difference in tokens comes from the maths.

GPT-2 was converted with llama.cpp's own converter to GGUF in three formats: f32, f16 and q8_0 (8-bit weights in blocks of 32, each with an f16 scale).

## Results

GPT-2 124M, 4 cores, 5 sessions each, median of session medians.

| engine | decode tok/s | prompt tok/s | first token | memory | tokens matching reference |
|---|---|---|---|---|---|
| PyTorch | 48.4 | 1,010 | 63 ms | 1,236 MB | 640 / 640 |
| NumPy | 61.4 | 1,167 | 53 ms | 1,070 MB | 640 / 640 |
| **gpt2-fast f32 (N6)** | **132.6** | **1,107** | **56 ms** | 560 MB | 640 / 640 |
| llama.cpp f32 | **310.0** | 895 | 71 ms | 543 MB | 640 / 640 |
| llama.cpp f16 | 393.3 | 839 | 76 ms | 309 MB | 640 / 640 |
| **gpt2-fast int8 (N7)** | **254.6** | 961 | 65 ms | 207 MB | 297 / 640 |
| llama.cpp q8_0 | **541.1** | **3,041** | **22 ms** | 281 MB | 182 / 640 |

**The honest summary:** gpt2-fast beats PyTorch and NumPy everywhere, and beats llama.cpp's f32 and f16 at processing the prompt. llama.cpp decodes 2.3 times faster than gpt2-fast in f32 and 2.1 times faster at 8 bits. Decode is the headline number, so llama.cpp wins this race.

**Correctness.** llama.cpp f32 and f16 produce all 640 reference tokens. Its f32 logits differ from PyTorch's by up to 0.028, above the 0.01 tolerance this project uses, so by the project's own rule it does not count as an exact match; gpt2-fast's f32 differs by at most 0.0004. At 8 bits, gpt2-fast keeps more of the reference: 297 matching tokens against 182 for q8_0, which diverges within the first 5 tokens on four of five prompts. Perplexity for q8_0 wasn't measured, because the bench's perplexity command only drives our engine.

## Why llama.cpp decodes faster

The llama.cpp flame graph (`results/runs/N9-llamacpp-f32/flame.svg`) puts 77% of its time in one function, `ggml_vec_dot_f32`: a dot product of a contiguous row of weights with the input vector. GGUF stores each output's weights as one contiguous row (output-major). For a single token, every output is one long, straight read through memory, and four threads each take a block of whole rows.

gpt2-fast keeps GPT-2's original input-major layout. Since N3 its multiply walks along rows of that layout, which is the right order for processing a prompt. For decode, each thread takes a slice of output columns, so it reads a short piece of every input row (768 bytes to 3 KB) and jumps to the next. At 310 tok/s llama.cpp moves about 155 GB/s of f32 weights; gpt2-fast at 132.6 tok/s moves about 66 GB/s.

That points at the next step: store the weights output-major and decode with dot products. We have not tested it yet, so this is a hypothesis, not an explanation.

For prompts the picture flips: with many tokens at once, gpt2-fast's register-blocked kernel (from N4) beats llama.cpp's f32 path, 1,107 against 895 tok/s. llama.cpp's q8_0 prompt speed (3,041 tok/s) comes from also quantizing the activations to 8 bits and using the CPU's int8 dot-product instructions; gpt2-fast's int8 keeps activations in f32.

## Measurement notes

- llama.cpp's decode session medians: f32 305-313, f16 331-400, q8_0 536-559 tok/s.
- llama.cpp ran with 4 threads (its default from the CPU affinity mask) and OpenMP, which it uses for its thread pool; 7.5% of its profile is OpenMP synchronization.
- The GPT-2 converter in `b11912` stops on the `attn.bias` buffers stored in Hugging Face's checkpoint; the conversion script drops those two buffer names first. They're constant causal masks, not weights.

## Post draft

> Step 9: the final race, GPT-2 124M on 4 CPU cores. gpt2-fast: 133 tok/s f32, 255 int8. PyTorch: 48. llama.cpp: 310 f32, 541 q8_0. We win on prompt processing in f32 (1,107 vs 895) and lose decode by 2.3x. The profile shows why: llama.cpp stores each output's weights as one contiguous row and spends 77% of its time in one dot-product loop, reading memory at ~155 GB/s to our ~66. So the race isn't over: next, the same layout trick.
