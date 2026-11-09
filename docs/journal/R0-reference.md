# R0: the reference engines

**Results:** `results/log.jsonl`, milestones `R0-torch` and `R0-numpy`. Raw sessions in `results/runs/R0-torch/` and `results/runs/R0-numpy/`.

## What this step is

Before writing any Rust, we need two things: the correct answer, and a speed to beat.

GPT-2 picks the next word by scoring all 50,257 tokens in its vocabulary and, in greedy mode, taking the highest score. Run that 128 times and you get a 128-token continuation. Greedy decoding has no randomness, so the same model on the same prompt must always produce the same 128 tokens. That makes correctness checkable: every engine in this project must reproduce the reference tokens exactly, for all five prompts.

The reference is Hugging Face `transformers` on PyTorch, in 32-bit floats on the CPU. The second engine is a from-scratch NumPy version of GPT-2 (`reference/gpt2_numpy.py`, 137 lines), which shows how the model works without a framework.

## The prompts

Five public-domain passages of 16, 32, 64, 128 and 255 tokens (`bench/prompts.json`): Thoreau's *Walden* (twice, at two lengths), *The Communist Manifesto*, *The Federalist No. 1* and *Pride and Prejudice*.

They were chosen by a rule, not by taste. Later versions of the engine will add numbers up in a different order (tiling, SIMD, threads). Floating-point addition isn't associative, so that changes results in the fifth or sixth digit. If the reference ever had two candidate tokens with almost the same score, a perfectly correct engine could pick the other one and drift from there. So `reference/select_prompts.py` kept only passages where, at every one of the 128 steps, the winning token beats the runner-up by at least 0.05 in raw score (logit). The closest call across all prompts is 0.053, on the 16-token prompt. The engines so far differ from the reference by about 0.0004, so there is plenty of room.

## How it is measured

Every engine runs as a separate process that reads token ids on stdin and prints each generated token on stdout the moment it is chosen. The benchmark (`crates/bench`) does all the timing on its own clock, so PyTorch, NumPy and Rust are timed the same way.

- Container: Linux on an Apple M4 Pro (OrbStack), limited to 4 CPU cores and 8 GB.
- Per prompt: 1 check run, 1 warm-up, 5 timed runs of 128 tokens. Every run's tokens are checked.
- 5 sessions, each a fresh container. Reported numbers are the median of the five session medians.
- Decode speed is tokens per second after the first token. Prefill speed is prompt tokens divided by time to first token.

## Results

| | decode tok/s | prefill tok/s | first token | peak memory |
|---|---|---|---|---|
| PyTorch (transformers 4.46.3, torch 2.5.1) | 48.4 | 1010 | 63 ms | 1236 MB |
| NumPy from scratch (numpy 2.1.3, OpenBLAS) | 61.4 | 1167 | 53 ms | 1070 MB |

Both match the reference on every token of every run. Session medians for decode ranged from 45.2 to 50.4 tok/s (PyTorch) and 55.1 to 61.7 (NumPy).

## What the profile shows

`results/runs/R0-torch/flame.svg` and `results/runs/R0-numpy/flame.svg` were recorded on the 64-token prompt, generating all 128 tokens.

- **NumPy** spends 66% of its CPU time in `sgemv_n`, OpenBLAS's matrix-times-vector routine. During decode the model processes one token at a time, so every weight matrix is multiplied by a single vector, and that is exactly the operation NumPy ends up calling. Another 19% is OpenBLAS threads waiting for work.
- **PyTorch** has 56% of its samples inside functions that copy or reorder matrices: oneDNN's `jit_uni_reorder` (23% self time) and OpenBLAS's `sgemm_itcopy` (21%), which packs a matrix into the layout its multiply kernel wants. The multiply kernels themselves (`sgemm_kernel`, `a64_ffhybrid_fp32_mla_6x16`) are about 14%. So the framework that is supposed to be the fast one spends more time rearranging weights than multiplying them on this model. We didn't measure why it reorders every call; it is probably because a 124M model on one sequence is too small for the per-call setup to pay off.

## What went wrong along the way

- PyTorch's first run decoded at 7.6 tok/s. Inside the container `os.cpu_count()` reports the host's 14 cores while only 4 are usable, and PyTorch started 14 threads on 4 cores. Setting the thread count to the cores the process may actually use (`os.sched_getaffinity`) took it to 48 tok/s. NumPy's OpenBLAS already read the affinity mask correctly; the worker now sets it explicitly anyway.
- The first prompt-selection run took over 30 minutes for the same reason.
- An early R0 session overlapped with another benchmark on the same machine and was thrown away. The runs above started only after the machine had been free of other containers for 3 minutes, and a sampler logged no overlap during them.

## Post draft

> Before optimizing anything: the bar. GPT-2 (124M) on 4 CPU cores. PyTorch: 48 tokens/s. A 137-line NumPy version: 61 tokens/s. The profile says why: PyTorch spends over half its time copying and reordering weight matrices, not multiplying them. Next: the same model in Rust with zero dependencies.
