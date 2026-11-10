# N5: threads

**Result:** `results/log.jsonl`, milestone `N5` (4 threads), plus `N5-threads1`, `N5-threads2` and `N5-threads3` for the scaling curve. Raw output in `results/runs/N5*/`. Code at tag `n5`.

## What changed

The container has 4 cores and N4 used one. N5 splits each piece of work across all of them:

- **Matrix multiplies when decoding one token:** each thread computes a different slice of the output columns, cut on 32-column boundaries so every column goes through exactly the same kernel as before.
- **Matrix multiplies on the prompt:** each thread takes a different block of rows.
- **Scoring:** each thread scores a different slice of the 50,257 vocabulary rows.
- **Attention:** split by head when decoding, by position when processing the prompt.

Every output number is still computed by one thread with the same instructions in the same order, so the logits are bit-identical to N4 at 1, 2, 3 and 4 threads.

**The thread pool is written from scratch with the standard library.** Decoding one token crosses into parallel code about 49 times, roughly every 0.15 ms. Spawning threads each time (`std::thread::scope`) cost 22-28 µs per crossing, which would have added over 1 ms per token. N5 keeps three worker threads alive. The caller publishes a job by bumping a counter; the workers spin briefly, then sleep; a second counter tells the caller when every worker is done. One crossing costs about 0.25 µs. `GPT2_THREADS` sets the thread count, and with 1 thread no pool is created at all.

## Results

5 sessions per thread count, median of session medians.

| threads | decode tok/s | vs 1 thread | prefill tok/s | vs 1 thread |
|---|---|---|---|---|
| 1 | 61.6 | 1.00× | 364 | 1.00× |
| 2 | 102.3 | 1.66× | 582 | 1.60× |
| 3 | 103.7 | 1.68× | 686 | 1.89× |
| 4 | **126.0** | **2.05×** | **822** | **2.26×** |

All 640 tokens match at every thread count. Against N4 (61.3 tok/s, single thread), N5 at 1 thread is the same speed (61.6), so the pool costs nothing when unused.

gpt2-fast now decodes 2.6 times faster than PyTorch (48.4) and twice as fast as NumPy (61.4), on the same 4 cores.

## Why it doesn't scale 4×

Decoding one token means reading every weight once: about 500 MB of floats per token. At 126 tokens per second that is 63 GB/s; one thread reads about 31 GB/s. More cores issue more memory requests, but they share the path to memory, and adding cores gives less each time. The flame graph agrees: 38.5% of samples sit on vector loads of weights.

Prompt processing reuses each weight across many tokens, so it is less tied to memory and scales better (2.26× at 4 threads).

What the threads cost: in the flame graph (`results/runs/N5/flame.svg`), workers spinning while waiting for work are 8.5% of samples and the caller waiting for workers to finish is 3.9%.

## Not explained yet: 3 threads ≈ 2 threads

Decode at 3 threads (103.7) is barely faster than at 2 (102.3), while prefill improves at every step. The work splits evenly at 3 threads: every matrix width (768, 2,304, 3,072) divides into equal 32-column slices. One guess is that the slowest thread sets the pace at every handoff, and that macOS may be running one of the VM's virtual cores on a slower efficiency core. That is untested; it will be checked by running 3 threads on different sets of the container's cores.

## What didn't work

- Spinning 16 times longer before sleeping: 15% faster at 2 threads on the host, but with all 14 host cores it fell to 19 tok/s, because idle workers on efficiency cores held everyone up at each handoff. Reverted to the shorter spin.
- The first row split for prompts sent single rows through the one-row kernel, which moved some columns between the vector and scalar paths and changed one bit in one value. Every row now goes through the multi-row kernel, and an exact-equality test guards it.

## Post draft

> Step 5: four cores. A tiny thread pool written from scratch (0.25 µs per handoff, against 25 µs to spawn threads) splits every matrix by columns, so each number is computed exactly as before: bit-identical output at any thread count. 61 → 126 tokens/s, 2.6x PyTorch. Not 4x: decoding reads 500 MB of weights per token, and at 63 GB/s the cores are mostly waiting on memory together.
