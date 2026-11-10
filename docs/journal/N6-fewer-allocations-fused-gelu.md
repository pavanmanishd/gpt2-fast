# N6: fewer allocations, GELU fused into the matrix multiply

**Result:** `results/log.jsonl`, milestone `N6`. Raw output in `results/runs/N6/`. Code at tag `n6`.

## What changed

Stop moving data that doesn't need to move. Two changes survived measurement:

1. **No memory allocation while generating.** Before, every token asked the allocator for fresh buffers: activations, the attention input, the 3,072-wide MLP hidden layer, the logits, and per-worker scratch space for attention scores. N6 allocates one workspace per generation and reuses it for every token, and the thread pool reuses its job list. Allocations per decoded token at 4 threads went from 184 (632 KB) to **0**, counted with an allocator that tallies every call.
2. **GELU applied as the result is stored.** The MLP's first matrix multiply produces a 3,072-wide hidden layer, which then went through GELU in a separate pass. Now the kernel applies bias and GELU in the same step that writes each block of results, so the hidden layer is written once instead of written, read and written again.

Same arithmetic in the same order, so the logits are bit-identical to N5 at 1 and 4 threads.

## Results

5 sessions, median of session medians, 4 threads.

| | N5 | N6 | change |
|---|---|---|---|
| decode tok/s | 125.95 | **132.57** | +5% |
| prefill tok/s | 822 | **1,107** | +35% |
| time to first token (64-token prompt) | 77 ms | **56 ms** | −27% |

All 640 tokens match. Session medians: decode 130.4-136.3, prefill 1,060-1,131.

With this step gpt2-fast passes PyTorch on prompt processing too: 1,107 against 1,010 tokens per second, and 56 against 63 ms to the first token. Decode was already ahead (132.6 against 48.4).

## Why it got faster

- **Prefill, +35%.** On the 256-token prompt the MLP hidden layer is 256 × 3,072 floats, 3 MB per layer, larger than the core's caches. The separate GELU pass read and wrote all of it again in every one of the 12 layers. Fusing it removed that pass. In host tests the GELU fusion alone gave +26% at 4 threads (995 → 1,252 tok/s) and nothing measurable at 1 thread; we didn't measure why the gain only shows with 4 threads.
- **Decode, +5%.** At one token per step, the buffers are small and reallocating them was cheap. The flame graph shows less time spent waiting at thread handoffs: the worker spin loop went from 8.5% to 4.9% of samples and the caller's wait in `Pool::run` from 3.9% to 1.9%. Removing allocations from inside the workers probably accounts for that, since 36 of the 184 allocations per token happened on worker threads, but this wasn't isolated.
- The flame graph (`results/runs/N6/flame.svg`) is now dominated by weight loads: 44% of samples are on NEON load instructions inside the matrix kernels. Decode is mostly waiting for weights to arrive from memory.

## What didn't work

- **Fusing the residual add into the projection kernels** (write `x += W·h` directly instead of a temporary then an add): no decode gain, prefill 3-4% slower at 1 thread. Dropped.
- **Writing keys and values straight into the KV cache** from the attention projection: no measurable change. Dropped.
- **Moving layer norm inside the parallel region**: layer norm is 0.9 µs per call, 22 µs per token, under 0.3% of the time, and fusing it would have made every worker repeat it. Not done.
- Passing the epilogue choice as a runtime value inside the kernel cost 5% at 1-thread decode; making it a compile-time parameter removed the cost.

## Post draft

> Step 6: stop shuffling memory. Zero heap allocations per token (was 184), and GELU now runs as the matrix multiply writes its output instead of in a second pass over a 3 MB buffer. Same bits out. Prompt processing +35% (now faster than PyTorch: 1,107 vs 1,010 tok/s), decode +5% to 133 tok/s. The profile says what's left: 44% of the time is waiting for weights to arrive from memory.
