# N4: hand-written NEON SIMD

**Result:** `results/log.jsonl`, milestone `N4`. Raw output in `results/runs/N4/`. Code at tag `n4`.

## What changed

The CPU core has 128-bit vector registers that hold four 32-bit floats, and instructions that work on all four at once. N4 writes those instructions by hand (Rust's `std::arch::aarch64` NEON intrinsics) in the three hot loops. Still one thread.

1. **Matrix multiply: keep results in registers.** N3's loop (vectorized by the compiler) loaded each output from memory, added to it, and stored it back for every input. N4's kernel keeps a 4 × 16 block of outputs in vector registers for the whole sum (1 × 32 when decoding a single token), reads the weights once, and writes each output once at the end. It also uses the fused multiply-add instruction (`fmla`), which computes `a + b × c` in one step with one rounding.
2. **Dot products with eight accumulators.** The final scoring step is 50,257 dot products of length 768, and attention has many short ones. A single running sum has to wait for each addition to finish before the next can start. N4 keeps eight independent 4-wide sums and adds them together at the end.
3. Other targets (x86, WebAssembly) keep the plain code; nothing there changed.

**This is the first change that alters the numbers.** Fused multiply-add rounds once instead of twice, and eight partial sums add in a different order than one. The logits moved by at most 0.00037 from N3, and all 640 reference tokens still match. The closest call in the reference is a 0.053 gap, about 140 times larger than the drift.

## Results

5 sessions, median of session medians, single thread.

| | N3 | N4 | change |
|---|---|---|---|
| decode tok/s | 43.16 | **61.32** | 1.42× |
| prefill tok/s | 185.2 | **363.4** | 1.96× |
| time to first token (64-token prompt) | 331 ms | 176 ms | |

Session medians for decode: 59.8-63.9. On one core, gpt2-fast now decodes faster than PyTorch on four (48.4) and matches the NumPy reference (61.4).

## Why it got faster

**The scoring step.** In N3, the flame graph put 49% of all samples inside `last_logits` (the 50,257 dot products), because one running sum forced every addition to wait for the previous one. In N4 it is 11%. In a host microbenchmark the scoring step went from 16.9 ms to 2.1 ms per token (8.1×), reading the 154 MB matrix at 74 GB/s instead of 9 GB/s.

**Prefill: compute-bound, so registers help.** With 256 prompt tokens each weight is used 256 times, so arithmetic matters. The register-blocked kernel reached 84-106 GFLOP/s on the three layer shapes in host microbenchmarks, against 37-43 for N3.

**Decode: memory-bound, so registers don't help.** Decoding one token uses each weight once. The single-row matrix multiply ran at the same speed as N3's compiler-vectorized loop (29-32 GFLOP/s in microbenchmarks). The flame graph agrees: 42% of samples sit on the vector load instructions that bring weights in from memory, and only 8% on the multiply-adds. Almost all of N4's decode gain comes from the scoring step.

## What didn't work

Register block shapes tried for prefill (rows × columns of outputs held in registers, GFLOP/s on the attention-input shape):

| shape | GFLOP/s | |
|---|---|---|
| 4 × 16 | 106 | kept |
| 8 × 8 | 111 | faster in isolation, 11% slower end to end |
| 6 × 16 | 94 | runs out of registers (24 accumulators + 6 inputs + 4 weights > 32) |
| 8 × 16 | 79 | runs out of registers |
| 4 × 32, 2 × 32, 4 × 8 | 60-75 | slower |

- 8 × 8 won the cache-hot microbenchmark but lost on the real workload, where each layer's 28 MB of weights streams in from memory and 4 × 16 reads longer runs of each weight row.
- Loading four vectors with one instruction (`vld1q_f32_x4`): 10% slower.
- Software prefetch hints: no gain, and slower on several shapes. Removed.
- Scoring four vocabulary rows at once: slower than one row with eight accumulators.
- Vectorizing layer norm: it is about 1% of prefill time. Skipped.

## Measurement note

N4 was first measured while other development work ran on the host (a Docker build and a headless browser). Those sessions gave 47.7-64.2 tok/s and a median of 53.2. The numbers above are from a re-run on an idle machine, which is what every milestone from N4 on uses.

## Post draft

> Step 4: SIMD by hand. Keep a 4×16 block of results in vector registers, use fused multiply-add, and give dot products eight running sums instead of one. Output now differs from the reference by at most 0.0004 and every token still matches. Prompt processing 2x faster. Decoding only 1.4x: the profile shows 42% of the time waiting for weights to arrive from memory. Faster arithmetic doesn't help when the bottleneck is memory. One core now beats PyTorch on four: 61 vs 48 tok/s.
