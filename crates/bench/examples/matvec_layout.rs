//! Single-token matrix-vector throughput for the two weight layouts.
//!
//! `input-major` is W[in][out], what GPT-2's Conv1D tensors use and what the
//! engine stored before this measurement. `output-major` is W[out][in], one
//! contiguous row per output. Both kernels are the shapes the engine uses:
//! column slices with a short piece of every input row, or whole output rows
//! as dot products. Run with `cargo run --release -p gpt2_bench --example
//! matvec_layout`.

#![allow(clippy::needless_range_loop)]

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("matvec_layout: only the aarch64 kernels are implemented");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    if std::env::args().nth(1).as_deref() == Some("tiles") {
        tiles::run();
    } else {
        layout_bench::run();
    }
}

#[cfg(target_arch = "aarch64")]
mod layout_bench {
    use std::arch::aarch64::*;
    use std::hint::black_box;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};

    const GROUP: usize = 64;
    const FLOATS_PER_ROW: usize = 8;
    const TARGET_BYTES: usize = 2 << 30;
    const WORKING_SET: usize = 128 << 20;
    const REPEATS: usize = 5;

    pub fn run() {
        println!(
            "{:>5} {:>7} {:>6} {:>9} {:>9} {:>7}",
            "dtype", "shape", "cores", "in-major", "out-major", "ratio"
        );
        for (n_in, n_out) in [(768, 3072), (3072, 768), (50257, 768)] {
            for threads in [1, 4] {
                let input = best_of(|| f32_input_major_throughput(n_in, n_out, threads));
                let output = best_of(|| f32_output_major_throughput(n_in, n_out, threads));
                println!(
                    "{:>5} {:>3}x{:<3} {:>6} {:>7.1} {:>9.1} {:>6.2}x",
                    "f32",
                    n_in,
                    n_out,
                    threads,
                    input,
                    output,
                    output / input
                );
            }
        }
        for (n_in, n_out) in [(768, 3072), (3072, 768), (50257, 768)] {
            for threads in [1, 4] {
                let input = best_of(|| int8_input_major_throughput(n_in, n_out, threads));
                let output = best_of(|| int8_output_major_throughput(n_in, n_out, threads));
                println!(
                    "{:>5} {:>3}x{:<3} {:>6} {:>7.1} {:>9.1} {:>6.2}x",
                    "int8",
                    n_in,
                    n_out,
                    threads,
                    input,
                    output,
                    output / input
                );
            }
        }
    }

    fn best_of(mut measure: impl FnMut() -> f64) -> f64 {
        (0..REPEATS).map(|_| measure()).fold(0.0, f64::max)
    }

    fn bytes(n_in: usize, n_out: usize, element: usize) -> usize {
        n_in * n_out * element
    }

    // Enough copies that one timed pass streams well past any cache.
    fn copies(total: usize) -> usize {
        (WORKING_SET / total).clamp(1, 32)
    }

    fn iterations(working_set: usize) -> usize {
        (TARGET_BYTES / working_set).max(1)
    }

    fn f32_values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + 11) % 251) as f32 / 128.0 - 0.98)
            .collect()
    }

    fn i8_values(count: usize) -> Vec<i8> {
        (0..count)
            .map(|i| ((i * 53 + 7) % 255) as i32 as i8)
            .collect()
    }

    fn f32_input_major_throughput(n_in: usize, n_out: usize, threads: usize) -> f64 {
        let copies = copies(bytes(n_in, n_out, 4));
        let data: Vec<Vec<f32>> = (0..copies).map(|_| f32_values(n_in * n_out)).collect();
        let x = black_box(f32_values(n_in));
        let ranges = column_ranges(n_out, threads);
        let passes = iterations(copies * bytes(n_in, n_out, 4));

        let elapsed = timed_threads(threads, n_out, |index, y| {
            let (first, end) = ranges[index];
            for _ in 0..passes {
                for weight in &data {
                    // SAFETY: `weight` holds `n_in * n_out` values and the column
                    // range lies inside one row of `n_out` values.
                    unsafe {
                        f32_columns(&x, weight, n_in, n_out, first, end, &mut y[first..end]);
                    }
                }
            }
        });
        rate(passes * copies * bytes(n_in, n_out, 4), elapsed)
    }

    fn f32_output_major_throughput(n_in: usize, n_out: usize, threads: usize) -> f64 {
        let copies = copies(bytes(n_in, n_out, 4));
        let data: Vec<Vec<f32>> = (0..copies).map(|_| f32_values(n_in * n_out)).collect();
        let x = black_box(f32_values(n_in));
        let ranges = row_ranges(n_out, threads);
        let passes = iterations(copies * bytes(n_in, n_out, 4));

        let elapsed = timed_threads(threads, n_out, |index, y| {
            let (first, end) = ranges[index];
            for _ in 0..passes {
                for weight in &data {
                    // SAFETY: `weight` holds `n_out` rows of `n_in` values and the
                    // row range lies inside them.
                    unsafe {
                        f32_rows(&x, weight, n_in, first, end, &mut y[first..end]);
                    }
                }
            }
        });
        rate(passes * copies * bytes(n_in, n_out, 4), elapsed)
    }

    fn int8_input_major_throughput(n_in: usize, n_out: usize, threads: usize) -> f64 {
        let copies = copies(bytes(n_in, n_out, 1));
        let data: Vec<Vec<i8>> = (0..copies).map(|_| i8_values(n_in * n_out)).collect();
        let groups = n_in.div_ceil(GROUP);
        let scales: Vec<Vec<f32>> = (0..copies)
            .map(|_| {
                (0..groups * n_out)
                    .map(|i| (i % 13 + 1) as f32 / 4096.0)
                    .collect()
            })
            .collect();
        let x = black_box(f32_values(n_in));
        let ranges = column_ranges(n_out, threads);
        let passes = iterations(copies * bytes(n_in, n_out, 1));

        let elapsed = timed_threads(threads, n_out, |index, y| {
            let (first, end) = ranges[index];
            for _ in 0..passes {
                for pass in 0..copies {
                    // SAFETY: `data[pass]` holds `n_in * n_out` values, `scales[pass]`
                    // holds `groups * n_out`, and the column range is inside a row.
                    unsafe {
                        int8_columns(
                            &x,
                            &data[pass],
                            &scales[pass],
                            n_in,
                            n_out,
                            first,
                            end,
                            &mut y[first..end],
                        );
                    }
                }
            }
        });
        rate(passes * copies * bytes(n_in, n_out, 1), elapsed)
    }

    fn int8_output_major_throughput(n_in: usize, n_out: usize, threads: usize) -> f64 {
        let copies = copies(bytes(n_in, n_out, 1));
        let data: Vec<Vec<i8>> = (0..copies).map(|_| i8_values(n_in * n_out)).collect();
        let groups = n_in.div_ceil(GROUP);
        let scales: Vec<Vec<f32>> = (0..copies)
            .map(|_| {
                (0..n_out * groups)
                    .map(|i| (i % 13 + 1) as f32 / 4096.0)
                    .collect()
            })
            .collect();
        let x = black_box(f32_values(n_in));
        let ranges = row_ranges(n_out, threads);
        let passes = iterations(copies * bytes(n_in, n_out, 1));

        let elapsed = timed_threads(threads, n_out, |index, y| {
            let (first, end) = ranges[index];
            for _ in 0..passes {
                for pass in 0..copies {
                    // SAFETY: `data[pass]` holds `n_out` rows of `n_in` values,
                    // `scales[pass]` holds `n_out * groups`, and the row range is inside.
                    unsafe {
                        int8_rows(
                            &x,
                            &data[pass],
                            &scales[pass],
                            n_in,
                            first,
                            end,
                            &mut y[first..end],
                        );
                    }
                }
            }
        });
        rate(passes * copies * bytes(n_in, n_out, 1), elapsed)
    }

    fn rate(processed: usize, elapsed: Duration) -> f64 {
        black_box(processed) as f64 / elapsed.as_secs_f64() / 1e9
    }

    fn column_ranges(n_out: usize, threads: usize) -> Vec<(usize, usize)> {
        (0..threads)
            .map(|index| {
                let start = (index * n_out / threads) / 32 * 32;
                let end = if index + 1 == threads {
                    n_out
                } else {
                    ((index + 1) * n_out / threads) / 32 * 32
                };
                (start, end)
            })
            .collect()
    }

    fn row_ranges(n_out: usize, threads: usize) -> Vec<(usize, usize)> {
        (0..threads)
            .map(|index| (index * n_out / threads, (index + 1) * n_out / threads))
            .collect()
    }

    fn timed_threads(
        threads: usize,
        n_out: usize,
        run: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Duration {
        if threads == 1 {
            let mut y = vec![0.0f32; n_out];
            let start = Instant::now();
            run(0, &mut y);
            let elapsed = start.elapsed();
            black_box(&y);
            return elapsed;
        }
        let barrier = Barrier::new(threads + 1);
        std::thread::scope(|scope| {
            for index in 0..threads {
                let barrier = &barrier;
                let run = &run;
                scope.spawn(move || {
                    let mut y = vec![0.0f32; n_out];
                    barrier.wait();
                    run(index, &mut y);
                    barrier.wait();
                    black_box(&y);
                });
            }
            barrier.wait();
            let start = Instant::now();
            barrier.wait();
            start.elapsed()
        })
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_in * n_out`, and
    // `first..end` is a 32-aligned range inside `0..n_out`.
    unsafe fn f32_columns(
        x: &[f32],
        weight: &[f32],
        n_in: usize,
        n_out: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let mut j = first;
        while j < end {
            let columns = FLOATS_PER_ROW * 4;
            let mut acc = [vdupq_n_f32(0.0); FLOATS_PER_ROW];
            let mut p = 0;
            while p + 4 <= n_in {
                let xv = vld1q_f32(x.add(p));
                for (c, acc_c) in acc.iter_mut().enumerate() {
                    let w = weight.add(p * n_out + j + c * 4);
                    *acc_c = vfmaq_laneq_f32::<0>(*acc_c, vld1q_f32(w), xv);
                    *acc_c = vfmaq_laneq_f32::<1>(*acc_c, vld1q_f32(w.add(n_out)), xv);
                    *acc_c = vfmaq_laneq_f32::<2>(*acc_c, vld1q_f32(w.add(2 * n_out)), xv);
                    *acc_c = vfmaq_laneq_f32::<3>(*acc_c, vld1q_f32(w.add(3 * n_out)), xv);
                }
                p += 4;
            }
            while p < n_in {
                let a = vdupq_n_f32(*x.add(p));
                for (c, acc_c) in acc.iter_mut().enumerate() {
                    *acc_c = vfmaq_f32(*acc_c, a, vld1q_f32(weight.add(p * n_out + j + c * 4)));
                }
                p += 1;
            }
            for (c, acc_c) in acc.iter().enumerate() {
                vst1q_f32(y.as_mut_ptr().add(j - first + c * 4), *acc_c);
            }
            j += columns;
        }
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_out * n_in`, and
    // `first..end` is inside `0..n_out`.
    unsafe fn f32_rows(
        x: &[f32],
        weight: &[f32],
        n_in: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        for row in first..end {
            let row_weight = weight.add(row * n_in);
            let mut acc = [vdupq_n_f32(0.0); 8];
            let mut p = 0;
            while p + 32 <= n_in {
                for (k, acc_k) in acc.iter_mut().enumerate() {
                    *acc_k = vfmaq_f32(
                        *acc_k,
                        vld1q_f32(row_weight.add(p + k * 4)),
                        vld1q_f32(x.add(p + k * 4)),
                    );
                }
                p += 32;
            }
            let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
            let mut sum = vaddvq_f32(pair(
                pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
                pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
            ));
            while p < n_in {
                sum += *row_weight.add(p) * *x.add(p);
                p += 1;
            }
            y[row - first] = sum;
        }
    }

    // SAFETY: reads 8 readable bytes at `ptr`.
    #[inline]
    unsafe fn widen8(ptr: *const i8) -> [float32x4_t; 2] {
        let low = vmovl_s8(vld1_s8(ptr));
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
        ]
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_in * n_out`, `scale` has
    // `groups * n_out`, and `first..end` is a 32-aligned range inside `0..n_out`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn int8_columns(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        n_in: usize,
        n_out: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let scale = scale.as_ptr();
        let mut j = first;
        while j < end {
            let mut total = [[vdupq_n_f32(0.0); 2]; 2];
            let mut p = 0;
            while p < n_in {
                let group_end = (p / GROUP * GROUP + GROUP).min(n_in);
                let scale_row = p / GROUP * n_out;
                let mut acc = [[vdupq_n_f32(0.0); 2]; 2];
                while p + 4 <= group_end {
                    let xv = vld1q_f32(x.add(p));
                    for c in 0..2 {
                        let column = j + c * 8;
                        let w0 = widen8(weight.add(p * n_out + column));
                        let w1 = widen8(weight.add((p + 1) * n_out + column));
                        let w2 = widen8(weight.add((p + 2) * n_out + column));
                        let w3 = widen8(weight.add((p + 3) * n_out + column));
                        for k in 0..2 {
                            acc[c][k] = vfmaq_laneq_f32::<0>(acc[c][k], w0[k], xv);
                            acc[c][k] = vfmaq_laneq_f32::<1>(acc[c][k], w1[k], xv);
                            acc[c][k] = vfmaq_laneq_f32::<2>(acc[c][k], w2[k], xv);
                            acc[c][k] = vfmaq_laneq_f32::<3>(acc[c][k], w3[k], xv);
                        }
                    }
                    p += 4;
                }
                while p < group_end {
                    let a = *x.add(p);
                    for c in 0..2 {
                        let w = widen8(weight.add(p * n_out + j + c * 8));
                        for k in 0..2 {
                            acc[c][k] = vfmaq_n_f32(acc[c][k], w[k], a);
                        }
                    }
                    p += 1;
                }
                for c in 0..2 {
                    for k in 0..2 {
                        let column = scale_row + j + c * 8 + k * 4;
                        total[c][k] =
                            vfmaq_f32(total[c][k], acc[c][k], vld1q_f32(scale.add(column)));
                    }
                }
            }
            for c in 0..2 {
                for k in 0..2 {
                    vst1q_f32(y.as_mut_ptr().add(j - first + c * 8 + k * 4), total[c][k]);
                }
            }
            j += 16;
        }
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_out * n_in`, `scale` has
    // `n_out * groups`, and `first..end` is inside `0..n_out`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn int8_rows(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        n_in: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let scale = scale.as_ptr();
        let groups = n_in.div_ceil(GROUP);
        for row in first..end {
            let row_weight = weight.add(row * n_in);
            let row_scale = scale.add(row * groups);
            let mut total = 0.0;
            let mut g = 0;
            while g < groups {
                let start = g * GROUP;
                let group_end = (start + GROUP).min(n_in);
                let mut acc = [vdupq_n_f32(0.0); 8];
                let mut p = start;
                while p + 32 <= group_end {
                    for k in 0..4 {
                        let w = widen8(row_weight.add(p + k * 8));
                        acc[2 * k] = vfmaq_f32(acc[2 * k], w[0], vld1q_f32(x.add(p + k * 8)));
                        acc[2 * k + 1] =
                            vfmaq_f32(acc[2 * k + 1], w[1], vld1q_f32(x.add(p + k * 8 + 4)));
                    }
                    p += 32;
                }
                let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
                let mut sum = vaddvq_f32(pair(
                    pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
                    pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
                ));
                while p < group_end {
                    sum += *x.add(p) * *row_weight.add(p) as f32;
                    p += 1;
                }
                total += sum * *row_scale.add(g);
                g += 1;
            }
            y[row - first] = total;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod tiles {
    use std::arch::aarch64::*;
    use std::hint::black_box;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};

    const GROUP: usize = 64;
    const TARGET_BYTES: usize = 2 << 30;
    const WORKING_SET: usize = 128 << 20;
    const REPEATS: usize = 5;
    const TOKENS: usize = 256;

    pub fn run() {
        for (n_in, n_out) in [(768, 3072), (3072, 768), (50257, 768)] {
            println!("f32 decode {n_in}x{n_out}, GB/s");
            report_f32::<1, 8>(n_in, n_out);
            report_f32::<2, 8>(n_in, n_out);
            report_f32::<2, 4>(n_in, n_out);
            report_f32::<4, 8>(n_in, n_out);
            report_f32::<4, 4>(n_in, n_out);
            println!("int8 decode {n_in}x{n_out}, GB/s");
            report_int8::<1, 8>(n_in, n_out);
            report_int8::<2, 8>(n_in, n_out);
            report_int8::<2, 4>(n_in, n_out);
            report_int8::<4, 8>(n_in, n_out);
            report_int8::<4, 4>(n_in, n_out);
        }
        for (n_in, n_out) in [(768, 3072), (3072, 768)] {
            println!("f32 prefill {n_in}x{n_out}, {TOKENS} tokens, GFLOP/s");
            report_prefill::<4, 4>(n_in, n_out);
            report_prefill::<4, 8>(n_in, n_out);
            report_prefill::<2, 8>(n_in, n_out);
            report_prefill::<4, 2>(n_in, n_out);
            report_prefill::<8, 4>(n_in, n_out);
            report_prefill::<3, 8>(n_in, n_out);
            report_prefill::<4, 6>(n_in, n_out);
            report_columns(n_in, n_out);
        }
    }

    fn report_f32<const R: usize, const A: usize>(n_in: usize, n_out: usize) {
        let mut one = 0.0f64;
        let mut four = 0.0f64;
        for _ in 0..REPEATS {
            one = one.max(decode_f32::<R, A>(n_in, n_out, 1));
            four = four.max(decode_f32::<R, A>(n_in, n_out, 4));
        }
        println!("  rows {R}, accumulators {A}: 1 core {one:.1}, 4 cores {four:.1}");
    }

    fn report_int8<const R: usize, const A: usize>(n_in: usize, n_out: usize) {
        let mut one = 0.0f64;
        let mut four = 0.0f64;
        for _ in 0..REPEATS {
            one = one.max(decode_int8::<R, A>(n_in, n_out, 1));
            four = four.max(decode_int8::<R, A>(n_in, n_out, 4));
        }
        println!("  rows {R}, accumulators {A}: 1 core {one:.1}, 4 cores {four:.1}");
    }

    fn report_columns(n_in: usize, n_out: usize) {
        let mut one = 0.0f64;
        let mut four = 0.0f64;
        for _ in 0..REPEATS {
            one = one.max(prefill_columns(n_in, n_out, 1));
            four = four.max(prefill_columns(n_in, n_out, 4));
        }
        println!("  input-major 4x16 (shipped): 1 core {one:.0}, 4 cores {four:.0}");
    }

    fn report_prefill<const R: usize, const C: usize>(n_in: usize, n_out: usize) {
        let mut one = 0.0f64;
        let mut four = 0.0f64;
        for _ in 0..REPEATS {
            one = one.max(prefill_f32::<R, C>(n_in, n_out, 1));
            four = four.max(prefill_f32::<R, C>(n_in, n_out, 4));
        }
        println!("  tokens {R}, outputs {C}: 1 core {one:.0}, 4 cores {four:.0}");
    }

    fn values(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i * 37 + 11) % 251) as f32 / 128.0 - 0.98)
            .collect()
    }

    fn integers(count: usize) -> Vec<i8> {
        (0..count)
            .map(|i| ((i * 53 + 7) % 255) as i32 as i8)
            .collect()
    }

    fn copies(total: usize) -> usize {
        (WORKING_SET / total).clamp(1, 32)
    }

    fn passes(working_set: usize) -> usize {
        (TARGET_BYTES / working_set).max(1)
    }

    fn rate(processed: usize, elapsed: Duration) -> f64 {
        black_box(processed) as f64 / elapsed.as_secs_f64() / 1e9
    }

    fn decode_f32<const R: usize, const A: usize>(
        n_in: usize,
        n_out: usize,
        threads: usize,
    ) -> f64 {
        let copies = copies(n_in * n_out * 4);
        let data: Vec<Vec<f32>> = (0..copies).map(|_| values(n_in * n_out)).collect();
        let x = black_box(values(n_in));
        let passes = passes(copies * n_in * n_out * 4);
        let bounds = row_bounds(n_out, threads);
        let elapsed = timed(threads, n_out, |index, y| {
            let (first, end) = bounds[index];
            for _ in 0..passes {
                for weight in &data {
                    // SAFETY: the row range is inside `n_out` rows of `n_in` values.
                    unsafe { f32_decode::<R, A>(&x, weight, n_in, first, end, &mut y[first..end]) };
                }
            }
        });
        rate(passes * copies * n_in * n_out * 4, elapsed)
    }

    fn decode_int8<const R: usize, const A: usize>(
        n_in: usize,
        n_out: usize,
        threads: usize,
    ) -> f64 {
        let copies = copies(n_in * n_out);
        let data: Vec<Vec<i8>> = (0..copies).map(|_| integers(n_in * n_out)).collect();
        let groups = n_in.div_ceil(GROUP);
        let scales: Vec<Vec<f32>> = (0..copies)
            .map(|_| {
                (0..n_out * groups)
                    .map(|i| (i % 13 + 1) as f32 / 4096.0)
                    .collect()
            })
            .collect();
        let x = black_box(values(n_in));
        let passes = passes(copies * n_in * n_out);
        let bounds = row_bounds(n_out, threads);
        let elapsed = timed(threads, n_out, |index, y| {
            let (first, end) = bounds[index];
            for _ in 0..passes {
                for (pass, weight) in data.iter().enumerate() {
                    // SAFETY: the row range is inside `n_out` rows of `n_in` values.
                    unsafe {
                        int8_decode::<R, A>(
                            &x,
                            weight,
                            &scales[pass],
                            n_in,
                            first,
                            end,
                            &mut y[first..end],
                        );
                    }
                }
            }
        });
        rate(passes * copies * n_in * n_out, elapsed)
    }

    fn prefill_f32<const R: usize, const C: usize>(
        n_in: usize,
        n_out: usize,
        threads: usize,
    ) -> f64 {
        let copies = copies(n_in * n_out * 4);
        let data: Vec<Vec<f32>> = (0..copies).map(|_| values(n_in * n_out)).collect();
        let bias = vec![0.0f32; n_out];
        let x = black_box(values(TOKENS * n_in));
        let passes = passes(copies * n_in * n_out * 4);
        let elapsed = timed_tokens(threads, |index, y| {
            let first = index * TOKENS / threads;
            let rows = (index + 1) * TOKENS / threads - first;
            let x = &x[first * n_in..];
            for _ in 0..passes {
                for weight in &data {
                    // SAFETY: `rows` token rows, `n_out` output rows, and `y` holds
                    // `rows * n_out` values.
                    unsafe {
                        f32_prefill::<R, C>(x, weight, &bias, rows, n_in, n_out, y);
                    }
                }
            }
        });
        black_box(&data);
        let flops =
            2.0 * passes as f64 * copies as f64 * TOKENS as f64 * n_in as f64 * n_out as f64;
        flops / elapsed.as_secs_f64() / 1e9
    }

    fn prefill_columns(n_in: usize, n_out: usize, threads: usize) -> f64 {
        let copies = copies(n_in * n_out * 4);
        let data: Vec<Vec<f32>> = (0..copies).map(|_| values(n_in * n_out)).collect();
        let bias = vec![0.0f32; n_out];
        let x = black_box(values(TOKENS * n_in));
        let passes = passes(copies * n_in * n_out * 4);
        let elapsed = timed_tokens(threads, |index, y| {
            let first = index * TOKENS / threads;
            let rows = (index + 1) * TOKENS / threads - first;
            let x = &x[first * n_in..];
            for _ in 0..passes {
                for weight in &data {
                    // SAFETY: `rows` token rows, `n_out` output rows, and `y` holds
                    // `rows * n_out` values.
                    unsafe {
                        f32_columns_prefill::<4, 4>(x, weight, &bias, rows, n_in, n_out, y);
                    }
                }
            }
        });
        black_box(&data);
        let flops =
            2.0 * passes as f64 * copies as f64 * TOKENS as f64 * n_in as f64 * n_out as f64;
        flops / elapsed.as_secs_f64() / 1e9
    }

    fn row_bounds(n_out: usize, threads: usize) -> Vec<(usize, usize)> {
        (0..threads)
            .map(|index| (index * n_out / threads, (index + 1) * n_out / threads))
            .collect()
    }

    fn timed(threads: usize, n_out: usize, run: impl Fn(usize, &mut [f32]) + Sync) -> Duration {
        if threads == 1 {
            let mut y = vec![0.0f32; n_out];
            let start = Instant::now();
            run(0, &mut y);
            let elapsed = start.elapsed();
            black_box(&y);
            return elapsed;
        }
        timed_scoped(threads, n_out, run)
    }

    fn timed_tokens(threads: usize, run: impl Fn(usize, &mut [f32]) + Sync) -> Duration {
        timed_scoped(threads, TOKENS * 3072, run)
    }

    fn timed_scoped(
        threads: usize,
        buffer: usize,
        run: impl Fn(usize, &mut [f32]) + Sync,
    ) -> Duration {
        if threads == 1 {
            let mut y = vec![0.0f32; buffer];
            let start = Instant::now();
            run(0, &mut y);
            let elapsed = start.elapsed();
            black_box(&y);
            return elapsed;
        }
        let barrier = Barrier::new(threads + 1);
        std::thread::scope(|scope| {
            for index in 0..threads {
                let barrier = &barrier;
                let run = &run;
                scope.spawn(move || {
                    let mut y = vec![0.0f32; buffer];
                    barrier.wait();
                    run(index, &mut y);
                    barrier.wait();
                    black_box(&y);
                });
            }
            barrier.wait();
            let start = Instant::now();
            barrier.wait();
            start.elapsed()
        })
    }

    unsafe fn horizontal<const A: usize>(acc: [float32x4_t; A]) -> f32 {
        let mut total = acc[0];
        for value in &acc[1..] {
            total = vaddq_f32(total, *value);
        }
        vaddvq_f32(total)
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_out * n_in`, and
    // `first..end` is inside `0..n_out`.
    unsafe fn f32_decode<const R: usize, const A: usize>(
        x: &[f32],
        weight: &[f32],
        n_in: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let mut row = first;
        while row + R <= end {
            let mut acc = [[vdupq_n_f32(0.0); A]; R];
            let mut p = 0;
            while p + A * 4 <= n_in {
                let mut xv = [vdupq_n_f32(0.0); A];
                for (a, value) in xv.iter_mut().enumerate() {
                    *value = vld1q_f32(x.add(p + a * 4));
                }
                for (r, acc_r) in acc.iter_mut().enumerate() {
                    let row_weight = weight.add((row + r) * n_in);
                    for (a, acc_ra) in acc_r.iter_mut().enumerate() {
                        *acc_ra = vfmaq_f32(*acc_ra, vld1q_f32(row_weight.add(p + a * 4)), xv[a]);
                    }
                }
                p += A * 4;
            }
            for (r, acc_r) in acc.iter().enumerate() {
                let row_weight = weight.add((row + r) * n_in);
                let mut sum = horizontal(*acc_r);
                let mut q = p;
                while q < n_in {
                    sum += *row_weight.add(q) * *x.add(q);
                    q += 1;
                }
                y[row + r - first] = sum;
            }
            row += R;
        }
        while row < end {
            let row_weight = weight.add(row * n_in);
            let mut acc = [vdupq_n_f32(0.0); A];
            let mut p = 0;
            while p + A * 4 <= n_in {
                for (a, acc_a) in acc.iter_mut().enumerate() {
                    *acc_a = vfmaq_f32(
                        *acc_a,
                        vld1q_f32(row_weight.add(p + a * 4)),
                        vld1q_f32(x.add(p + a * 4)),
                    );
                }
                p += A * 4;
            }
            let mut sum = horizontal(acc);
            while p < n_in {
                sum += *row_weight.add(p) * *x.add(p);
                p += 1;
            }
            y[row - first] = sum;
            row += 1;
        }
    }

    // SAFETY: reads 8 readable bytes at `ptr`.
    #[inline]
    unsafe fn widen8(ptr: *const i8) -> [float32x4_t; 2] {
        let low = vmovl_s8(vld1_s8(ptr));
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
        ]
    }

    // SAFETY: `x` has `n_in` values, `weight` has `n_out * n_in`, `scale` has
    // `n_out * groups`, and `first..end` is inside `0..n_out`.
    unsafe fn int8_decode<const R: usize, const A: usize>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        n_in: usize,
        first: usize,
        end: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let scale = scale.as_ptr();
        let groups = n_in.div_ceil(GROUP);
        let mut row = first;
        while row < end {
            let count = (end - row).min(R);
            let mut total = [0.0f32; R];
            for g in 0..groups {
                let start = g * GROUP;
                let group_end = (start + GROUP).min(n_in);
                let mut acc = [[vdupq_n_f32(0.0); A]; R];
                let mut p = start;
                while p + A * 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); A];
                    for (a, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add(p + a * 4));
                    }
                    for (r, acc_r) in acc.iter_mut().enumerate() {
                        let row_weight = weight.add((row + r) * n_in);
                        for h in 0..A / 2 {
                            let w = widen8(row_weight.add(p + h * 8));
                            acc_r[2 * h] = vfmaq_f32(acc_r[2 * h], w[0], xv[2 * h]);
                            acc_r[2 * h + 1] = vfmaq_f32(acc_r[2 * h + 1], w[1], xv[2 * h + 1]);
                        }
                    }
                    p += A * 4;
                }
                for (r, acc_r) in acc.iter().enumerate() {
                    let mut sum = horizontal(*acc_r);
                    let row_weight = weight.add((row + r) * n_in);
                    let mut q = p;
                    while q < group_end {
                        sum += (*row_weight.add(q)) as f32 * *x.add(q);
                        q += 1;
                    }
                    total[r] += sum * *scale.add((row + r) * groups + g);
                }
            }
            for r in 0..count {
                y[row + r - first] = total[r];
            }
            row += count;
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_prefill<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
        let mut i = 0;
        while i + R <= rows {
            let mut j = 0;
            while j + C <= n_out {
                f32_tile::<R, C>(x, weight, bias, n_in, n_out, i, j, y);
                j += C;
            }
            while j < n_out {
                f32_tile::<R, 1>(x, weight, bias, n_in, n_out, i, j, y);
                j += 1;
            }
            i += R;
        }
        while i < rows {
            let mut j = 0;
            while j + C <= n_out {
                f32_tile::<1, C>(x, weight, bias, n_in, n_out, i, j, y);
                j += C;
            }
            while j < n_out {
                f32_tile::<1, 1>(x, weight, bias, n_in, n_out, i, j, y);
                j += 1;
            }
            i += 1;
        }
    }

    // SAFETY: `i + R <= rows` and `j + C <= n_out`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_tile<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        i: usize,
        j: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let bias = bias.as_ptr();
        let mut acc = [[vdupq_n_f32(0.0); C]; R];
        let mut p = 0;
        while p + 4 <= n_in {
            let mut xv = [vdupq_n_f32(0.0); R];
            for (r, value) in xv.iter_mut().enumerate() {
                *value = vld1q_f32(x.add((i + r) * n_in + p));
            }
            for c in 0..C {
                let w = vld1q_f32(weight.add((j + c) * n_in + p));
                for (r, value) in xv.iter().enumerate() {
                    acc[r][c] = vfmaq_f32(acc[r][c], w, *value);
                }
            }
            p += 4;
        }
        for r in 0..R {
            for c in 0..C {
                let mut sum = vaddvq_f32(acc[r][c]);
                let mut q = p;
                while q < n_in {
                    sum += *weight.add((j + c) * n_in + q) * *x.add((i + r) * n_in + q);
                    q += 1;
                }
                y[(i + r) * n_out + j + c] = sum + *bias.add(j + c);
            }
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_in * n_out`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_columns_prefill<const R: usize, const V: usize>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let bias = bias.as_ptr();
        let mut i = 0;
        while i < rows {
            let count = (rows - i).min(R);
            let mut j = 0;
            while j < n_out {
                let mut acc = [[vdupq_n_f32(0.0); V]; R];
                let mut p = 0;
                while p + 4 <= n_in {
                    let mut xv = [vdupq_n_f32(0.0); R];
                    for (r, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add((i + r) * n_in + p));
                    }
                    for c in 0..V {
                        let w = weight.add(p * n_out + j + c * 4);
                        let w0 = vld1q_f32(w);
                        let w1 = vld1q_f32(w.add(n_out));
                        let w2 = vld1q_f32(w.add(2 * n_out));
                        let w3 = vld1q_f32(w.add(3 * n_out));
                        for (r, value) in xv.iter().enumerate().take(count) {
                            acc[r][c] = vfmaq_laneq_f32::<0>(acc[r][c], w0, *value);
                            acc[r][c] = vfmaq_laneq_f32::<1>(acc[r][c], w1, *value);
                            acc[r][c] = vfmaq_laneq_f32::<2>(acc[r][c], w2, *value);
                            acc[r][c] = vfmaq_laneq_f32::<3>(acc[r][c], w3, *value);
                        }
                    }
                    p += 4;
                }
                for r in 0..count {
                    for c in 0..V {
                        let mut sum = vaddvq_f32(acc[r][c]);
                        let mut q = p;
                        while q < n_in {
                            sum += *weight.add(q * n_out + j + c * 4) * *x.add((i + r) * n_in + q);
                            q += 1;
                        }
                        let value = sum + *bias.add(j + c * 4);
                        y[(i + r) * n_out + j + c * 4] = value;
                    }
                }
                j += V * 4;
            }
            i += count;
        }
    }
}
