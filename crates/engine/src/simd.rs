#[derive(Clone, Copy)]
pub(crate) enum Epilogue {
    Store,
    Gelu,
}

fn gelu(x: f32) -> f32 {
    let coefficient = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (coefficient * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    use std::ops::Range;

    use super::{gelu, Epilogue};

    const DECODE_ROWS: usize = 2;
    const DECODE_ACCUMULATORS: usize = 4;
    const INT8_DECODE_ROWS: usize = 2;
    const INT8_DECODE_ACCUMULATORS: usize = 4;
    const PREFILL_TOKENS: usize = 4;
    const PREFILL_OUTPUTS: usize = 4;
    // A block of output rows stays in cache while every prompt row passes over it.
    const PREFILL_OUTPUT_BLOCK: usize = 16;

    #[inline]
    fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
        let value = sum + bias.unwrap_or(0.0);
        if with_gelu {
            gelu(value)
        } else {
            value
        }
    }

    #[inline]
    unsafe fn horizontal8(acc: [float32x4_t; 8]) -> f32 {
        let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
        vaddvq_f32(pair(
            pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
            pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
        ))
    }

    #[inline]
    unsafe fn horizontal4(acc: [float32x4_t; 4]) -> f32 {
        let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
        vaddvq_f32(pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])))
    }

    // SAFETY: `ptr` points to at least 8 readable bytes.
    #[inline]
    unsafe fn widen8(ptr: *const i8) -> [float32x4_t; 2] {
        let low = vmovl_s8(vld1_s8(ptr));
        [
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
        ]
    }

    // Every output is one contiguous row of weights, so a decoded token is a
    // long dot product with independent accumulators.
    // SAFETY: `weight` has `n_out * n_in` values and `y` has `end - first`;
    // `bias`, when present, has `end - first` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_outputs(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        first: usize,
        end: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let mut row = first;
        while row + DECODE_ROWS <= end {
            let mut acc = [[vdupq_n_f32(0.0); DECODE_ACCUMULATORS]; DECODE_ROWS];
            let mut p = 0;
            while p + DECODE_ACCUMULATORS * 4 <= n_in {
                let mut xv = [vdupq_n_f32(0.0); DECODE_ACCUMULATORS];
                for (a, value) in xv.iter_mut().enumerate() {
                    *value = vld1q_f32(x.add(p + a * 4));
                }
                for (r, acc_r) in acc.iter_mut().enumerate() {
                    let row_weight = weight.add((row + r) * n_in);
                    for (a, acc_ra) in acc_r.iter_mut().enumerate() {
                        *acc_ra = vfmaq_f32(*acc_ra, vld1q_f32(row_weight.add(p + a * 4)), xv[a]);
                    }
                }
                p += DECODE_ACCUMULATORS * 4;
            }
            let mut sums = [0.0f32; DECODE_ROWS];
            for (r, sum) in sums.iter_mut().enumerate() {
                let row_weight = weight.add((row + r) * n_in);
                *sum = horizontal4(acc[r]);
                let mut q = p;
                while q < n_in {
                    *sum += *row_weight.add(q) * *x.add(q);
                    q += 1;
                }
            }
            for (r, sum) in sums.iter().enumerate() {
                *y.get_unchecked_mut(row + r - first) = finish(
                    *sum,
                    bias.map(|bias| *bias.get_unchecked(row + r - first)),
                    with_gelu,
                );
            }
            row += DECODE_ROWS;
        }
        while row < end {
            let row_weight = weight.add(row * n_in);
            let mut acc = [vdupq_n_f32(0.0); DECODE_ACCUMULATORS];
            let mut p = 0;
            while p + DECODE_ACCUMULATORS * 4 <= n_in {
                for (a, acc_a) in acc.iter_mut().enumerate() {
                    *acc_a = vfmaq_f32(
                        *acc_a,
                        vld1q_f32(row_weight.add(p + a * 4)),
                        vld1q_f32(x.add(p + a * 4)),
                    );
                }
                p += DECODE_ACCUMULATORS * 4;
            }
            let mut sum = horizontal4(acc);
            while p < n_in {
                sum += *row_weight.add(p) * *x.add(p);
                p += 1;
            }
            *y.get_unchecked_mut(row - first) = finish(
                sum,
                bias.map(|bias| *bias.get_unchecked(row - first)),
                with_gelu,
            );
            row += 1;
        }
    }

    // SAFETY: `weight` has `n_out * n_in` values, `y` has `end - first`, and
    // `bias`, when present, has `end - first` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn int8_outputs(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: Option<&[f32]>,
        group: usize,
        n_in: usize,
        first: usize,
        end: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let scale = scale.as_ptr();
        let groups = n_in.div_ceil(group);
        let mut row = first;
        while row + INT8_DECODE_ROWS <= end {
            let mut total = [0.0f32; INT8_DECODE_ROWS];
            for g in 0..groups {
                let start = g * group;
                let group_end = (start + group).min(n_in);
                let mut acc = [[vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS]; INT8_DECODE_ROWS];
                let mut p = start;
                while p + INT8_DECODE_ACCUMULATORS * 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                    for (a, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add(p + a * 4));
                    }
                    for (r, acc_r) in acc.iter_mut().enumerate() {
                        let row_weight = weight.add((row + r) * n_in);
                        for h in 0..INT8_DECODE_ACCUMULATORS / 2 {
                            let w = widen8(row_weight.add(p + h * 8));
                            acc_r[2 * h] = vfmaq_f32(acc_r[2 * h], w[0], xv[2 * h]);
                            acc_r[2 * h + 1] = vfmaq_f32(acc_r[2 * h + 1], w[1], xv[2 * h + 1]);
                        }
                    }
                    p += INT8_DECODE_ACCUMULATORS * 4;
                }
                for (r, acc_r) in acc.iter().enumerate() {
                    let row_weight = weight.add((row + r) * n_in);
                    let mut sum = horizontal4(*acc_r);
                    let mut q = p;
                    while q < group_end {
                        sum += (*row_weight.add(q)) as f32 * *x.add(q);
                        q += 1;
                    }
                    total[r] += sum * *scale.add((row + r) * groups + g);
                }
            }
            for (r, value) in total.iter().enumerate() {
                *y.get_unchecked_mut(row + r - first) = finish(
                    *value,
                    bias.map(|bias| *bias.get_unchecked(row + r - first)),
                    with_gelu,
                );
            }
            row += INT8_DECODE_ROWS;
        }
        while row < end {
            let row_weight = weight.add(row * n_in);
            let mut total = 0.0;
            for g in 0..groups {
                let start = g * group;
                let group_end = (start + group).min(n_in);
                let mut acc = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                let mut p = start;
                while p + INT8_DECODE_ACCUMULATORS * 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                    for (a, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add(p + a * 4));
                    }
                    for h in 0..INT8_DECODE_ACCUMULATORS / 2 {
                        let w = widen8(row_weight.add(p + h * 8));
                        acc[2 * h] = vfmaq_f32(acc[2 * h], w[0], xv[2 * h]);
                        acc[2 * h + 1] = vfmaq_f32(acc[2 * h + 1], w[1], xv[2 * h + 1]);
                    }
                    p += INT8_DECODE_ACCUMULATORS * 4;
                }
                let mut sum = horizontal4(acc);
                while p < group_end {
                    sum += (*row_weight.add(p)) as f32 * *x.add(p);
                    p += 1;
                }
                total += sum * *scale.add(row * groups + g);
            }
            *y.get_unchecked_mut(row - first) = finish(
                total,
                bias.map(|bias| *bias.get_unchecked(row - first)),
                with_gelu,
            );
            row += 1;
        }
    }

    // Prefill keeps the same stored layout: a tile of input rows is multiplied
    // by a tile of output rows, each output row read contiguously.
    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values; `i + R <= rows` and
    // `j + C <= n_out`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn f32_tile<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        i: usize,
        j: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
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
        for (r, acc_r) in acc.iter().enumerate() {
            for (c, accumulated) in acc_r.iter().enumerate() {
                let mut sum = vaddvq_f32(*accumulated);
                let mut q = p;
                while q < n_in {
                    sum += *weight.add((j + c) * n_in + q) * *x.add((i + r) * n_in + q);
                    q += 1;
                }
                *y.get_unchecked_mut((i + r) * n_out + j + c) =
                    finish(sum, Some(*bias.get_unchecked(j + c)), with_gelu);
            }
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_prefill(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let mut block = 0;
        while block < n_out {
            let end = (block + PREFILL_OUTPUT_BLOCK).min(n_out);
            let mut i = 0;
            while i + PREFILL_TOKENS <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    f32_tile::<PREFILL_TOKENS, PREFILL_OUTPUTS>(
                        x, weight, bias, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    f32_tile::<PREFILL_TOKENS, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                    j += 1;
                }
                i += PREFILL_TOKENS;
            }
            while i < rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    f32_tile::<1, PREFILL_OUTPUTS>(
                        x, weight, bias, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    f32_tile::<1, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                    j += 1;
                }
                i += 1;
            }
            block = end;
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `scale` has
    // `n_out * n_in.div_ceil(group)`, `bias` has `n_out`, and `y` has
    // `rows * n_out` values; `i + R <= rows` and `j + C <= n_out`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn int8_tile<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        group: usize,
        n_in: usize,
        n_out: usize,
        i: usize,
        j: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let scale = scale.as_ptr();
        let groups = n_in.div_ceil(group);
        let mut total = [[0.0f32; C]; R];
        for g in 0..groups {
            let start = g * group;
            let group_end = (start + group).min(n_in);
            let mut acc = [[vdupq_n_f32(0.0); C]; R];
            let mut extra = [[0.0f32; C]; R];
            let mut p = start;
            while p + 8 <= group_end {
                let mut xv = [[vdupq_n_f32(0.0); 2]; R];
                for (r, value) in xv.iter_mut().enumerate() {
                    *value = [
                        vld1q_f32(x.add((i + r) * n_in + p)),
                        vld1q_f32(x.add((i + r) * n_in + p + 4)),
                    ];
                }
                for c in 0..C {
                    let w = widen8(weight.add((j + c) * n_in + p));
                    for (r, value) in xv.iter().enumerate() {
                        acc[r][c] = vfmaq_f32(acc[r][c], w[0], value[0]);
                        acc[r][c] = vfmaq_f32(acc[r][c], w[1], value[1]);
                    }
                }
                p += 8;
            }
            while p < group_end {
                for (r, extra_r) in extra.iter_mut().enumerate() {
                    for (c, extra_rc) in extra_r.iter_mut().enumerate() {
                        *extra_rc +=
                            *x.add((i + r) * n_in + p) * (*weight.add((j + c) * n_in + p)) as f32;
                    }
                }
                p += 1;
            }
            for (r, extra_r) in extra.iter().enumerate() {
                for (c, extra_rc) in extra_r.iter().enumerate() {
                    total[r][c] +=
                        (vaddvq_f32(acc[r][c]) + *extra_rc) * *scale.add((j + c) * groups + g);
                }
            }
        }
        for (r, total_r) in total.iter().enumerate() {
            for (c, value) in total_r.iter().enumerate() {
                *y.get_unchecked_mut((i + r) * n_out + j + c) =
                    finish(*value, Some(*bias.get_unchecked(j + c)), with_gelu);
            }
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `scale` has
    // `n_out * n_in.div_ceil(group)`, `bias` has `n_out`, and `y` has
    // `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn int8_prefill(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        group: usize,
        rows: usize,
        n_in: usize,
        n_out: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let mut block = 0;
        while block < n_out {
            let end = (block + PREFILL_OUTPUT_BLOCK).min(n_out);
            let mut i = 0;
            while i + PREFILL_TOKENS <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    int8_tile::<PREFILL_TOKENS, PREFILL_OUTPUTS>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    int8_tile::<PREFILL_TOKENS, 1>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += 1;
                }
                i += PREFILL_TOKENS;
            }
            while i < rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    int8_tile::<1, PREFILL_OUTPUTS>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    int8_tile::<1, 1>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += 1;
                }
                i += 1;
            }
            block = end;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_rows(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        // SAFETY: the slices are sized by the caller for `rows * n_in`,
        // `n_out * n_in`, `n_out` and `rows * n_out`.
        unsafe {
            f32_prefill(
                x,
                weight,
                bias,
                rows,
                n_in,
                n_out,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_outputs(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        // SAFETY: `weight` holds `n_out * n_in` values and `y` holds this output
        // range; `bias` is sliced with the outputs.
        unsafe {
            f32_outputs(
                x,
                weight,
                bias,
                n_in,
                outputs.start,
                outputs.end,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_rows(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        // SAFETY: the slices are sized by the caller for `rows * n_in`,
        // `n_out * n_in`, the per-output scales, `n_out` and `rows * n_out`.
        unsafe {
            int8_prefill(
                x,
                weight,
                scale,
                bias,
                group,
                rows,
                n_in,
                n_out,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_outputs(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        // SAFETY: `weight` holds `n_out * n_in` values, `scale` the per-output
        // groups, and `y` holds this output range; `bias` is sliced with it.
        unsafe {
            int8_outputs(
                x,
                weight,
                scale,
                bias,
                group,
                n_in,
                outputs.start,
                outputs.end,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[cfg(test)]
    pub(crate) fn int8_dot(x: &[f32], weight: &[i8], scale: &[f32], group: usize) -> f32 {
        assert_eq!(x.len(), weight.len(), "dot product needs equal lengths");
        let n = x.len();
        let x = x.as_ptr();
        let weight = weight.as_ptr();

        // SAFETY: every load stays within `0..n`, and the scalar tail handles the rest.
        unsafe {
            let mut total = 0.0;
            let mut p = 0;
            while p < n {
                let group_start = p / group * group;
                let group_end = (group_start + group).min(n);
                let mut acc = [vdupq_n_f32(0.0); 8];
                while p + 32 <= group_end {
                    let g0 = widen8(weight.add(p));
                    let g1 = widen8(weight.add(p + 8));
                    let g2 = widen8(weight.add(p + 16));
                    let g3 = widen8(weight.add(p + 24));
                    for k in 0..2 {
                        acc[k] = vfmaq_f32(acc[k], g0[k], vld1q_f32(x.add(p + k * 4)));
                        acc[2 + k] = vfmaq_f32(acc[2 + k], g1[k], vld1q_f32(x.add(p + 8 + k * 4)));
                        acc[4 + k] = vfmaq_f32(acc[4 + k], g2[k], vld1q_f32(x.add(p + 16 + k * 4)));
                        acc[6 + k] = vfmaq_f32(acc[6 + k], g3[k], vld1q_f32(x.add(p + 24 + k * 4)));
                    }
                    p += 32;
                }
                while p + 16 <= group_end {
                    let g0 = widen8(weight.add(p));
                    let g1 = widen8(weight.add(p + 8));
                    for k in 0..2 {
                        acc[k] = vfmaq_f32(acc[k], g0[k], vld1q_f32(x.add(p + k * 4)));
                        acc[2 + k] = vfmaq_f32(acc[2 + k], g1[k], vld1q_f32(x.add(p + 8 + k * 4)));
                    }
                    p += 16;
                }
                let mut sum = horizontal8(acc);
                while p < group_end {
                    sum += *x.add(p) * *weight.add(p) as f32;
                    p += 1;
                }
                total += sum * *scale.get_unchecked(group_start / group);
            }
            total
        }
    }

    pub(crate) fn weighted_sum(values: &[f32], stride: usize, weights: &[f32], output: &mut [f32]) {
        let width = output.len();
        assert!(width <= stride, "output is wider than the value stride");
        assert!(
            values.len() >= weights.len().saturating_sub(1) * stride + width,
            "values are shorter than the weighted sum needs"
        );

        let values = values.as_ptr();
        let mut offset = 0;
        while offset + 32 <= width {
            // SAFETY: `offset + 32 <= width <= stride`, so each row read is inside
            // `values`, and `output` has `width >= offset + 32` elements.
            unsafe {
                let mut acc = [vdupq_n_f32(0.0); 8];
                for (past, &weight) in weights.iter().enumerate() {
                    let row = values.add(past * stride + offset);
                    let weight = vdupq_n_f32(weight);
                    for (c, acc_c) in acc.iter_mut().enumerate() {
                        *acc_c = vfmaq_f32(*acc_c, weight, vld1q_f32(row.add(c * 4)));
                    }
                }
                for (c, acc_c) in acc.iter().enumerate() {
                    vst1q_f32(output.as_mut_ptr().add(offset + c * 4), *acc_c);
                }
            }
            offset += 32;
        }
        while offset < width {
            let mut sum = 0.0;
            for (past, &weight) in weights.iter().enumerate() {
                // SAFETY: `offset < width <= stride` and the input length check above.
                sum += weight * unsafe { *values.add(past * stride + offset) };
            }
            output[offset] = sum;
            offset += 1;
        }
    }

    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dot product needs equal lengths");
        let n = a.len();
        let a = a.as_ptr();
        let b = b.as_ptr();

        // SAFETY: every load stays within `0..n`, and the scalar tail handles the rest.
        unsafe {
            let mut acc = [vdupq_n_f32(0.0); 8];
            let mut i = 0;
            while i + 32 <= n {
                for (k, acc_k) in acc.iter_mut().enumerate() {
                    *acc_k = vfmaq_f32(
                        *acc_k,
                        vld1q_f32(a.add(i + k * 4)),
                        vld1q_f32(b.add(i + k * 4)),
                    );
                }
                i += 32;
            }
            let mut sum = horizontal8(acc);
            while i < n {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            sum
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use std::arch::wasm32::*;
    use std::ops::Range;
    use std::ptr;

    use super::{gelu, Epilogue};

    const DECODE_ACCUMULATORS: usize = 8;
    const PREFILL_TOKENS: usize = 4;
    const PREFILL_OUTPUTS: usize = 2;
    const INT8_PREFILL_TOKENS: usize = 2;
    const INT8_PREFILL_OUTPUTS: usize = 2;

    #[inline]
    fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
        let value = sum + bias.unwrap_or(0.0);
        if with_gelu {
            gelu(value)
        } else {
            value
        }
    }

    // SAFETY: `ptr` points to at least four readable f32 values.
    #[inline]
    unsafe fn load(ptr: *const f32) -> v128 {
        ptr::read_unaligned(ptr.cast())
    }

    // SAFETY: `ptr` points to at least four writable f32 values.
    #[inline]
    unsafe fn store(ptr: *mut f32, value: v128) {
        ptr::write_unaligned(ptr.cast(), value);
    }

    #[inline]
    fn horizontal_sum(value: v128) -> f32 {
        f32x4_extract_lane::<0>(value)
            + f32x4_extract_lane::<1>(value)
            + f32x4_extract_lane::<2>(value)
            + f32x4_extract_lane::<3>(value)
    }

    #[inline]
    fn multiply_add(target: v128, left: v128, right: v128) -> v128 {
        f32x4_add(target, f32x4_mul(left, right))
    }

    #[inline]
    unsafe fn horizontal8(acc: [v128; 8]) -> f32 {
        let pair = |left: v128, right: v128| f32x4_add(left, right);
        horizontal_sum(pair(
            pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
            pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
        ))
    }

    // SAFETY: `ptr` points to at least eight readable i8 values.
    #[inline]
    unsafe fn widen8(ptr: *const i8) -> [v128; 2] {
        let raw = ptr::read_unaligned(ptr.cast::<u64>()) as i64;
        let bytes = i64x2_replace_lane::<0>(i64x2_splat(0), raw);
        let low = i16x8_extend_low_i8x16(bytes);
        [
            f32x4_convert_i32x4(i32x4_extend_low_i16x8(low)),
            f32x4_convert_i32x4(i32x4_extend_high_i16x8(low)),
        ]
    }

    // SAFETY: `weight` has `n_out * n_in` values and `y` has `end - first`;
    // `bias`, when present, has `end - first` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_outputs(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        first: usize,
        end: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        for row in first..end {
            let row_weight = weight.add(row * n_in);
            let mut acc = [f32x4_splat(0.0); DECODE_ACCUMULATORS];
            let mut p = 0;
            while p + DECODE_ACCUMULATORS * 4 <= n_in {
                for (a, acc_a) in acc.iter_mut().enumerate() {
                    *acc_a = multiply_add(
                        *acc_a,
                        load(row_weight.add(p + a * 4)),
                        load(x.add(p + a * 4)),
                    );
                }
                p += DECODE_ACCUMULATORS * 4;
            }
            let mut sum = horizontal8(acc);
            while p < n_in {
                sum += *row_weight.add(p) * *x.add(p);
                p += 1;
            }
            y[row - first] = finish(sum, bias.map(|bias| bias[row - first]), with_gelu);
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values; `i + R <= rows` and
    // `j + C <= n_out`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn f32_tile<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        i: usize,
        j: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let mut acc = [[f32x4_splat(0.0); C]; R];
        let mut p = 0;
        while p + 4 <= n_in {
            let mut xv = [f32x4_splat(0.0); R];
            for (r, value) in xv.iter_mut().enumerate() {
                *value = load(x.add((i + r) * n_in + p));
            }
            for c in 0..C {
                let w = load(weight.add((j + c) * n_in + p));
                for (r, value) in xv.iter().enumerate() {
                    acc[r][c] = multiply_add(acc[r][c], w, *value);
                }
            }
            p += 4;
        }
        for (r, acc_r) in acc.iter().enumerate() {
            for (c, accumulated) in acc_r.iter().enumerate() {
                let mut sum = horizontal_sum(*accumulated);
                let mut q = p;
                while q < n_in {
                    sum += *weight.add((j + c) * n_in + q) * *x.add((i + r) * n_in + q);
                    q += 1;
                }
                y[(i + r) * n_out + j + c] = finish(sum, Some(bias[j + c]), with_gelu);
            }
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `bias` has
    // `n_out`, and `y` has `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn f32_prefill(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let mut i = 0;
        while i + PREFILL_TOKENS <= rows {
            let mut j = 0;
            while j + PREFILL_OUTPUTS <= n_out {
                f32_tile::<PREFILL_TOKENS, PREFILL_OUTPUTS>(
                    x, weight, bias, n_in, n_out, i, j, with_gelu, y,
                );
                j += PREFILL_OUTPUTS;
            }
            while j < n_out {
                f32_tile::<PREFILL_TOKENS, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                j += 1;
            }
            i += PREFILL_TOKENS;
        }
        while i < rows {
            let mut j = 0;
            while j + PREFILL_OUTPUTS <= n_out {
                f32_tile::<1, PREFILL_OUTPUTS>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                j += PREFILL_OUTPUTS;
            }
            while j < n_out {
                f32_tile::<1, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                j += 1;
            }
            i += 1;
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `scale` has
    // `n_out * n_in.div_ceil(group)`, `bias` has `n_out`, and `y` has
    // `rows * n_out` values; `i + R <= rows` and `j + C <= n_out`.
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    unsafe fn int8_tile<const R: usize, const C: usize>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        group: usize,
        n_in: usize,
        n_out: usize,
        i: usize,
        j: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let x = x.as_ptr();
        let weight = weight.as_ptr();
        let groups = n_in.div_ceil(group);
        let mut total = [[0.0f32; C]; R];
        for g in 0..groups {
            let start = g * group;
            let group_end = (start + group).min(n_in);
            let mut acc = [[f32x4_splat(0.0); C]; R];
            let mut extra = [[0.0f32; C]; R];
            let mut p = start;
            while p + 8 <= group_end {
                let mut xv = [[f32x4_splat(0.0); 2]; R];
                for (r, value) in xv.iter_mut().enumerate() {
                    *value = [
                        load(x.add((i + r) * n_in + p)),
                        load(x.add((i + r) * n_in + p + 4)),
                    ];
                }
                for c in 0..C {
                    let w = widen8(weight.add((j + c) * n_in + p));
                    for (r, value) in xv.iter().enumerate() {
                        acc[r][c] = multiply_add(acc[r][c], w[0], value[0]);
                        acc[r][c] = multiply_add(acc[r][c], w[1], value[1]);
                    }
                }
                p += 8;
            }
            while p < group_end {
                for (r, extra_r) in extra.iter_mut().enumerate() {
                    for (c, extra_rc) in extra_r.iter_mut().enumerate() {
                        *extra_rc +=
                            *x.add((i + r) * n_in + p) * (*weight.add((j + c) * n_in + p)) as f32;
                    }
                }
                p += 1;
            }
            for (r, extra_r) in extra.iter().enumerate() {
                for (c, extra_rc) in extra_r.iter().enumerate() {
                    total[r][c] +=
                        (horizontal_sum(acc[r][c]) + *extra_rc) * scale[(j + c) * groups + g];
                }
            }
        }
        for (r, total_r) in total.iter().enumerate() {
            for (c, value) in total_r.iter().enumerate() {
                y[(i + r) * n_out + j + c] = finish(*value, Some(bias[j + c]), with_gelu);
            }
        }
    }

    // SAFETY: `x` has `rows * n_in`, `weight` has `n_out * n_in`, `scale` has
    // `n_out * n_in.div_ceil(group)`, `bias` has `n_out`, and `y` has
    // `rows * n_out` values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn int8_prefill(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        group: usize,
        rows: usize,
        n_in: usize,
        n_out: usize,
        with_gelu: bool,
        y: &mut [f32],
    ) {
        let mut i = 0;
        while i + INT8_PREFILL_TOKENS <= rows {
            let mut j = 0;
            while j + INT8_PREFILL_OUTPUTS <= n_out {
                int8_tile::<INT8_PREFILL_TOKENS, INT8_PREFILL_OUTPUTS>(
                    x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                );
                j += INT8_PREFILL_OUTPUTS;
            }
            while j < n_out {
                int8_tile::<INT8_PREFILL_TOKENS, 1>(
                    x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                );
                j += 1;
            }
            i += INT8_PREFILL_TOKENS;
        }
        while i < rows {
            let mut j = 0;
            while j < n_out {
                int8_tile::<1, 1>(
                    x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                );
                j += 1;
            }
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_rows(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        // SAFETY: the slices are sized by the caller for `rows * n_in`,
        // `n_out * n_in`, `n_out` and `rows * n_out`.
        unsafe {
            f32_prefill(
                x,
                weight,
                bias,
                rows,
                n_in,
                n_out,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_outputs(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        // SAFETY: `weight` holds `n_out * n_in` values and `y` holds this output
        // range; `bias` is sliced with the outputs.
        unsafe {
            f32_outputs(
                x,
                weight,
                bias,
                n_in,
                outputs.start,
                outputs.end,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_rows(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        // SAFETY: the slices are sized by the caller for `rows * n_in`,
        // `n_out * n_in`, the per-output scales, `n_out` and `rows * n_out`.
        unsafe {
            int8_prefill(
                x,
                weight,
                scale,
                bias,
                group,
                rows,
                n_in,
                n_out,
                matches!(epilogue, Epilogue::Gelu),
                y,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_outputs(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        // SAFETY: `weight` holds `n_out * n_in` values, `scale` the per-output
        // groups, and `y` holds this output range; `bias` is sliced with it.
        let groups = n_in.div_ceil(group);
        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        for (offset, row) in outputs.enumerate() {
            let row_weight = &weight[row * n_in..(row + 1) * n_in];
            let row_scale = &scale[row * groups..(row + 1) * groups];
            let sum = int8_dot(x, row_weight, row_scale, group);
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
        }
    }

    pub(crate) fn int8_dot(x: &[f32], weight: &[i8], scale: &[f32], group: usize) -> f32 {
        assert_eq!(x.len(), weight.len(), "dot product needs equal lengths");
        let n = x.len();
        let x = x.as_ptr();
        let weight = weight.as_ptr();

        // SAFETY: every load stays within `0..n`, and the scalar tail handles the rest.
        unsafe {
            let mut total = 0.0;
            let mut p = 0;
            while p < n {
                let group_start = p / group * group;
                let group_end = (group_start + group).min(n);
                let mut acc = [f32x4_splat(0.0); 8];
                while p + 32 <= group_end {
                    let g0 = widen8(weight.add(p));
                    let g1 = widen8(weight.add(p + 8));
                    let g2 = widen8(weight.add(p + 16));
                    let g3 = widen8(weight.add(p + 24));
                    for k in 0..2 {
                        acc[k] = multiply_add(acc[k], g0[k], load(x.add(p + k * 4)));
                        acc[2 + k] = multiply_add(acc[2 + k], g1[k], load(x.add(p + 8 + k * 4)));
                        acc[4 + k] = multiply_add(acc[4 + k], g2[k], load(x.add(p + 16 + k * 4)));
                        acc[6 + k] = multiply_add(acc[6 + k], g3[k], load(x.add(p + 24 + k * 4)));
                    }
                    p += 32;
                }
                while p + 16 <= group_end {
                    let g0 = widen8(weight.add(p));
                    let g1 = widen8(weight.add(p + 8));
                    for k in 0..2 {
                        acc[k] = multiply_add(acc[k], g0[k], load(x.add(p + k * 4)));
                        acc[2 + k] = multiply_add(acc[2 + k], g1[k], load(x.add(p + 8 + k * 4)));
                    }
                    p += 16;
                }
                let mut sum = horizontal8(acc);
                while p < group_end {
                    sum += *x.add(p) * *weight.add(p) as f32;
                    p += 1;
                }
                total += sum * *scale.get_unchecked(group_start / group);
            }
            total
        }
    }

    pub(crate) fn weighted_sum(values: &[f32], stride: usize, weights: &[f32], output: &mut [f32]) {
        let width = output.len();
        assert!(width <= stride, "output is wider than the value stride");
        assert!(
            values.len() >= weights.len().saturating_sub(1) * stride + width,
            "values are shorter than the weighted sum needs"
        );

        let values = values.as_ptr();
        let mut offset = 0;
        while offset + 32 <= width {
            // SAFETY: `offset + 32 <= width <= stride`, so each row read is inside
            // `values`, and `output` has `width >= offset + 32` elements.
            unsafe {
                let mut acc = [f32x4_splat(0.0); 8];
                for (past, &weight) in weights.iter().enumerate() {
                    let row = values.add(past * stride + offset);
                    let weight = f32x4_splat(weight);
                    for (c, acc_c) in acc.iter_mut().enumerate() {
                        *acc_c = multiply_add(*acc_c, weight, load(row.add(c * 4)));
                    }
                }
                for (c, acc_c) in acc.iter().enumerate() {
                    store(output.as_mut_ptr().add(offset + c * 4), *acc_c);
                }
            }
            offset += 32;
        }
        while offset < width {
            let mut sum = 0.0;
            for (past, &weight) in weights.iter().enumerate() {
                // SAFETY: `offset < width <= stride` and the input length check above.
                sum += weight * unsafe { *values.add(past * stride + offset) };
            }
            output[offset] = sum;
            offset += 1;
        }
    }

    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dot product needs equal lengths");
        let n = a.len();
        let a = a.as_ptr();
        let b = b.as_ptr();

        // SAFETY: every load stays within `0..n`, and the scalar tail handles the rest.
        unsafe {
            let mut acc = [f32x4_splat(0.0); 8];
            let mut i = 0;
            while i + 32 <= n {
                for (k, acc_k) in acc.iter_mut().enumerate() {
                    *acc_k = multiply_add(*acc_k, load(a.add(i + k * 4)), load(b.add(i + k * 4)));
                }
                i += 32;
            }
            let mut sum = horizontal8(acc);
            while i < n {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            sum
        }
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod scalar {
    use std::ops::Range;

    use super::{gelu, Epilogue};

    #[inline]
    fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
        let value = sum + bias.unwrap_or(0.0);
        if with_gelu {
            gelu(value)
        } else {
            value
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_rows(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        for i in 0..rows {
            let x = &x[i * n_in..(i + 1) * n_in];
            for j in 0..n_out {
                let sum: f32 = x
                    .iter()
                    .zip(&weight[j * n_in..(j + 1) * n_in])
                    .map(|(a, b)| a * b)
                    .sum();
                y[i * n_out + j] = finish(sum, Some(bias[j]), with_gelu);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_outputs(
        x: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        for (offset, row) in outputs.enumerate() {
            let sum: f32 = x
                .iter()
                .zip(&weight[row * n_in..(row + 1) * n_in])
                .map(|(a, b)| a * b)
                .sum();
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_rows(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        let groups = n_in.div_ceil(group);
        for i in 0..rows {
            for j in 0..n_out {
                let mut sum = 0.0;
                for g in 0..groups {
                    let start = g * group;
                    let end = (start + group).min(n_in);
                    let mut partial = 0.0;
                    for p in start..end {
                        partial += x[i * n_in + p] * weight[j * n_in + p] as f32;
                    }
                    sum += partial * scale[j * groups + g];
                }
                y[i * n_out + j] = finish(sum, Some(bias[j]), with_gelu);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_outputs(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        let groups = n_in.div_ceil(group);
        for (offset, row) in outputs.enumerate() {
            let row_weight = &weight[row * n_in..(row + 1) * n_in];
            let row_scale = &scale[row * groups..(row + 1) * groups];
            let sum = int8_dot(x, row_weight, row_scale, group);
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
        }
    }

    pub(crate) fn int8_dot(x: &[f32], weight: &[i8], scale: &[f32], group: usize) -> f32 {
        assert_eq!(x.len(), weight.len(), "dot product needs equal lengths");
        let mut total = 0.0;
        let mut p = 0;
        while p < x.len() {
            let end = (p / group * group + group).min(x.len());
            let mut partial = 0.0;
            for q in p..end {
                partial += x[q] * weight[q] as f32;
            }
            total += partial * scale[p / group];
            p = end;
        }
        total
    }

    pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "dot product needs equal lengths");
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    pub(crate) fn weighted_sum(values: &[f32], stride: usize, weights: &[f32], output: &mut [f32]) {
        let width = output.len();
        assert!(width <= stride, "output is wider than the value stride");
        assert!(
            values.len() >= weights.len().saturating_sub(1) * stride + width,
            "values are shorter than the weighted sum needs"
        );
        for (offset, value) in output.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (past, &weight) in weights.iter().enumerate() {
                sum += weight * values[past * stride + offset];
            }
            *value = sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{
    dot, int8_linear_outputs, int8_linear_rows, linear_outputs, linear_rows, weighted_sum,
};

#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) use neon::int8_dot;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::{
    dot, int8_linear_outputs, int8_linear_rows, linear_outputs, linear_rows, weighted_sum,
};

#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) use wasm::int8_dot;

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
pub(crate) use scalar::{
    dot, int8_linear_outputs, int8_linear_rows, linear_outputs, linear_rows, weighted_sum,
};

#[cfg(all(
    test,
    not(any(
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))
))]
pub(crate) use scalar::int8_dot;

#[cfg(test)]
mod tests {
    use super::{
        dot, int8_dot, int8_linear_outputs, int8_linear_rows, linear_outputs, linear_rows,
        weighted_sum, Epilogue,
    };

    fn linear(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0; rows * n_out];
        if rows == 1 {
            linear_outputs(
                x,
                weight,
                Some(bias),
                n_in,
                0..n_out,
                &mut y,
                Epilogue::Store,
            );
        } else {
            linear_rows(x, weight, bias, rows, n_in, n_out, &mut y, Epilogue::Store);
        }
        y
    }

    fn close(actual: f32, expected: f32) {
        let tolerance = 1e-5 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn reference(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0; rows * n_out];
        for i in 0..rows {
            for j in 0..n_out {
                let mut sum = 0.0;
                for p in 0..n_in {
                    sum += x[i * n_in + p] * weight[j * n_in + p];
                }
                y[i * n_out + j] = sum + bias[j];
            }
        }
        y
    }

    #[test]
    fn linear_matches_reference_on_odd_sizes() {
        for (rows, n_in, n_out) in [
            (1, 1, 1),
            (1, 37, 29),
            (2, 37, 29),
            (5, 37, 29),
            (7, 13, 17),
            (11, 37, 29),
            (17, 37, 29),
            (5, 64, 65),
            (9, 1, 33),
            (33, 97, 130),
        ] {
            let x: Vec<f32> = (0..rows * n_in)
                .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
                .collect();
            let weight: Vec<f32> = (0..n_in * n_out)
                .map(|i| ((i * 53 + 7) % 97) as f32 / 97.0 - 0.5)
                .collect();
            let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 90.0 - 1.0).collect();

            let actual = linear(&x, &weight, &bias, rows, n_in, n_out);
            let expected = reference(&x, &weight, &bias, rows, n_in, n_out);
            assert_eq!(actual.len(), expected.len());
            for (a, e) in actual.iter().zip(&expected) {
                close(*a, *e);
            }
        }
    }

    #[test]
    fn weighted_sum_matches_reference_on_odd_sizes() {
        for (count, stride, width) in [
            (1, 7, 7),
            (3, 13, 7),
            (5, 13, 13),
            (17, 64, 64),
            (3, 64, 40),
        ] {
            let values: Vec<f32> = (0..(count - 1) * stride + width)
                .map(|i| ((i * 31 + 17) % 83) as f32 / 83.0 - 0.5)
                .collect();
            let weights: Vec<f32> = (0..count)
                .map(|i| ((i * 47 + 5) % 73) as f32 / 73.0 - 0.5)
                .collect();
            let mut actual = vec![0.0; width];
            weighted_sum(&values, stride, &weights, &mut actual);
            for (offset, a) in actual.iter().enumerate() {
                let expected: f32 = weights
                    .iter()
                    .enumerate()
                    .map(|(past, weight)| weight * values[past * stride + offset])
                    .sum();
                close(*a, expected);
            }
        }
    }

    #[test]
    fn dot_matches_reference_on_odd_sizes() {
        for n in [1, 3, 15, 16, 17, 31, 37, 64, 769] {
            let a: Vec<f32> = (0..n)
                .map(|i| ((i * 29 + 3) % 71) as f32 / 71.0 - 0.5)
                .collect();
            let b: Vec<f32> = (0..n)
                .map(|i| ((i * 43 + 5) % 67) as f32 / 67.0 - 0.5)
                .collect();
            let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            close(dot(&a, &b), expected);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_linear(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        epilogue: Epilogue,
        group: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0; rows * n_out];
        if rows == 1 {
            int8_linear_outputs(
                x,
                weight,
                scale,
                Some(bias),
                n_in,
                0..n_out,
                &mut y,
                epilogue,
                group,
            );
        } else {
            int8_linear_rows(
                x, weight, scale, bias, rows, n_in, n_out, &mut y, epilogue, group,
            );
        }
        y
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_reference(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        gelu_epilogue: bool,
        group: usize,
    ) -> Vec<f32> {
        let mut y = vec![0.0; rows * n_out];
        let groups = n_in.div_ceil(group);
        for i in 0..rows {
            for j in 0..n_out {
                let mut sum = 0.0f64;
                for g in 0..groups {
                    let start = g * group;
                    let end = (start + group).min(n_in);
                    let mut partial = 0.0f64;
                    for p in start..end {
                        partial += x[i * n_in + p] as f64 * weight[j * n_in + p] as f64;
                    }
                    sum += partial * scale[j * groups + g] as f64;
                }
                let value = sum as f32 + bias[j];
                y[i * n_out + j] = if gelu_epilogue {
                    super::gelu(value)
                } else {
                    value
                };
            }
        }
        y
    }

    #[test]
    fn int8_linear_matches_dequantized_reference_on_odd_sizes() {
        for (rows, n_in, n_out, group) in [
            (1usize, 1usize, 1usize, 32usize),
            (1, 37, 29, 7),
            (1, 64, 33, 32),
            (2, 37, 29, 7),
            (5, 37, 29, 16),
            (7, 65, 33, 64),
            (4, 96, 17, 32),
            (9, 33, 64, 8),
        ] {
            let x: Vec<f32> = (0..rows * n_in)
                .map(|i| ((i * 37 + 11) % 101) as f32 / 101.0 - 0.5)
                .collect();
            let weight: Vec<i8> = (0..n_in * n_out)
                .map(|i| ((i * 53 + 7) % 255) as i32 as i8)
                .collect();
            let groups = n_in.div_ceil(group);
            let scale: Vec<f32> = (0..groups * n_out)
                .map(|j| (j % 7 + 1) as f32 / 200.0)
                .collect();
            let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 90.0 - 1.0).collect();

            for epilogue in [Epilogue::Store, Epilogue::Gelu] {
                let actual = int8_linear(
                    &x, &weight, &scale, &bias, rows, n_in, n_out, epilogue, group,
                );
                let expected = int8_reference(
                    &x,
                    &weight,
                    &scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    matches!(epilogue, Epilogue::Gelu),
                    group,
                );
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    let tolerance = 1e-4 * e.abs().max(1.0);
                    assert!(
                        (a - e).abs() <= tolerance,
                        "rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}: expected {e}, got {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn int8_dot_matches_dequantized_reference_on_odd_sizes() {
        for (n, group) in [
            (1usize, 32usize),
            (3, 1),
            (15, 7),
            (16, 32),
            (17, 16),
            (31, 8),
            (32, 32),
            (33, 64),
            (64, 32),
            (769, 128),
        ] {
            let a: Vec<f32> = (0..n)
                .map(|i| ((i * 29 + 3) % 71) as f32 / 71.0 - 0.5)
                .collect();
            let b: Vec<i8> = (0..n).map(|i| ((i * 43 + 5) % 255) as i8).collect();
            let groups = n.div_ceil(group);
            let scale: Vec<f32> = (0..groups).map(|g| (g % 5 + 1) as f32 / 100.0).collect();
            let expected: f64 = (0..groups)
                .map(|g| {
                    let end = (g * group + group).min(n);
                    let partial: f64 = (g * group..end).map(|p| a[p] as f64 * b[p] as f64).sum();
                    partial * scale[g] as f64
                })
                .sum();
            let actual = int8_dot(&a, &b, &scale, group) as f64;
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "n {n}, group {group}: expected {expected}, got {actual}"
            );
        }
    }
}
