#[derive(Clone, Copy)]
pub(crate) enum Epilogue {
    Store,
    Gelu,
}

#[derive(Clone, Copy)]
pub(crate) struct Outliers<'a> {
    pub indices: &'a [u32],
    pub values: &'a [f32],
}

pub(crate) fn gelu_exact(x: f32) -> f32 {
    let coefficient = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (coefficient * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(target_arch = "aarch64")]
const GELU_COEFFICIENT: f32 = 0.7978845608028654;

// tanh(y) for |y| <= 0.75 is y times a degree-6 polynomial in y^2; beyond it,
// 1 - 2/(exp(2|y|) + 1) with the exponential range-reduced to [-ln2/2, ln2/2].
// Worst error against f64 tanh is one f32 ulp, and libm's tanhf costs roughly
// two hundred cycles more per value.
#[cfg(target_arch = "aarch64")]
const TANH_SMALL: [f32; 7] = [
    1.0000000000e+00,
    -3.3333332411e-01,
    1.3333223298e-01,
    -5.3938657908e-02,
    2.1621819977e-02,
    -7.9409813272e-03,
    1.9129803016e-03,
];

#[cfg(target_arch = "aarch64")]
const EXP: [f32; 9] = [
    1.0000000000e+00,
    1.0000000000e+00,
    4.9999999981e-01,
    1.6666666680e-01,
    4.1666700572e-02,
    8.3333087795e-03,
    1.3882306881e-03,
    1.9893930896e-04,
    2.8073170020e-05,
];

#[cfg(target_arch = "aarch64")]
pub(crate) fn gelu_in_place(y: &mut [f32]) {
    use std::arch::aarch64::*;

    let (chunks, tail) = y.as_chunks_mut::<4>();
    // SAFETY: each chunk holds exactly four floats; the constants are compile-time.
    unsafe {
        let coefficient = vdupq_n_f32(GELU_COEFFICIENT);
        let cubic = vdupq_n_f32(0.044715);
        let two = vdupq_n_f32(2.0);
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let limit = vdupq_n_f32(0.75);
        let clamp = vdupq_n_f32(40.0);
        let log2 = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let bias = vdupq_n_s32(127);
        let mut small = [vdupq_n_f32(0.0); 7];
        for (value, coefficient) in small.iter_mut().zip(TANH_SMALL) {
            *value = vdupq_n_f32(coefficient);
        }
        let mut exp = [vdupq_n_f32(0.0); 9];
        for (value, coefficient) in exp.iter_mut().zip(EXP) {
            *value = vdupq_n_f32(coefficient);
        }

        for chunk in chunks {
            let x = vld1q_f32(chunk.as_ptr());
            let cube = vmulq_f32(vmulq_f32(x, x), x);
            let argument = vmulq_f32(coefficient, vfmaq_f32(x, cubic, cube));

            let magnitude = vabsq_f32(argument);
            let squared = vmulq_f32(argument, argument);
            let mut polynomial = small[6];
            for value in small[..6].iter().rev() {
                polynomial = vfmaq_f32(*value, polynomial, squared);
            }
            let near = vmulq_f32(argument, polynomial);

            let reduced = vminq_f32(vmulq_f32(magnitude, two), clamp);
            let exponent = vcvtnq_s32_f32(vmulq_f32(reduced, log2));
            let remainder = vfmaq_f32(
                reduced,
                vcvtq_f32_s32(exponent),
                vnegq_f32(ln2),
            );
            let mut series = exp[8];
            for value in exp[..8].iter().rev() {
                series = vfmaq_f32(*value, series, remainder);
            }
            let power = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(exponent, bias), 23));
            let exponential = vmulq_f32(series, power);
            let magnitude_tanh = vsubq_f32(
                one,
                vdivq_f32(two, vaddq_f32(exponential, one)),
            );
            let sign = vandq_u32(
                vreinterpretq_u32_f32(argument),
                vdupq_n_u32(0x8000_0000),
            );
            let far = vreinterpretq_f32_u32(vorrq_u32(
                vreinterpretq_u32_f32(magnitude_tanh),
                sign,
            ));

            let tanh = vbslq_f32(vcleq_f32(magnitude, limit), near, far);
            let value = vmulq_f32(vmulq_f32(half, x), vaddq_f32(one, tanh));
            vst1q_f32(chunk.as_mut_ptr(), value);
        }
    }
    for value in tail {
        *value = gelu_exact(*value);
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn gelu_in_place(y: &mut [f32]) {
    for value in y.iter_mut() {
        *value = gelu_exact(*value);
    }
}

#[inline]
fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
    let value = sum + bias.unwrap_or(0.0);
    if with_gelu {
        gelu_exact(value)
    } else {
        value
    }
}

// The integer dot product is exact; its conversion to f32 and the two multiplies
// and add happen in this order so the scalar and dot-product kernels produce the
// same bits.
#[inline]
pub(crate) fn add_scaled_dot(
    total: f32,
    dot: i32,
    weight_scale: f32,
    activation_scale: f32,
) -> f32 {
    let combined = weight_scale * activation_scale;
    total + dot as f32 * combined
}

// Removed activations keep their exact value, so they are added back in ascending
// input order for the scalar and vector kernels to produce the same bits.
#[inline]
fn add_outliers(
    total: f32,
    outliers: Outliers,
    weight: &[i8],
    weight_scale: &[f32],
    n_in: usize,
    row: usize,
) -> f32 {
    let weight_groups = n_in.div_ceil(crate::quantize::GROUP_SIZE);
    let stride = crate::int8_layout::pair_stride(n_in);
    let base = (row / 2) * stride + (row % 2) * crate::int8_layout::BLOCK;
    let scale = &weight_scale[row * weight_groups..(row + 1) * weight_groups];
    let mut total = total;
    for (&index, &value) in outliers.indices.iter().zip(outliers.values) {
        let index = index as usize;
        let at = base
            + (index / crate::int8_layout::BLOCK) * 2 * crate::int8_layout::BLOCK
            + index % crate::int8_layout::BLOCK;
        total += value
            * (scale[index / crate::quantize::GROUP_SIZE] * weight[at] as f32);
    }
    total
}

// The two rows of an output pair share every outlier, so one weight block read
// serves both corrections while each row still adds them in index order. The
// block offset and weight group of an outlier are the same for every output
// row, so the caller resolves them once.
#[cfg(target_arch = "aarch64")]
#[inline]
fn add_outliers_pair(
    total: [f32; 2],
    outliers: &[(usize, usize, f32)],
    weight: &[i8],
    weight_scale: &[f32],
    n_in: usize,
    row: usize,
) -> [f32; 2] {
    let weight_groups = n_in.div_ceil(crate::quantize::GROUP_SIZE);
    let stride = crate::int8_layout::pair_stride(n_in);
    let base = (row / 2) * stride;
    let first = &weight_scale[row * weight_groups..(row + 1) * weight_groups];
    let second = &weight_scale[(row + 1) * weight_groups..(row + 2) * weight_groups];
    let mut total = total;
    let apply = |total: &mut [f32; 2], offset: usize, group: usize, value: f32| {
        let at = base + offset;
        total[0] += value * (first[group] * weight[at] as f32);
        total[1] += value * (second[group] * weight[at + crate::int8_layout::BLOCK] as f32);
    };
    match outliers {
        [(o0, g0, v0), (o1, g1, v1), (o2, g2, v2), (o3, g3, v3)] => {
            apply(&mut total, *o0, *g0, *v0);
            apply(&mut total, *o1, *g1, *v1);
            apply(&mut total, *o2, *g2, *v2);
            apply(&mut total, *o3, *g3, *v3);
        }
        _ => {
            for &(offset, group, value) in outliers {
                apply(&mut total, offset, group, value);
            }
        }
    }
    total
}

mod int8_activations {
    use std::ops::Range;

    use super::{add_outliers, add_scaled_dot, finish, Epilogue, Outliers};
    use crate::int8_layout;
    use crate::quantize::GROUP_SIZE;

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_outputs(
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        group: usize,
        outputs: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        outliers: Option<Outliers>,
    ) {
        let weight_groups = n_in.div_ceil(GROUP_SIZE);
        let activation_groups = n_in.div_ceil(group);
        assert!(activations.len() >= n_in, "activation row is too short");
        assert!(activation_scale.len() >= activation_groups);
        assert!(weight.len() >= outputs.end * n_in, "weights are too short");
        assert!(weight_scale.len() >= outputs.end * weight_groups);
        assert!(y.len() >= outputs.len(), "output is too short");

        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        for (offset, row) in outputs.enumerate() {
            let scale_row = &weight_scale[row * weight_groups..(row + 1) * weight_groups];
            let mut sum = row_value(
                |p| activations[p],
                activation_scale,
                weight,
                scale_row,
                n_in,
                group,
                row,
            );
            if let Some(outliers) = outliers {
                sum = add_outliers(sum, outliers, weight, weight_scale, n_in, row);
            }
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_rows(
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        group: usize,
        y: &mut [f32],
        epilogue: Epilogue,
        outliers: Option<&[Outliers]>,
    ) {
        let weight_groups = n_in.div_ceil(GROUP_SIZE);
        let activation_groups = n_in.div_ceil(group);
        assert!(activations.len() >= rows * n_in);
        assert!(activation_scale.len() >= rows * activation_groups);
        assert!(weight.len() >= n_out * n_in, "weights are too short");
        assert!(weight_scale.len() >= n_out * weight_groups);
        assert!(bias.len() >= n_out);
        assert!(y.len() >= rows * n_out, "output is too short");
        if let Some(outliers) = outliers {
            assert_eq!(outliers.len(), rows, "one outlier list per row");
        }

        let with_gelu = matches!(epilogue, Epilogue::Gelu);
        for row in 0..rows {
            let activation_scale_row =
                &activation_scale[row * activation_groups..(row + 1) * activation_groups];
            for output in 0..n_out {
                let weight_scale_row =
                    &weight_scale[output * weight_groups..(output + 1) * weight_groups];
                let sum = row_value(
                    |p| int8_layout::byte(activations, n_in, row, p),
                    activation_scale_row,
                    weight,
                    weight_scale_row,
                    n_in,
                    group,
                    output,
                );
                let sum = match outliers {
                    Some(outliers) => {
                        add_outliers(sum, outliers[row], weight, weight_scale, n_in, output)
                    }
                    None => sum,
                };
                y[row * n_out + output] = finish(sum, Some(bias[output]), with_gelu);
            }
        }
    }

    // The closure reads one activation byte; the weights are read through the
    // pair layout.
    #[inline]
    fn row_value(
        activation: impl Fn(usize) -> i8,
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        n_in: usize,
        group: usize,
        output: usize,
    ) -> f32 {
        let weight_groups = n_in.div_ceil(GROUP_SIZE);
        debug_assert!(activation_scale.len() >= n_in.div_ceil(group));
        debug_assert!(weight.len() >= n_in);
        debug_assert!(weight_scale.len() >= weight_groups);

        let mut total = 0.0f32;
        for weight_group in 0..weight_groups {
            let start = weight_group * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(n_in);
            let weight_scale = weight_scale[weight_group];
            let first = start / group;
            let last = (end - 1) / group;
            for (activation_group, &activation_scale) in activation_scale
                .iter()
                .enumerate()
                .take(last + 1)
                .skip(first)
            {
                let group_start = activation_group * group;
                let group_end = (group_start + group).min(n_in);
                let mut dot = 0i32;
                for p in group_start..group_end {
                    dot += activation(p) as i32
                        * int8_layout::byte(weight, n_in, output, p) as i32;
                }
                total = add_scaled_dot(total, dot, weight_scale, activation_scale);
            }
        }
        total
    }
}

#[cfg(target_arch = "aarch64")]
mod int8_activations_neon {
    use std::arch::aarch64::*;

    use super::{add_outliers, add_outliers_pair, add_scaled_dot, finish, Outliers};
    use crate::int8_layout;
    use crate::quantize::GROUP_SIZE;

    const DECODE_PAIRS: usize = 4;
    // A block of output rows stays in cache while every prompt row passes over it.
    const PREFILL_OUTPUT_BLOCK: usize = 32;
    const PREFILL_ROWS: usize = 8;
    const PREFILL_OUTPUTS: usize = 4;

    #[inline]
    #[target_feature(enable = "neon,i8mm")]
    unsafe fn smmla(acc: int32x4_t, left: int8x16_t, right: int8x16_t) -> int32x4_t {
        let mut out = acc;
        std::arch::asm!(
            "smmla {out:v}.4s, {left:v}.16b, {right:v}.16b",
            out = inout(vreg) out,
            left = in(vreg) left,
            right = in(vreg) right,
            options(pure, nomem, nostack),
        );
        out
    }

    // One output row against one activation row. The activation row is either
    // `stride` bytes apart per eight columns (8 when plain, 16 when paired).
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn single_output(
        activations: *const i8,
        activation_stride: usize,
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        group: usize,
        output: usize,
        y: &mut [f32],
        y_base: usize,
        bias_base: usize,
        with_gelu: bool,
        outliers: Option<Outliers>,
    ) {
        let stride = int8_layout::pair_stride(n_in);
        let pair_base = (output / 2) * stride;
        let half = (output % 2) * int8_layout::BLOCK;
        let weight_groups = n_in.div_ceil(GROUP_SIZE);
        let weight_data = weight;
        let weight = weight.as_ptr();
        let mut total = 0.0f32;

        for weight_group in 0..weight_groups {
            let start = weight_group * GROUP_SIZE;
            let end = (start + GROUP_SIZE).min(n_in);
            let weight_scale_row = *weight_scale.get_unchecked(output * weight_groups + weight_group);
            for activation_group in start / group..=(end - 1) / group {
                let group_start = activation_group * group;
                let group_end = (group_start + group).min(n_in);
                let mut acc = vdupq_n_s32(0);
                let mut p = group_start;
                while p + int8_layout::BLOCK <= group_end {
                    let chunk = vld1_s8(activations.add((p / int8_layout::BLOCK) * activation_stride));
                    let input = vcombine_s8(chunk, chunk);
                    let weights = vld1q_s8(
                        weight.add(pair_base + (p / int8_layout::BLOCK) * 2 * int8_layout::BLOCK),
                    );
                    acc = vdotq_s32(acc, input, weights);
                    p += int8_layout::BLOCK;
                }
                let dot = if p == group_end {
                    let sums = vpaddq_s32(acc, acc);
                    if half == 0 {
                        vgetq_lane_s32::<0>(sums)
                    } else {
                        vgetq_lane_s32::<1>(sums)
                    }
                } else {
                    let mut lanes = [0i32; 4];
                    vst1q_s32(lanes.as_mut_ptr(), acc);
                    let mut dots = [lanes[0] + lanes[1], lanes[2] + lanes[3]];
                    while p < group_end {
                        let a = *activations.add(
                            (p / int8_layout::BLOCK) * activation_stride + p % int8_layout::BLOCK,
                        ) as i32;
                        let at = pair_base
                            + (p / int8_layout::BLOCK) * 2 * int8_layout::BLOCK
                            + p % int8_layout::BLOCK;
                        dots[0] += a * *weight.add(at) as i32;
                        dots[1] += a * *weight.add(at + int8_layout::BLOCK) as i32;
                        p += 1;
                    }
                    dots[(half != 0) as usize]
                };
                total = add_scaled_dot(
                    total,
                    dot,
                    weight_scale_row,
                    *activation_scale.get_unchecked(activation_group),
                );
            }
        }
        let total = match outliers {
            Some(outliers) => {
                add_outliers(total, outliers, weight_data, weight_scale, n_in, output)
            }
            None => total,
        };
        *y.get_unchecked_mut(output - y_base) = finish(
            total,
            bias.map(|bias| *bias.get_unchecked(output - bias_base)),
            with_gelu,
        );
    }

    // 2 * P output rows against one activation row, one eight-byte block at a
    // time. One weight vector holds both rows of an output pair.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn row_outputs<const P: usize>(
        activations: *const i8,
        activation_stride: usize,
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        group: usize,
        first: usize,
        y: &mut [f32],
        y_base: usize,
        bias_base: usize,
        with_gelu: bool,
        outliers: Option<&[(usize, usize, f32)]>,
    ) {
        let stride = int8_layout::pair_stride(n_in);
        let weight_groups = n_in.div_ceil(GROUP_SIZE);
        let weight_ptr = weight.as_ptr();
        let mut total = [vdupq_n_f32(0.0); P];

        for weight_group in 0..weight_groups {
            let start = weight_group * GROUP_SIZE;
            let weight_end = (start + GROUP_SIZE).min(n_in);
            let mut scales = [[0.0f32; 4]; P];
            for (pair, scale) in scales.iter_mut().enumerate() {
                let output = first + 2 * pair;
                let first_scale = *weight_scale.get_unchecked(output * weight_groups + weight_group);
                let second_scale =
                    *weight_scale.get_unchecked((output + 1) * weight_groups + weight_group);
                *scale = [first_scale, second_scale, first_scale, second_scale];
            }
            for activation_group in start / group..=(weight_end - 1) / group {
                let group_start = activation_group * group;
                let group_end = (group_start + group).min(n_in);
                let mut acc = [vdupq_n_s32(0); P];
                let mut p = group_start;
                while activation_stride == 8 && p + 16 <= group_end {
                    let x16 = vld1q_s8(activations.add((p / int8_layout::BLOCK) * activation_stride));
                    let x0 = vcombine_s8(vget_low_s8(x16), vget_low_s8(x16));
                    let x1 = vcombine_s8(vget_high_s8(x16), vget_high_s8(x16));
                    for (pair, accumulated) in acc.iter_mut().enumerate() {
                        let base = ((first / 2) + pair) * stride + (p / int8_layout::BLOCK) * 16;
                        *accumulated =
                            vdotq_s32(*accumulated, x0, vld1q_s8(weight_ptr.add(base)));
                        *accumulated =
                            vdotq_s32(*accumulated, x1, vld1q_s8(weight_ptr.add(base + 16)));
                    }
                    p += 16;
                }
                while p + int8_layout::BLOCK <= group_end {
                    let chunk =
                        vld1_s8(activations.add((p / int8_layout::BLOCK) * activation_stride));
                    let input = vcombine_s8(chunk, chunk);
                    for (pair, accumulated) in acc.iter_mut().enumerate() {
                        let weights = vld1q_s8(
                            weight_ptr.add(
                                ((first / 2) + pair) * stride
                                    + (p / int8_layout::BLOCK) * 2 * int8_layout::BLOCK,
                            ),
                        );
                        *accumulated = vdotq_s32(*accumulated, input, weights);
                    }
                    p += int8_layout::BLOCK;
                }
                let scale = *activation_scale.get_unchecked(activation_group);
                for (pair, accumulated) in acc.iter().enumerate() {
                    let sums = vpaddq_s32(*accumulated, *accumulated);
                    let sums = if p == group_end {
                        vcvtq_f32_s32(sums)
                    } else {
                        let mut dots = [0i32; 2];
                        vst1_s32(dots.as_mut_ptr(), vget_low_s32(sums));
                        let mut q = p;
                        while q < group_end {
                            let a = *activations.add(
                                (q / int8_layout::BLOCK) * activation_stride
                                    + q % int8_layout::BLOCK,
                            ) as i32;
                            let at = ((first / 2) + pair) * stride
                                + (q / int8_layout::BLOCK) * 2 * int8_layout::BLOCK
                                + q % int8_layout::BLOCK;
                            dots[0] += a * *weight_ptr.add(at) as i32;
                            dots[1] += a * *weight_ptr.add(at + int8_layout::BLOCK) as i32;
                            q += 1;
                        }
                        vcvtq_f32_s32(vcombine_s32(
                            vld1_s32(dots.as_ptr()),
                            vdup_n_s32(0),
                        ))
                    };
                    let combined = vmulq_f32(vld1q_f32(scales[pair].as_ptr()), vdupq_n_f32(scale));
                    total[pair] = vaddq_f32(total[pair], vmulq_f32(sums, combined));
                }
            }
        }

        for (pair, accumulated) in total.iter().enumerate() {
            let output = first + 2 * pair;
            let mut sums = [0.0f32; 4];
            vst1q_f32(sums.as_mut_ptr(), *accumulated);
            let mut pair_sums = [sums[0], sums[1]];
            if let Some(outliers) = outliers {
                pair_sums = add_outliers_pair(pair_sums, outliers, weight, weight_scale, n_in, output);
            }
            for (lane, sum) in pair_sums.iter().enumerate() {
                let output = output + lane;
                *y.get_unchecked_mut(output - y_base) = finish(
                    *sum,
                    bias.map(|bias| *bias.get_unchecked(output - bias_base)),
                    with_gelu,
                );
            }
        }
    }

    // SAFETY: the slices cover `n_in` inputs, the selected output rows' weights and
    // scales, and the outputs; the CPU has the dot-product extension.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn linear_outputs(
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        group: usize,
        first: usize,
        end: usize,
        with_gelu: bool,
        y: &mut [f32],
        outliers: Option<Outliers>,
    ) {
        let prepared: Option<Vec<(usize, usize, f32)>> = outliers.map(|outliers| {
            outliers
                .indices
                .iter()
                .zip(outliers.values)
                .map(|(&index, &value)| {
                    let index = index as usize;
                    (
                        (index / int8_layout::BLOCK) * 2 * int8_layout::BLOCK
                            + index % int8_layout::BLOCK,
                        index / GROUP_SIZE,
                        value,
                    )
                })
                .collect()
        });
        let prepared = prepared.as_deref();
        let activations = activations.as_ptr();
        let mut output = first;
        if output % 2 == 1 {
            single_output(
                activations,
                8,
                activation_scale,
                weight,
                weight_scale,
                bias,
                n_in,
                group,
                output,
                y,
                first,
                first,
                with_gelu,
                outliers,
            );
            output += 1;
        }
        while output + 2 * DECODE_PAIRS <= end {
            row_outputs::<DECODE_PAIRS>(
                activations,
                8,
                activation_scale,
                weight,
                weight_scale,
                bias,
                n_in,
                group,
                output,
                y,
                first,
                first,
                with_gelu,
                prepared,
            );
            output += 2 * DECODE_PAIRS;
        }
        while output + 4 <= end {
            row_outputs::<2>(
                activations,
                8,
                activation_scale,
                weight,
                weight_scale,
                bias,
                n_in,
                group,
                output,
                y,
                first,
                first,
                with_gelu,
                prepared,
            );
            output += 4;
        }
        while output + 2 <= end {
            row_outputs::<1>(
                activations,
                8,
                activation_scale,
                weight,
                weight_scale,
                bias,
                n_in,
                group,
                output,
                y,
                first,
                first,
                with_gelu,
                prepared,
            );
            output += 2;
        }
        while output < end {
            single_output(
                activations,
                8,
                activation_scale,
                weight,
                weight_scale,
                bias,
                n_in,
                group,
                output,
                y,
                first,
                first,
                with_gelu,
                outliers,
            );
            output += 1;
        }
    }

    // SAFETY: as `linear_outputs`, with `rows` activation rows.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn linear_rows_dotprod(
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        group: usize,
        with_gelu: bool,
        y: &mut [f32],
        outliers: Option<&[Outliers]>,
    ) {
        if rows == 0 {
            return;
        }
        let stride = int8_layout::pair_stride(n_in);
        let activation_groups = n_in.div_ceil(group);
        for row in 0..rows {
            let activations = activations
                .as_ptr()
                .add((row / 2) * stride + (row % 2) * int8_layout::BLOCK);
            let activation_stride = 16;
            let activation_scale = &activation_scale[row * activation_groups..];
            let output = &mut y[row * n_out..(row + 1) * n_out];
            let prepared: Option<Vec<(usize, usize, f32)>> = outliers.map(|outliers| {
                outliers[row]
                    .indices
                    .iter()
                    .zip(outliers[row].values)
                    .map(|(&index, &value)| {
                        let index = index as usize;
                        (
                            (index / int8_layout::BLOCK) * 2 * int8_layout::BLOCK
                                + index % int8_layout::BLOCK,
                            index / GROUP_SIZE,
                            value,
                        )
                    })
                    .collect()
            });
            let prepared = prepared.as_deref();
            let outliers = outliers.map(|outliers| outliers[row]);
            let mut first = 0;
            while first + 2 * DECODE_PAIRS <= n_out {
                row_outputs::<DECODE_PAIRS>(
                    activations,
                    activation_stride,
                    activation_scale,
                    weight,
                    weight_scale,
                    Some(bias),
                    n_in,
                    group,
                    first,
                    output,
                    0,
                    0,
                    with_gelu,
                    prepared,
                );
                first += 2 * DECODE_PAIRS;
            }
            while first + 4 <= n_out {
                row_outputs::<2>(
                    activations,
                    activation_stride,
                    activation_scale,
                    weight,
                    weight_scale,
                    Some(bias),
                    n_in,
                    group,
                    first,
                    output,
                    0,
                    0,
                    with_gelu,
                    prepared,
                );
                first += 4;
            }
            while first + 2 <= n_out {
                row_outputs::<1>(
                    activations,
                    activation_stride,
                    activation_scale,
                    weight,
                    weight_scale,
                    Some(bias),
                    n_in,
                    group,
                    first,
                    output,
                    0,
                    0,
                    with_gelu,
                    prepared,
                );
                first += 2;
            }
            while first < n_out {
                single_output(
                    activations,
                    activation_stride,
                    activation_scale,
                    weight,
                    weight_scale,
                    Some(bias),
                    n_in,
                    group,
                    first,
                    output,
                    0,
                    0,
                    with_gelu,
                    outliers,
                );
                first += 1;
            }
        }
    }

    struct Prefill<'a> {
        activations: &'a [i8],
        activation_scale: &'a [f32],
        weight: &'a [i8],
        weight_scale: &'a [f32],
        bias: &'a [f32],
        n_in: usize,
        n_out: usize,
        group: usize,
        with_gelu: bool,
        outliers: Option<&'a [Outliers<'a>]>,
    }

    impl Prefill<'_> {
        // A tile of RP row pairs and CP output pairs. `rows` and `outputs` are the
        // real counts, so the padded row of a final pair is never stored.
        // SAFETY: `i + 2 * RP <= rows + 1`, `j + 2 * CP <= n_out + 1`, the
        // activation and weight slices cover the padded pair layout, and the CPU
        // has FEAT_I8MM.
        #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
        #[target_feature(enable = "neon,i8mm")]
        unsafe fn tile<const RP: usize, const CP: usize>(
            &self,
            i: usize,
            j: usize,
            rows: usize,
            outputs: usize,
            y: &mut [f32],
        ) {
            let stride = int8_layout::pair_stride(self.n_in);
            let activation_base = (i / 2) * stride;
            let weight_base = (j / 2) * stride;
            let weight_groups = self.n_in.div_ceil(GROUP_SIZE);
            let activation_groups = self.n_in.div_ceil(self.group);
            let mut total = [[vdupq_n_f32(0.0); CP]; RP];

            for weight_group in 0..weight_groups {
                let start = weight_group * GROUP_SIZE;
                let weight_end = (start + GROUP_SIZE).min(self.n_in);
                let mut scales = [[0.0f32; 2]; CP];
                for (cp, scale) in scales.iter_mut().enumerate() {
                    if 2 * cp < outputs {
                        scale[0] = *self
                            .weight_scale
                            .get_unchecked((j + 2 * cp) * weight_groups + weight_group);
                    }
                    if 2 * cp + 1 < outputs {
                        scale[1] = *self
                            .weight_scale
                            .get_unchecked((j + 2 * cp + 1) * weight_groups + weight_group);
                    }
                }
                for activation_group in start / self.group..=(weight_end - 1) / self.group {
                    let group_start = activation_group * self.group;
                    let group_end = (group_start + self.group).min(self.n_in);
                    let mut acc = [[vdupq_n_s32(0); CP]; RP];
                    let first_block = group_start / int8_layout::BLOCK;
                    let last_block = group_end.div_ceil(int8_layout::BLOCK);
                    for block in first_block..last_block {
                        let mut input = [vdupq_n_s8(0); RP];
                        for (rp, value) in input.iter_mut().enumerate() {
                            *value = vld1q_s8(
                                self.activations
                                    .as_ptr()
                                    .add(activation_base + rp * stride + block * 2 * int8_layout::BLOCK),
                            );
                        }
                        for cp in 0..CP {
                            let weights = vld1q_s8(
                                self.weight.as_ptr().add(
                                    weight_base + cp * stride + block * 2 * int8_layout::BLOCK,
                                ),
                            );
                            for rp in 0..RP {
                                acc[rp][cp] = smmla(acc[rp][cp], input[rp], weights);
                            }
                        }
                    }
                    for (rp, accumulated) in acc.iter().enumerate() {
                        let row = i + 2 * rp;
                        let mut activation_scales = [0.0f32; 4];
                        if 2 * rp < rows {
                            activation_scales[0] = *self
                                .activation_scale
                                .get_unchecked(row * activation_groups + activation_group);
                            activation_scales[1] = activation_scales[0];
                        }
                        if 2 * rp + 1 < rows {
                            let scale = *self.activation_scale.get_unchecked(
                                (row + 1) * activation_groups + activation_group,
                            );
                            activation_scales[2] = scale;
                            activation_scales[3] = scale;
                        }
                        let activation_scales =
                            vld1q_f32(activation_scales.as_ptr());
                        for (cp, value) in accumulated.iter().enumerate() {
                            let combined = vmulq_f32(
                                vld1q_f32(
                                    [
                                        scales[cp][0],
                                        scales[cp][1],
                                        scales[cp][0],
                                        scales[cp][1],
                                    ]
                                    .as_ptr(),
                                ),
                                activation_scales,
                            );
                            total[rp][cp] = vaddq_f32(
                                total[rp][cp],
                                vmulq_f32(vcvtq_f32_s32(*value), combined),
                            );
                        }
                    }
                }
            }

            for (rp, total_rp) in total.iter().enumerate() {
                for (cp, value) in total_rp.iter().enumerate() {
                    let mut sums = [0.0f32; 4];
                    vst1q_f32(sums.as_mut_ptr(), *value);
                    for (lane, sum) in sums.iter().enumerate() {
                        let row = i + 2 * rp + lane / 2;
                        let output = j + 2 * cp + lane % 2;
                        if row >= i + rows || output >= j + outputs {
                            continue;
                        }
                        let sum = match self.outliers {
                            Some(outliers) => add_outliers(
                                *sum,
                                outliers[row],
                                self.weight,
                                self.weight_scale,
                                self.n_in,
                                output,
                            ),
                            None => *sum,
                        };
                        *y.get_unchecked_mut(row * self.n_out + output) =
                            finish(sum, Some(self.bias[output]), self.with_gelu);
                    }
                }
            }
        }
    }

    // SAFETY: the slices cover `rows * n_in` activations, `n_out * n_in` weights,
    // `n_out` biases and `rows * n_out` outputs; activations are pair-interleaved.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "neon,i8mm")]
    pub(super) unsafe fn linear_rows_smmla(
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        group: usize,
        with_gelu: bool,
        y: &mut [f32],
        outliers: Option<&[Outliers]>,
    ) {
        if rows == 0 || n_out == 0 {
            return;
        }
        let prefill = Prefill {
            activations,
            activation_scale,
            weight,
            weight_scale,
            bias,
            n_in,
            n_out,
            group,
            with_gelu,
            outliers,
        };
        let mut block = 0;
        while block < n_out {
            let end = (block + PREFILL_OUTPUT_BLOCK).min(n_out);
            let mut i = 0;
            while i + PREFILL_ROWS <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    prefill.tile::<{ PREFILL_ROWS / 2 }, { PREFILL_OUTPUTS / 2 }>(
                        i,
                        j,
                        PREFILL_ROWS,
                        PREFILL_OUTPUTS,
                        y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j + 2 <= end {
                    prefill.tile::<{ PREFILL_ROWS / 2 }, 1>(i, j, PREFILL_ROWS, 2, y);
                    j += 2;
                }
                if j < end {
                    prefill.tile::<{ PREFILL_ROWS / 2 }, 1>(i, j, PREFILL_ROWS, 1, y);
                }
                i += PREFILL_ROWS;
            }
            while i + 4 <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    prefill.tile::<2, { PREFILL_OUTPUTS / 2 }>(
                        i,
                        j,
                        4,
                        PREFILL_OUTPUTS,
                        y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j + 2 <= end {
                    prefill.tile::<2, 1>(i, j, 4, 2, y);
                    j += 2;
                }
                if j < end {
                    prefill.tile::<2, 1>(i, j, 4, 1, y);
                }
                i += 4;
            }
            while i + 2 <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    prefill.tile::<1, { PREFILL_OUTPUTS / 2 }>(
                        i,
                        j,
                        2,
                        PREFILL_OUTPUTS,
                        y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j + 2 <= end {
                    prefill.tile::<1, 1>(i, j, 2, 2, y);
                    j += 2;
                }
                if j < end {
                    prefill.tile::<1, 1>(i, j, 2, 1, y);
                }
                i += 2;
            }
            while i < rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    prefill.tile::<1, { PREFILL_OUTPUTS / 2 }>(
                        i,
                        j,
                        1,
                        PREFILL_OUTPUTS,
                        y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j + 2 <= end {
                    prefill.tile::<1, 1>(i, j, 1, 2, y);
                    j += 2;
                }
                if j < end {
                    prefill.tile::<1, 1>(i, j, 1, 1, y);
                }
                i += 1;
            }
            block = end;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Int8ActivationKernels {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    DotProduct,
    #[cfg(target_arch = "aarch64")]
    Smmla,
}

impl Int8ActivationKernels {
    pub(crate) fn detect() -> Int8ActivationKernels {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod")
                && std::arch::is_aarch64_feature_detected!("i8mm")
            {
                return Int8ActivationKernels::Smmla;
            }
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return Int8ActivationKernels::DotProduct;
            }
        }
        Int8ActivationKernels::Scalar
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_outputs(
        self,
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: Option<&[f32]>,
        n_in: usize,
        group: usize,
        outputs: std::ops::Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        outliers: Option<Outliers>,
    ) {
        let weight_groups = n_in.div_ceil(crate::quantize::GROUP_SIZE);
        assert!(activations.len() >= n_in, "activation row is too short");
        assert!(activation_scale.len() >= n_in.div_ceil(group));
        assert!(weight.len() >= outputs.end * n_in, "weights are too short");
        assert!(weight_scale.len() >= outputs.end * weight_groups);
        assert!(y.len() >= outputs.len(), "output is too short");

        #[cfg(target_arch = "aarch64")]
        if group.is_multiple_of(crate::int8_layout::BLOCK) {
            match self {
                Int8ActivationKernels::DotProduct | Int8ActivationKernels::Smmla => {
                    // SAFETY: the lengths are checked above and detection confirmed the
                    // dot-product extension.
                    unsafe {
                        int8_activations_neon::linear_outputs(
                            activations,
                            activation_scale,
                            weight,
                            weight_scale,
                            bias,
                            n_in,
                            group,
                            outputs.start,
                            outputs.end,
                            matches!(epilogue, Epilogue::Gelu),
                            y,
                            outliers,
                        );
                    }
                    return;
                }
                Int8ActivationKernels::Scalar => {}
            }
        }
        int8_activations::linear_outputs(
            activations,
            activation_scale,
            weight,
            weight_scale,
            bias,
            n_in,
            group,
            outputs,
            y,
            epilogue,
            outliers,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_rows(
        self,
        activations: &[i8],
        activation_scale: &[f32],
        weight: &[i8],
        weight_scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        group: usize,
        y: &mut [f32],
        epilogue: Epilogue,
        outliers: Option<&[Outliers]>,
    ) {
        let weight_groups = n_in.div_ceil(crate::quantize::GROUP_SIZE);
        assert!(activations.len() >= rows * n_in);
        assert!(activation_scale.len() >= rows * n_in.div_ceil(group));
        assert!(weight.len() >= n_out * n_in, "weights are too short");
        assert!(weight_scale.len() >= n_out * weight_groups);
        assert!(bias.len() >= n_out);
        assert!(y.len() >= rows * n_out, "output is too short");
        if let Some(outliers) = outliers {
            assert_eq!(outliers.len(), rows, "one outlier list per row");
        }

        #[cfg(target_arch = "aarch64")]
        if group.is_multiple_of(crate::int8_layout::BLOCK) {
            let with_gelu = matches!(epilogue, Epilogue::Gelu);
            match self {
                Int8ActivationKernels::Smmla => {
                    // SAFETY: the lengths are checked above and detection confirmed
                    // the i8mm extension.
                    unsafe {
                        int8_activations_neon::linear_rows_smmla(
                            activations,
                            activation_scale,
                            weight,
                            weight_scale,
                            bias,
                            rows,
                            n_in,
                            n_out,
                            group,
                            with_gelu,
                            y,
                            outliers,
                        );
                    }
                    return;
                }
                Int8ActivationKernels::DotProduct => {
                    // SAFETY: the lengths are checked above and detection confirmed
                    // the dot-product extension.
                    unsafe {
                        int8_activations_neon::linear_rows_dotprod(
                            activations,
                            activation_scale,
                            weight,
                            weight_scale,
                            bias,
                            rows,
                            n_in,
                            n_out,
                            group,
                            with_gelu,
                            y,
                            outliers,
                        );
                    }
                    return;
                }
                Int8ActivationKernels::Scalar => {}
            }
        }
        int8_activations::linear_rows(
            activations,
            activation_scale,
            weight,
            weight_scale,
            bias,
            rows,
            n_in,
            n_out,
            group,
            y,
            epilogue,
            outliers,
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    use std::ops::Range;

    use super::{gelu_exact, Epilogue};
    use crate::int8_layout;

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
            gelu_exact(value)
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
        let weight_data = weight;
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
                let mut extra = [0.0f32; INT8_DECODE_ROWS];
                let mut p = start;
                while p < group_end && !p.is_multiple_of(int8_layout::BLOCK) {
                    for (r, extra_r) in extra.iter_mut().enumerate() {
                        *extra_r += *x.add(p)
                            * int8_layout::byte(weight_data, n_in, row + r, p) as f32;
                    }
                    p += 1;
                }
                while p + INT8_DECODE_ACCUMULATORS * 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                    for (a, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add(p + a * 4));
                    }
                    for (r, acc_r) in acc.iter_mut().enumerate() {
                        for h in 0..INT8_DECODE_ACCUMULATORS / 2 {
                            let w = widen8(
                                weight.add(int8_layout::block_offset(n_in, row + r, p + h * 8)),
                            );
                            acc_r[2 * h] = vfmaq_f32(acc_r[2 * h], w[0], xv[2 * h]);
                            acc_r[2 * h + 1] = vfmaq_f32(acc_r[2 * h + 1], w[1], xv[2 * h + 1]);
                        }
                    }
                    p += INT8_DECODE_ACCUMULATORS * 4;
                }
                while p < group_end {
                    for (r, extra_r) in extra.iter_mut().enumerate() {
                        *extra_r += *x.add(p)
                            * int8_layout::byte(weight_data, n_in, row + r, p) as f32;
                    }
                    p += 1;
                }
                for (r, acc_r) in acc.iter().enumerate() {
                    total[r] += (horizontal4(*acc_r) + extra[r]) * *scale.add((row + r) * groups + g);
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
            let mut total = 0.0;
            for g in 0..groups {
                let start = g * group;
                let group_end = (start + group).min(n_in);
                let mut acc = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                let mut extra = 0.0f32;
                let mut p = start;
                while p < group_end && !p.is_multiple_of(int8_layout::BLOCK) {
                    extra += int8_layout::byte(weight_data, n_in, row, p) as f32 * *x.add(p);
                    p += 1;
                }
                while p + INT8_DECODE_ACCUMULATORS * 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); INT8_DECODE_ACCUMULATORS];
                    for (a, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add(p + a * 4));
                    }
                    for h in 0..INT8_DECODE_ACCUMULATORS / 2 {
                        let w = widen8(weight.add(int8_layout::block_offset(n_in, row, p + h * 8)));
                        acc[2 * h] = vfmaq_f32(acc[2 * h], w[0], xv[2 * h]);
                        acc[2 * h + 1] = vfmaq_f32(acc[2 * h + 1], w[1], xv[2 * h + 1]);
                    }
                    p += INT8_DECODE_ACCUMULATORS * 4;
                }
                let mut sum = horizontal4(acc) + extra;
                while p < group_end {
                    sum += int8_layout::byte(weight_data, n_in, row, p) as f32 * *x.add(p);
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
            while i + 2 <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    f32_tile::<2, PREFILL_OUTPUTS>(
                        x, weight, bias, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    f32_tile::<2, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                    j += 1;
                }
                i += 2;
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
        let weight_data = weight;
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
            while p < group_end && !p.is_multiple_of(int8_layout::BLOCK) {
                for (r, extra_r) in extra.iter_mut().enumerate() {
                    for (c, extra_rc) in extra_r.iter_mut().enumerate() {
                        *extra_rc += *x.add((i + r) * n_in + p)
                            * int8_layout::byte(weight_data, n_in, j + c, p) as f32;
                    }
                }
                p += 1;
            }
            while p + 8 <= group_end {
                let mut xv = [[vdupq_n_f32(0.0); 2]; R];
                for (r, value) in xv.iter_mut().enumerate() {
                    *value = [
                        vld1q_f32(x.add((i + r) * n_in + p)),
                        vld1q_f32(x.add((i + r) * n_in + p + 4)),
                    ];
                }
                for c in 0..C {
                    let w = widen8(weight.add(int8_layout::block_offset(n_in, j + c, p)));
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
                        *extra_rc += *x.add((i + r) * n_in + p)
                            * int8_layout::byte(weight_data, n_in, j + c, p) as f32;
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
            while i + 2 <= rows {
                let mut j = block;
                while j + PREFILL_OUTPUTS <= end {
                    int8_tile::<2, PREFILL_OUTPUTS>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += PREFILL_OUTPUTS;
                }
                while j < end {
                    int8_tile::<2, 1>(
                        x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                    );
                    j += 1;
                }
                i += 2;
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
        while offset + 64 <= width {
            // SAFETY: `offset + 64 <= width <= stride`, so each row read is inside
            // `values`, and `output` has `width >= offset + 64` elements.
            unsafe {
                let mut acc = [vdupq_n_f32(0.0); 16];
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
            offset += 64;
        }
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

    use super::{gelu_exact, Epilogue};
    use crate::int8_layout;

    const DECODE_ACCUMULATORS: usize = 8;
    const PREFILL_TOKENS: usize = 4;
    const PREFILL_OUTPUTS: usize = 2;
    const INT8_PREFILL_TOKENS: usize = 2;
    const INT8_PREFILL_OUTPUTS: usize = 2;

    #[inline]
    fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
        let value = sum + bias.unwrap_or(0.0);
        if with_gelu {
            gelu_exact(value)
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
        while i + 2 <= rows {
            let mut j = 0;
            while j + PREFILL_OUTPUTS <= n_out {
                f32_tile::<2, PREFILL_OUTPUTS>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                j += PREFILL_OUTPUTS;
            }
            while j < n_out {
                f32_tile::<2, 1>(x, weight, bias, n_in, n_out, i, j, with_gelu, y);
                j += 1;
            }
            i += 2;
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
        let weight_data = weight;
        let weight = weight.as_ptr();
        let groups = n_in.div_ceil(group);
        let mut total = [[0.0f32; C]; R];
        for g in 0..groups {
            let start = g * group;
            let group_end = (start + group).min(n_in);
            let mut acc = [[f32x4_splat(0.0); C]; R];
            let mut extra = [[0.0f32; C]; R];
            let mut p = start;
            while p < group_end && !p.is_multiple_of(int8_layout::BLOCK) {
                for (r, extra_r) in extra.iter_mut().enumerate() {
                    for (c, extra_rc) in extra_r.iter_mut().enumerate() {
                        *extra_rc += *x.add((i + r) * n_in + p)
                            * int8_layout::byte(weight_data, n_in, j + c, p) as f32;
                    }
                }
                p += 1;
            }
            while p + 8 <= group_end {
                let mut xv = [[f32x4_splat(0.0); 2]; R];
                for (r, value) in xv.iter_mut().enumerate() {
                    *value = [
                        load(x.add((i + r) * n_in + p)),
                        load(x.add((i + r) * n_in + p + 4)),
                    ];
                }
                for c in 0..C {
                    let w = widen8(weight.add(int8_layout::block_offset(n_in, j + c, p)));
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
                        *extra_rc += *x.add((i + r) * n_in + p)
                            * int8_layout::byte(weight_data, n_in, j + c, p) as f32;
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
        while i + 2 <= rows {
            let mut j = 0;
            while j < n_out {
                int8_tile::<2, 1>(
                    x, weight, scale, bias, group, n_in, n_out, i, j, with_gelu, y,
                );
                j += 1;
            }
            i += 2;
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
            let mut sum = 0.0;
            for g in 0..groups {
                let start = g * group;
                let end = (start + group).min(n_in);
                let mut partial = 0.0;
                for p in start..end {
                    partial += x[p] * int8_layout::byte(weight, n_in, row, p) as f32;
                }
                sum += partial * scale[row * groups + g];
            }
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
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

    use super::{gelu_exact, Epilogue};
    use crate::int8_layout;

    #[inline]
    fn finish(sum: f32, bias: Option<f32>, with_gelu: bool) -> f32 {
        let value = sum + bias.unwrap_or(0.0);
        if with_gelu {
            gelu_exact(value)
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
                        partial += x[i * n_in + p] * int8_layout::byte(weight, n_in, j, p) as f32;
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
            let mut sum = 0.0;
            for g in 0..groups {
                let start = g * group;
                let end = (start + group).min(n_in);
                let mut partial = 0.0;
                for p in start..end {
                    partial += x[p] * int8_layout::byte(weight, n_in, row, p) as f32;
                }
                sum += partial * scale[row * groups + g];
            }
            y[offset] = finish(sum, bias.map(|bias| bias[offset]), with_gelu);
        }
    }

    #[cfg(test)]
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
        dot, int8_activations, int8_dot, int8_linear_outputs, int8_linear_rows, linear_outputs,
        linear_rows, weighted_sum, Epilogue, Int8ActivationKernels, Outliers,
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
    fn gelu_in_place_matches_the_libm_tanh_within_a_few_ulps() {
        let values: Vec<f32> = (-30000..30000)
            .map(|i| i as f32 / 1000.0)
            .chain([-1e6, -1e4, -100.0, -33.0, 33.0, 100.0, 1e4, 1e6])
            .collect();
        let mut actual = values.clone();
        super::gelu_in_place(&mut actual);
        for (input, got) in values.iter().zip(&actual) {
            let expected = super::gelu_exact(*input);
            let error = (got - expected).abs();
            let tolerance = 4.0 * f32::EPSILON * expected.abs().max(1.0);
            assert!(
                error <= tolerance,
                "gelu({input}): expected {expected}, got {got}"
            );
        }
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
        let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight.to_vec());
        if rows == 1 {
            int8_linear_outputs(
                x,
                &weight,
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
                x, &weight, scale, bias, rows, n_in, n_out, &mut y, epilogue, group,
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
                    super::gelu_exact(value)
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

    struct ActivationCase {
        rows: usize,
        n_in: usize,
        n_out: usize,
        group: usize,
    }

    struct ActivationFixture {
        activations: Vec<i8>,
        activation_scale: Vec<f32>,
        weight: Vec<i8>,
        weight_scale: Vec<f32>,
        bias: Vec<f32>,
    }

    fn activation_data(case: &ActivationCase) -> ActivationFixture {
        let ActivationCase {
            rows,
            n_in,
            n_out,
            group,
        } = *case;
        let activations = (0..rows * n_in)
            .map(|i| ((i * 37 + 11) % 255) as i32 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let activations = crate::int8_layout::interleave_owned(rows, n_in, activations);
        let activation_groups = n_in.div_ceil(group);
        let activation_scale = (0..rows * activation_groups)
            .map(|i| (i % 9 + 1) as f32 / 200.0)
            .collect();
        let weight = (0..n_out * n_in)
            .map(|i| ((i * 53 + 7) % 255) as i32 - 127)
            .map(|value| value as i8)
            .collect::<Vec<_>>();
        let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight);
        let weight_groups = n_in.div_ceil(crate::quantize::GROUP_SIZE);
        let weight_scale = (0..n_out * weight_groups)
            .map(|i| (i % 11 + 1) as f32 / 300.0)
            .collect();
        let bias = (0..n_out).map(|i| i as f32 / 17.0 - 1.0).collect();
        ActivationFixture {
            activations,
            activation_scale,
            weight,
            weight_scale,
            bias,
        }
    }

    fn plain_activation(fixture: &ActivationFixture, n_in: usize, row: usize) -> Vec<i8> {
        let mut plain = vec![0i8; n_in];
        crate::int8_layout::read_row(&fixture.activations, n_in, row, &mut plain);
        plain
    }

    #[test]
    fn int8_activation_kernels_match_the_scalar_order_bit_for_bit() {
        let kernels = Int8ActivationKernels::detect();
        for case in [
            ActivationCase {
                rows: 1,
                n_in: 1,
                n_out: 1,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 7,
                n_out: 3,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 15,
                n_out: 5,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 16,
                n_out: 4,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 17,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 31,
                n_out: 7,
                group: 32,
            },
            ActivationCase {
                rows: 1,
                n_in: 64,
                n_out: 8,
                group: 64,
            },
            ActivationCase {
                rows: 1,
                n_in: 65,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 2,
                n_in: 37,
                n_out: 5,
                group: 16,
            },
            ActivationCase {
                rows: 3,
                n_in: 64,
                n_out: 6,
                group: 32,
            },
            ActivationCase {
                rows: 4,
                n_in: 96,
                n_out: 7,
                group: 64,
            },
            ActivationCase {
                rows: 5,
                n_in: 130,
                n_out: 11,
                group: 16,
            },
            ActivationCase {
                rows: 7,
                n_in: 64,
                n_out: 17,
                group: 64,
            },
            ActivationCase {
                rows: 6,
                n_in: 33,
                n_out: 19,
                group: 32,
            },
            ActivationCase {
                rows: 3,
                n_in: 768,
                n_out: 9,
                group: 64,
            },
            ActivationCase {
                rows: 5,
                n_in: 768,
                n_out: 13,
                group: 16,
            },
        ] {
            let ActivationFixture {
                activations,
                activation_scale,
                weight,
                weight_scale,
                bias,
            } = activation_data(&case);
            let (rows, n_in, n_out, group) = (case.rows, case.n_in, case.n_out, case.group);

            for epilogue in [Epilogue::Store, Epilogue::Gelu] {
                if rows == 1 {
                    let plain = plain_activation(
                        &ActivationFixture {
                            activations: activations.clone(),
                            activation_scale: activation_scale.clone(),
                            weight: weight.clone(),
                            weight_scale: weight_scale.clone(),
                            bias: bias.clone(),
                        },
                        n_in,
                        0,
                    );
                    let mut expected = vec![0.0f32; n_out];
                    int8_activations::linear_outputs(
                        &plain,
                        &activation_scale,
                        &weight,
                        &weight_scale,
                        Some(&bias),
                        n_in,
                        group,
                        0..n_out,
                        &mut expected,
                        epilogue,
                        None,
                    );
                    let mut actual = vec![0.0f32; n_out];
                    kernels.linear_outputs(
                        &plain,
                        &activation_scale,
                        &weight,
                        &weight_scale,
                        Some(&bias),
                        n_in,
                        group,
                        0..n_out,
                        &mut actual,
                        epilogue,
                        None,
                    );
                    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            e.to_bits(),
                            "rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                        );
                    }
                }

                let mut expected = vec![0.0f32; rows * n_out];
                int8_activations::linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut expected,
                    epilogue,
                    None,
                );
                let mut actual = vec![0.0f32; rows * n_out];
                kernels.linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut actual,
                    epilogue,
                    None,
                );
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                    );
                }

                #[cfg(target_arch = "aarch64")]
                {
                    let with_gelu = matches!(epilogue, Epilogue::Gelu);
                    if std::arch::is_aarch64_feature_detected!("dotprod") {
                        let mut dotprod = vec![0.0f32; rows * n_out];
                        // SAFETY: the case sizes match the slices and the CPU has
                        // the dot-product extension.
                        unsafe {
                            super::int8_activations_neon::linear_rows_dotprod(
                                &activations,
                                &activation_scale,
                                &weight,
                                &weight_scale,
                                &bias,
                                rows,
                                n_in,
                                n_out,
                                group,
                                with_gelu,
                                &mut dotprod,
                                None,
                            );
                        }
                        for (index, (a, e)) in dotprod.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                e.to_bits(),
                                "dotprod rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn smmla_dotprod_and_scalar_prefill_agree_bit_for_bit() {
        if !(std::arch::is_aarch64_feature_detected!("dotprod")
            && std::arch::is_aarch64_feature_detected!("i8mm"))
        {
            return;
        }
        let kernels = Int8ActivationKernels::detect();
        for case in [
            ActivationCase {
                rows: 2,
                n_in: 37,
                n_out: 5,
                group: 16,
            },
            ActivationCase {
                rows: 3,
                n_in: 64,
                n_out: 7,
                group: 32,
            },
            ActivationCase {
                rows: 5,
                n_in: 65,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 8,
                n_in: 130,
                n_out: 13,
                group: 64,
            },
            ActivationCase {
                rows: 9,
                n_in: 768,
                n_out: 17,
                group: 16,
            },
            ActivationCase {
                rows: 7,
                n_in: 3072,
                n_out: 11,
                group: 32,
            },
        ] {
            let ActivationFixture {
                activations,
                activation_scale,
                weight,
                weight_scale,
                bias,
            } = activation_data(&case);
            let (rows, n_in, n_out, group) = (case.rows, case.n_in, case.n_out, case.group);
            let per_row: Vec<(Vec<u32>, Vec<f32>)> = (0..rows)
                .map(|row| {
                    let indices = (0..3)
                        .map(|i| ((i * n_in / 3 + row * 5) % n_in) as u32)
                        .collect();
                    let values = (0..3).map(|i| (i + row) as f32 - 1.5).collect();
                    (indices, values)
                })
                .collect();
            let row_outliers: Vec<Outliers> = per_row
                .iter()
                .map(|(indices, values)| Outliers { indices, values })
                .collect();

            for epilogue in [Epilogue::Store, Epilogue::Gelu] {
                let with_gelu = matches!(epilogue, Epilogue::Gelu);
                let mut expected = vec![0.0f32; rows * n_out];
                int8_activations::linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut expected,
                    epilogue,
                    Some(&row_outliers),
                );

                let mut smmla = vec![0.0f32; rows * n_out];
                // SAFETY: the fixture sizes match the slices and the CPU has i8mm.
                unsafe {
                    super::int8_activations_neon::linear_rows_smmla(
                        &activations,
                        &activation_scale,
                        &weight,
                        &weight_scale,
                        &bias,
                        rows,
                        n_in,
                        n_out,
                        group,
                        with_gelu,
                        &mut smmla,
                        Some(&row_outliers),
                    );
                }
                for (index, (a, e)) in smmla.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "smmla rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                    );
                }

                let mut dotprod = vec![0.0f32; rows * n_out];
                // SAFETY: the fixture sizes match the slices and the CPU has dotprod.
                unsafe {
                    super::int8_activations_neon::linear_rows_dotprod(
                        &activations,
                        &activation_scale,
                        &weight,
                        &weight_scale,
                        &bias,
                        rows,
                        n_in,
                        n_out,
                        group,
                        with_gelu,
                        &mut dotprod,
                        Some(&row_outliers),
                    );
                }
                for (index, (a, e)) in dotprod.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "dotprod rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                    );
                }

                let mut dispatch = vec![0.0f32; rows * n_out];
                kernels.linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut dispatch,
                    epilogue,
                    Some(&row_outliers),
                );
                assert_eq!(dispatch, expected);
            }
        }
    }

    fn outliers(n_in: usize, count: usize) -> (Vec<u32>, Vec<f32>) {
        assert!(count <= n_in);
        let indices = (0..count)
            .map(|i| (i * n_in / count.max(1)) as u32)
            .collect();
        let values = (0..count).map(|i| i as f32 * 1.5 - 2.0).collect();
        (indices, values)
    }

    #[test]
    fn int8_activation_outliers_match_the_scalar_order_bit_for_bit() {
        let kernels = Int8ActivationKernels::detect();
        for case in [
            ActivationCase {
                rows: 1,
                n_in: 1,
                n_out: 1,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 17,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 33,
                n_out: 7,
                group: 32,
            },
            ActivationCase {
                rows: 1,
                n_in: 64,
                n_out: 13,
                group: 64,
            },
            ActivationCase {
                rows: 1,
                n_in: 65,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 1,
                n_in: 130,
                n_out: 11,
                group: 32,
            },
            ActivationCase {
                rows: 1,
                n_in: 769,
                n_out: 17,
                group: 64,
            },
        ] {
            let ActivationFixture {
                activations,
                activation_scale,
                weight,
                weight_scale,
                bias,
            } = activation_data(&case);
            let (n_in, n_out, group) = (case.n_in, case.n_out, case.group);
            let plain = plain_activation(
                &ActivationFixture {
                    activations: activations.clone(),
                    activation_scale: activation_scale.clone(),
                    weight: weight.clone(),
                    weight_scale: weight_scale.clone(),
                    bias: bias.clone(),
                },
                n_in,
                0,
            );
            for count in [0usize, 4, 8] {
                if count > 0 && n_in < 8 {
                    continue;
                }
                let (indices, values) = outliers(n_in, count);
                let outliers = Some(Outliers {
                    indices: &indices,
                    values: &values,
                });
                let mut expected = vec![0.0f32; n_out];
                int8_activations::linear_outputs(
                    &plain,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    Some(&bias),
                    n_in,
                    group,
                    0..n_out,
                    &mut expected,
                    Epilogue::Store,
                    outliers,
                );
                let mut actual = vec![0.0f32; n_out];
                kernels.linear_outputs(
                    &plain,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    Some(&bias),
                    n_in,
                    group,
                    0..n_out,
                    &mut actual,
                    Epilogue::Store,
                    outliers,
                );
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "in {n_in}, out {n_out}, group {group}, outliers {count}, element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn int8_activation_row_outliers_match_the_scalar_order_bit_for_bit() {
        let kernels = Int8ActivationKernels::detect();
        for case in [
            ActivationCase {
                rows: 1,
                n_in: 65,
                n_out: 9,
                group: 16,
            },
            ActivationCase {
                rows: 2,
                n_in: 37,
                n_out: 5,
                group: 16,
            },
            ActivationCase {
                rows: 3,
                n_in: 64,
                n_out: 6,
                group: 32,
            },
            ActivationCase {
                rows: 5,
                n_in: 130,
                n_out: 11,
                group: 16,
            },
            ActivationCase {
                rows: 7,
                n_in: 64,
                n_out: 17,
                group: 64,
            },
        ] {
            let ActivationFixture {
                activations,
                activation_scale,
                weight,
                weight_scale,
                bias,
            } = activation_data(&case);
            let (rows, n_in, n_out, group) = (case.rows, case.n_in, case.n_out, case.group);
            let per_row: Vec<(Vec<u32>, Vec<f32>)> = (0..rows)
                .map(|row| {
                    let indices = (0..4)
                        .map(|i| ((i * n_in / 4 + row * 7) % n_in) as u32)
                        .collect();
                    let values = (0..4).map(|i| (i + row) as f32 * 1.5 - 2.0).collect();
                    (indices, values)
                })
                .collect();
            let row_outliers: Vec<Outliers> = per_row
                .iter()
                .map(|(indices, values)| Outliers { indices, values })
                .collect();

            for epilogue in [Epilogue::Store, Epilogue::Gelu] {
                let mut expected = vec![0.0f32; rows * n_out];
                int8_activations::linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut expected,
                    epilogue,
                    Some(&row_outliers),
                );
                let mut actual = vec![0.0f32; rows * n_out];
                kernels.linear_rows(
                    &activations,
                    &activation_scale,
                    &weight,
                    &weight_scale,
                    &bias,
                    rows,
                    n_in,
                    n_out,
                    group,
                    &mut actual,
                    epilogue,
                    Some(&row_outliers),
                );
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "rows {rows}, in {n_in}, out {n_out}, group {group}, element {index}"
                    );
                }

                if rows == 1 {
                    let plain = plain_activation(
                        &ActivationFixture {
                            activations: activations.clone(),
                            activation_scale: activation_scale.clone(),
                            weight: weight.clone(),
                            weight_scale: weight_scale.clone(),
                            bias: bias.clone(),
                        },
                        n_in,
                        0,
                    );
                    let mut one = vec![0.0f32; n_out];
                    kernels.linear_outputs(
                        &plain,
                        &activation_scale,
                        &weight,
                        &weight_scale,
                        Some(&bias),
                        n_in,
                        group,
                        0..n_out,
                        &mut one,
                        epilogue,
                        Some(row_outliers[0]),
                    );
                    for (index, (a, e)) in one.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            e.to_bits(),
                            "single row, in {n_in}, out {n_out}, group {group}, element {index}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn outlier_scheme_matches_the_plain_scheme_when_k_is_zero() {
        use crate::quantize::{quantize_activation_outliers, quantize_activations};

        let (n_in, n_out, group) = (65usize, 9usize, 32usize);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 7.0 - 5.0)
            .collect();
        let weight: Vec<i8> = (0..n_out * n_in)
            .map(|i| ((i * 53 + 7) % 255) as i32 - 127)
            .map(|value| value as i8)
            .collect();
        let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight);
        let weight_scale: Vec<f32> = (0..n_out * n_in.div_ceil(64))
            .map(|i| (i % 11 + 1) as f32 / 300.0)
            .collect();
        let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 17.0 - 1.0).collect();
        let groups = n_in.div_ceil(group);

        let mut plain_data = vec![0i8; n_in];
        let mut plain_scale = vec![0.0f32; groups];
        quantize_activations(&x, 1, n_in, group, &mut plain_data, &mut plain_scale);
        let mut expected = vec![0.0f32; n_out];
        Int8ActivationKernels::detect().linear_outputs(
            &plain_data,
            &plain_scale,
            &weight,
            &weight_scale,
            Some(&bias),
            n_in,
            group,
            0..n_out,
            &mut expected,
            Epilogue::Store,
            None,
        );

        let mut data = vec![1i8; n_in];
        let mut scale = vec![1.0f32; groups];
        let mut indices = vec![0u32; n_in];
        let mut removed = vec![0.0f32; n_in];
        let count = quantize_activation_outliers(
            &x,
            group,
            0,
            &mut data,
            &mut scale,
            &mut indices,
            &mut removed,
        );
        assert_eq!(count, 0);
        let mut actual = vec![0.0f32; n_out];
        Int8ActivationKernels::detect().linear_outputs(
            &data,
            &scale,
            &weight,
            &weight_scale,
            Some(&bias),
            n_in,
            group,
            0..n_out,
            &mut actual,
            Epilogue::Store,
            Some(Outliers {
                indices: &indices[..count],
                values: &removed[..count],
            }),
        );
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "element {index}");
        }
    }

    #[test]
    fn outlier_correction_adds_the_removed_values_back_exactly() {
        use crate::quantize::quantize_activation_outliers;

        let (n_in, n_out, group, keep) = (130usize, 7usize, 32usize, 4usize);
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 7.0 - 5.0)
            .collect();
        let mut x = x;
        x[1] = 300.0;
        x[35] = -400.0;
        let weight: Vec<i8> = (0..n_out * n_in)
            .map(|i| ((i * 53 + 7) % 255) as i32 - 127)
            .map(|value| value as i8)
            .collect();
        let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight);
        let weight_scale: Vec<f32> = (0..n_out * n_in.div_ceil(64))
            .map(|i| (i % 11 + 1) as f32 / 300.0)
            .collect();
        let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 17.0 - 1.0).collect();
        let groups = n_in.div_ceil(group);
        let weight_groups = n_in.div_ceil(64);

        let mut data = vec![0i8; n_in];
        let mut scale = vec![0.0f32; groups];
        let mut indices = vec![0u32; n_in];
        let mut removed = vec![0.0f32; n_in];
        let count = quantize_activation_outliers(
            &x,
            group,
            keep,
            &mut data,
            &mut scale,
            &mut indices,
            &mut removed,
        );
        assert_eq!(count, keep);

        let mut actual = vec![0.0f32; n_out];
        Int8ActivationKernels::detect().linear_outputs(
            &data,
            &scale,
            &weight,
            &weight_scale,
            Some(&bias),
            n_in,
            group,
            0..n_out,
            &mut actual,
            Epilogue::Store,
            Some(Outliers {
                indices: &indices[..count],
                values: &removed[..count],
            }),
        );

        for output in 0..n_out {
            let mut expected = 0.0f64;
            for weight_group in 0..weight_groups {
                let start = weight_group * 64;
                let end = (start + 64).min(n_in);
                let first = start / group;
                let last = (end - 1) / group;
                for (activation_group, &activation_scale) in
                    scale.iter().enumerate().take(last + 1).skip(first)
                {
                    let start = activation_group * group;
                    let end = (start + group).min(n_in);
                    let dot: i64 = (start..end)
                        .map(|p| {
                            data[p] as i64
                                * crate::int8_layout::byte(&weight, n_in, output, p) as i64
                        })
                        .sum();
                    expected += dot as f64
                        * (weight_scale[output * weight_groups + weight_group] as f64
                            * activation_scale as f64);
                }
            }
            for (&index, &value) in indices[..count].iter().zip(&removed[..count]) {
                let index = index as usize;
                expected += value as f64
                    * (weight_scale[output * weight_groups + index / 64] as f64
                        * crate::int8_layout::byte(&weight, n_in, output, index) as f64);
            }
            expected += bias[output] as f64;
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual[output] as f64 - expected).abs() <= tolerance,
                "output {output}: expected {expected}, got {}",
                actual[output]
            );
        }
    }

    #[test]
    #[ignore]
    fn int8_activation_kernel_weight_bandwidth() {
        use std::time::{Duration, Instant};

        use crate::parallel::Threads;

        let kernels = Int8ActivationKernels::detect();
        let measure = |n_in: usize, n_out: usize, rows: usize, count: usize, new: bool| {
            let threads = Threads::with_count(count).expect("thread pool");
            let groups = n_in.div_ceil(64);
            let activations: Vec<f32> = (0..rows * n_in)
                .map(|i| ((i * 29 + 3) % 71) as f32 / 71.0 - 0.5)
                .collect();
            let quantized: Vec<i8> = activations
                .iter()
                .map(|value| (value * 127.0) as i8)
                .collect();
            let quantized = crate::int8_layout::interleave_owned(rows, n_in, quantized);
            let activation_scale = vec![0.01f32; rows * groups];
            let weight: Vec<i8> = (0..n_out * n_in)
                .map(|i| ((i * 43 + 5) % 255) as i32 - 127)
                .map(|value| value as i8)
                .collect();
            let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight);
            let weight_scale = vec![0.01f32; n_out * groups];
            let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 1000.0).collect();
            let mut y = vec![0.0f32; rows * n_out];

            let mut run = || {
                if rows == 1 {
                    threads.run(&mut y, 1, |start, part| {
                        if new {
                            kernels.linear_outputs(
                                &quantized,
                                &activation_scale,
                                &weight,
                                &weight_scale,
                                Some(&bias[start..start + part.len()]),
                                n_in,
                                64,
                                start..start + part.len(),
                                part,
                                Epilogue::Store,
                                None,
                            );
                        } else {
                            int8_linear_outputs(
                                &activations,
                                &weight,
                                &weight_scale,
                                Some(&bias[start..start + part.len()]),
                                n_in,
                                start..start + part.len(),
                                part,
                                Epilogue::Store,
                                64,
                            );
                        }
                    });
                } else {
                    threads.run(&mut y, 2 * n_out, |start, part| {
                        let first = start / n_out;
                        let count = part.len() / n_out;
                        let stride = crate::int8_layout::pair_stride(n_in);
                        let scales = &activation_scale[first * groups..(first + count) * groups];
                        if new {
                            kernels.linear_rows(
                                &quantized[(first / 2) * stride..],
                                scales,
                                &weight,
                                &weight_scale,
                                &bias,
                                count,
                                n_in,
                                n_out,
                                64,
                                part,
                                Epilogue::Store,
                                None,
                            );
                        } else {
                            int8_linear_rows(
                                &activations[first * n_in..(first + count) * n_in],
                                &weight,
                                &weight_scale,
                                &bias,
                                count,
                                n_in,
                                n_out,
                                part,
                                Epilogue::Store,
                                64,
                            );
                        }
                    });
                }
            };

            run();
            let iterations = (2_000_000_000 / (n_out * n_in)).max(5);
            let mut times = Vec::new();
            for _ in 0..5 {
                let start = Instant::now();
                for _ in 0..iterations {
                    run();
                }
                times.push(start.elapsed());
            }
            times.sort();
            let elapsed: Duration = times[2];
            n_out as f64 * n_in as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9
        };

        for (n_in, n_out) in [(768usize, 3072usize), (3072, 768), (768, 50257)] {
            for rows in [1usize, 16] {
                for threads in [1usize, 4] {
                    let old = measure(n_in, n_out, rows, threads, false);
                    let new = measure(n_in, n_out, rows, threads, true);
                    println!(
                        "{n_in}->{n_out} rows {rows} threads {threads}: old {old:.1} GB/s, new {new:.1} GB/s"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod batch_identity {
    use super::{Epilogue, Int8ActivationKernels, Outliers};

    #[test]
    fn rows_match_single_row_outputs_at_model_sizes() {
        let kernels = Int8ActivationKernels::detect();
        for (rows, n_in, n_out, group) in [
            (5usize, 768usize, 768usize, 64usize),
            (5, 768, 2304, 64),
            (5, 768, 3072, 64),
            (5, 3072, 768, 64),
            (5, 768, 1000, 64),
            (5, 768, 50257, 64),
        ] {
            let activations: Vec<i8> = (0..rows * n_in)
                .map(|i| ((i * 37 + 11) % 255) as i32 - 127)
                .map(|v| v as i8)
                .collect();
            let activations = crate::int8_layout::interleave_owned(rows, n_in, activations);
            let activation_groups = n_in.div_ceil(group);
            let activation_scale: Vec<f32> = (0..rows * activation_groups)
                .map(|i| (i % 9 + 1) as f32 / 200.0)
                .collect();
            let weight: Vec<i8> = (0..n_out * n_in)
                .map(|i| ((i * 53 + 7) % 255) as i32 - 127)
                .map(|v| v as i8)
                .collect();
            let weight = crate::int8_layout::interleave_owned(n_out, n_in, weight);
            let weight_groups = n_in.div_ceil(64);
            let weight_scale: Vec<f32> = (0..n_out * weight_groups)
                .map(|i| (i % 11 + 1) as f32 / 300.0)
                .collect();
            let bias: Vec<f32> = (0..n_out).map(|i| i as f32 / 17.0 - 1.0).collect();
            let per_row: Vec<(Vec<u32>, Vec<f32>)> = (0..rows)
                .map(|row| {
                    let indices = (0..4)
                        .map(|i| ((i * n_in / 4 + row * 7) % n_in) as u32)
                        .collect();
                    let values = (0..4).map(|i| (i + row) as f32 * 1.5 - 2.0).collect();
                    (indices, values)
                })
                .collect();
            let row_outliers: Vec<Outliers> = per_row
                .iter()
                .map(|(indices, values)| Outliers { indices, values })
                .collect();

            let mut actual = vec![0.0f32; rows * n_out];
            kernels.linear_rows(
                &activations,
                &activation_scale,
                &weight,
                &weight_scale,
                &bias,
                rows,
                n_in,
                n_out,
                group,
                &mut actual,
                Epilogue::Store,
                Some(&row_outliers),
            );
            for row in 0..rows {
                let mut expected = vec![0.0f32; n_out];
                let mut plain = vec![0i8; n_in];
                crate::int8_layout::read_row(&activations, n_in, row, &mut plain);
                kernels.linear_outputs(
                    &plain,
                    &activation_scale[row * activation_groups..(row + 1) * activation_groups],
                    &weight,
                    &weight_scale,
                    Some(&bias),
                    n_in,
                    group,
                    0..n_out,
                    &mut expected,
                    Epilogue::Store,
                    Some(row_outliers[row]),
                );
                for (index, (a, e)) in actual[row * n_out..(row + 1) * n_out]
                    .iter()
                    .zip(&expected)
                    .enumerate()
                {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "rows {rows}, in {n_in}, out {n_out}, group {group}, row {row}, element {index}"
                    );
                }
            }
        }
    }
}
