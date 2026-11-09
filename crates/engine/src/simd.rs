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

    struct Linear<'a> {
        x: &'a [f32],
        weight: &'a [f32],
        bias: &'a [f32],
        n_in: usize,
        n_out: usize,
        first_column: usize,
        columns: usize,
    }

    impl Linear<'_> {
        // SAFETY: `i + R <= rows`, `j + V * 4 <= first_column + columns`, `y` has
        // `rows * columns` elements, and all slices are sized for `rows * n_in`,
        // `n_in * n_out` and `rows * n_out` elements.
        #[allow(clippy::needless_range_loop)]
        unsafe fn block<const R: usize, const V: usize, const GELU: bool>(
            &self,
            i: usize,
            j: usize,
            y: &mut [f32],
        ) {
            let x = self.x.as_ptr();
            let weight = self.weight.as_ptr();
            let bias = self.bias.as_ptr();
            let n_in = self.n_in;
            let n_out = self.n_out;
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
                    for r in 0..R {
                        acc[r][c] = vfmaq_laneq_f32::<0>(acc[r][c], w0, xv[r]);
                        acc[r][c] = vfmaq_laneq_f32::<1>(acc[r][c], w1, xv[r]);
                        acc[r][c] = vfmaq_laneq_f32::<2>(acc[r][c], w2, xv[r]);
                        acc[r][c] = vfmaq_laneq_f32::<3>(acc[r][c], w3, xv[r]);
                    }
                }
                p += 4;
            }
            while p < n_in {
                for r in 0..R {
                    let a = vdupq_n_f32(*x.add((i + r) * n_in + p));
                    for c in 0..V {
                        let w = vld1q_f32(weight.add(p * n_out + j + c * 4));
                        acc[r][c] = vfmaq_f32(acc[r][c], a, w);
                    }
                }
                p += 1;
            }

            for r in 0..R {
                let row = (i + r) * self.columns + j - self.first_column;
                let out = y.as_mut_ptr().add(row);
                for (c, accumulated) in acc[r].iter().enumerate() {
                    let value = vaddq_f32(*accumulated, vld1q_f32(bias.add(j + c * 4)));
                    let out = out.add(c * 4);
                    if GELU {
                        let mut lanes = [0.0; 4];
                        vst1q_f32(lanes.as_mut_ptr(), value);
                        vst1q_f32(out, vld1q_f32(lanes.map(gelu).as_ptr()));
                    } else {
                        vst1q_f32(out, value);
                    }
                }
            }
        }

        fn block_rows<const R: usize, const V: usize, const GELU: bool>(
            &self,
            i: usize,
            y: &mut [f32],
        ) {
            let end = self.first_column + self.columns;
            let mut j = self.first_column;
            while j + V * 4 <= end {
                // SAFETY: the loop condition guarantees `i + R <= rows`.
                unsafe { self.block::<R, V, GELU>(i, j, y) };
                j += V * 4;
            }
            if j < end {
                self.edge_columns::<R, GELU>(i, j, y);
            }
        }

        fn edge_columns<const R: usize, const GELU: bool>(
            &self,
            i: usize,
            j: usize,
            y: &mut [f32],
        ) {
            for r in 0..R {
                for column in j..self.first_column + self.columns {
                    let mut sum = 0.0;
                    for p in 0..self.n_in {
                        sum +=
                            self.x[(i + r) * self.n_in + p] * self.weight[p * self.n_out + column];
                    }
                    let value = sum + self.bias[column];
                    y[(i + r) * self.columns + column - self.first_column] =
                        if GELU { gelu(value) } else { value };
                }
            }
        }
    }

    fn tiled<const R: usize, const V: usize, const GELU: bool>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
        let kernel = Linear {
            x,
            weight,
            bias,
            n_in,
            n_out,
            first_column: 0,
            columns: n_out,
        };
        let mut i = 0;
        while i < rows {
            let remaining = rows - i;
            if remaining >= R {
                kernel.block_rows::<R, V, GELU>(i, y);
                i += R;
            } else if remaining >= 4 {
                kernel.block_rows::<4, V, GELU>(i, y);
                i += 4;
            } else if remaining >= 2 {
                kernel.block_rows::<2, V, GELU>(i, y);
                i += 2;
            } else {
                kernel.block_rows::<1, V, GELU>(i, y);
                i += 1;
            }
        }
    }

    // Row slices of a multi-row matrix must stay on the 16-column kernel; the
    // rows == 1 kernel widens to 32 columns and rounds differently on the edge.
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
        match epilogue {
            Epilogue::Store => tiled::<4, 4, false>(x, weight, bias, rows, n_in, n_out, y),
            Epilogue::Gelu => tiled::<4, 4, true>(x, weight, bias, rows, n_in, n_out, y),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_columns(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        assert!(
            columns.start.is_multiple_of(32),
            "column split is not aligned to the vector block"
        );
        let kernel = Linear {
            x,
            weight,
            bias,
            n_in,
            n_out,
            first_column: columns.start,
            columns: columns.len(),
        };
        match epilogue {
            Epilogue::Store => kernel.block_rows::<1, 8, false>(0, y),
            Epilogue::Gelu => kernel.block_rows::<1, 8, true>(0, y),
        }
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

    struct Int8Linear<'a> {
        x: &'a [f32],
        weight: &'a [i8],
        scale: &'a [f32],
        bias: &'a [f32],
        n_in: usize,
        n_out: usize,
        first_column: usize,
        columns: usize,
        group: usize,
    }

    impl Int8Linear<'_> {
        // SAFETY: `i + R <= rows`, `j + V * 8 <= first_column + columns`, `y`
        // has `rows * columns` elements, and all slices are sized for
        // `rows * n_in`, `n_in * n_out` and `rows * n_out` elements.
        #[allow(clippy::needless_range_loop)]
        unsafe fn block<const R: usize, const V: usize, const GELU: bool>(
            &self,
            i: usize,
            j: usize,
            y: &mut [f32],
        ) {
            let x = self.x.as_ptr();
            let weight = self.weight.as_ptr();
            let scale = self.scale.as_ptr();
            let bias = self.bias.as_ptr();
            let n_in = self.n_in;
            let n_out = self.n_out;
            let group = self.group;
            let mut total = [[[vdupq_n_f32(0.0); 2]; R]; V];

            let mut p = 0;
            while p < n_in {
                let group_end = (p / group * group + group).min(n_in);
                let scale_row = p / group * n_out;
                let mut acc = [[[vdupq_n_f32(0.0); 2]; R]; V];
                while p + 4 <= group_end {
                    let mut xv = [vdupq_n_f32(0.0); R];
                    for (r, value) in xv.iter_mut().enumerate() {
                        *value = vld1q_f32(x.add((i + r) * n_in + p));
                    }
                    for c in 0..V {
                        let column = j + c * 8;
                        let w0 = widen8(weight.add(p * n_out + column));
                        let w1 = widen8(weight.add((p + 1) * n_out + column));
                        let w2 = widen8(weight.add((p + 2) * n_out + column));
                        let w3 = widen8(weight.add((p + 3) * n_out + column));
                        for r in 0..R {
                            for k in 0..2 {
                                acc[c][r][k] = vfmaq_laneq_f32::<0>(acc[c][r][k], w0[k], xv[r]);
                                acc[c][r][k] = vfmaq_laneq_f32::<1>(acc[c][r][k], w1[k], xv[r]);
                                acc[c][r][k] = vfmaq_laneq_f32::<2>(acc[c][r][k], w2[k], xv[r]);
                                acc[c][r][k] = vfmaq_laneq_f32::<3>(acc[c][r][k], w3[k], xv[r]);
                            }
                        }
                    }
                    p += 4;
                }
                while p < group_end {
                    for r in 0..R {
                        let a = *x.add((i + r) * n_in + p);
                        for c in 0..V {
                            let w = widen8(weight.add(p * n_out + j + c * 8));
                            for k in 0..2 {
                                acc[c][r][k] = vfmaq_n_f32(acc[c][r][k], w[k], a);
                            }
                        }
                    }
                    p += 1;
                }

                for c in 0..V {
                    for r in 0..R {
                        for k in 0..2 {
                            let column = scale_row + j + c * 8 + k * 4;
                            total[c][r][k] = vfmaq_f32(
                                total[c][r][k],
                                acc[c][r][k],
                                vld1q_f32(scale.add(column)),
                            );
                        }
                    }
                }
            }

            for c in 0..V {
                for r in 0..R {
                    let row = (i + r) * self.columns + j + c * 8 - self.first_column;
                    let out = y.as_mut_ptr().add(row);
                    for k in 0..2 {
                        let column = j + c * 8 + k * 4;
                        let value = vaddq_f32(total[c][r][k], vld1q_f32(bias.add(column)));
                        let out = out.add(k * 4);
                        if GELU {
                            let mut lanes = [0.0; 4];
                            vst1q_f32(lanes.as_mut_ptr(), value);
                            vst1q_f32(out, vld1q_f32(lanes.map(gelu).as_ptr()));
                        } else {
                            vst1q_f32(out, value);
                        }
                    }
                }
            }
        }

        fn block_rows<const R: usize, const V: usize, const GELU: bool>(
            &self,
            i: usize,
            y: &mut [f32],
        ) {
            let end = self.first_column + self.columns;
            let mut j = self.first_column;
            while j + V * 8 <= end {
                // SAFETY: the loop condition guarantees `i + R <= rows`.
                unsafe { self.block::<R, V, GELU>(i, j, y) };
                j += V * 8;
            }
            if j < end {
                self.edge_columns::<R, GELU>(i, j, y);
            }
        }

        fn edge_columns<const R: usize, const GELU: bool>(
            &self,
            i: usize,
            j: usize,
            y: &mut [f32],
        ) {
            for r in 0..R {
                for column in j..self.first_column + self.columns {
                    let mut sum = 0.0;
                    let mut p = 0;
                    while p < self.n_in {
                        let end = (p / self.group * self.group + self.group).min(self.n_in);
                        let mut partial = 0.0;
                        for q in p..end {
                            partial += self.x[(i + r) * self.n_in + q]
                                * self.weight[q * self.n_out + column] as f32;
                        }
                        sum += partial * self.scale[p / self.group * self.n_out + column];
                        p = end;
                    }
                    let value = sum + self.bias[column];
                    y[(i + r) * self.columns + column - self.first_column] =
                        if GELU { gelu(value) } else { value };
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_tiled<const R: usize, const V: usize, const GELU: bool>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        group: usize,
    ) {
        let kernel = Int8Linear {
            x,
            weight,
            scale,
            bias,
            n_in,
            n_out,
            first_column: 0,
            columns: n_out,
            group,
        };
        let mut i = 0;
        while i < rows {
            let remaining = rows - i;
            if remaining >= R {
                kernel.block_rows::<R, V, GELU>(i, y);
                i += R;
            } else if remaining >= 4 {
                kernel.block_rows::<4, V, GELU>(i, y);
                i += 4;
            } else if remaining >= 2 {
                kernel.block_rows::<2, V, GELU>(i, y);
                i += 2;
            } else {
                kernel.block_rows::<1, V, GELU>(i, y);
                i += 1;
            }
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
        match epilogue {
            Epilogue::Store => {
                int8_tiled::<4, 1, false>(x, weight, scale, bias, rows, n_in, n_out, y, group)
            }
            Epilogue::Gelu => {
                int8_tiled::<4, 1, true>(x, weight, scale, bias, rows, n_in, n_out, y, group)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_columns(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        assert!(
            columns.start.is_multiple_of(32),
            "column split is not aligned to the vector block"
        );
        let kernel = Int8Linear {
            x,
            weight,
            scale,
            bias,
            n_in,
            n_out,
            first_column: columns.start,
            columns: columns.len(),
            group,
        };
        match epilogue {
            Epilogue::Store => kernel.block_rows::<1, 2, false>(0, y),
            Epilogue::Gelu => kernel.block_rows::<1, 2, true>(0, y),
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
                let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
                let mut sum = vaddvq_f32(pair(
                    pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
                    pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
                ));
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
            let pair = |left: float32x4_t, right: float32x4_t| vaddq_f32(left, right);
            let mut sum = vaddvq_f32(pair(
                pair(pair(acc[0], acc[1]), pair(acc[2], acc[3])),
                pair(pair(acc[4], acc[5]), pair(acc[6], acc[7])),
            ));
            while i < n {
                sum += *a.add(i) * *b.add(i);
                i += 1;
            }
            sum
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{dot, linear_columns, linear_rows, weighted_sum, Epilogue};

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
                linear_columns(
                    x,
                    weight,
                    bias,
                    n_in,
                    n_out,
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
                        sum += x[i * n_in + p] * weight[p * n_out + j];
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
    }
}

#[cfg(not(target_arch = "aarch64"))]
mod scalar {
    use std::ops::Range;

    use super::{gelu, Epilogue};

    const ROW_TILE: usize = 8;

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
        match epilogue {
            Epilogue::Store => self::rows::<false>(x, weight, bias, rows, n_in, n_out, y),
            Epilogue::Gelu => self::rows::<true>(x, weight, bias, rows, n_in, n_out, y),
        }
    }

    fn rows<const GELU: bool>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
        y.fill(0.0);
        for i in (0..rows).step_by(ROW_TILE) {
            let block = (rows - i).min(ROW_TILE);
            let y_block = &mut y[i * n_out..(i + block) * n_out];
            for p in 0..n_in {
                let w_row = &weight[p * n_out..(p + 1) * n_out];
                for r in 0..block {
                    let a = x[(i + r) * n_in + p];
                    let y_row = &mut y_block[r * n_out..(r + 1) * n_out];
                    for (sum, &w) in y_row.iter_mut().zip(w_row) {
                        *sum += a * w;
                    }
                }
            }
            for r in 0..block {
                let y_row = &mut y_block[r * n_out..(r + 1) * n_out];
                for (value, &b) in y_row.iter_mut().zip(bias) {
                    let sum = *value + b;
                    *value = if GELU { gelu(sum) } else { sum };
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_columns(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
    ) {
        match epilogue {
            Epilogue::Store => self::columns::<false>(x, weight, bias, n_in, n_out, columns, y),
            Epilogue::Gelu => self::columns::<true>(x, weight, bias, n_in, n_out, columns, y),
        }
    }

    fn columns<const GELU: bool>(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
    ) {
        let (start, end) = (columns.start, columns.end);
        y.fill(0.0);
        for p in 0..n_in {
            let a = x[p];
            let w_row = &weight[p * n_out + start..p * n_out + end];
            for (sum, &w) in y.iter_mut().zip(w_row) {
                *sum += a * w;
            }
        }
        for (value, &b) in y.iter_mut().zip(&bias[start..end]) {
            let sum = *value + b;
            *value = if GELU { gelu(sum) } else { sum };
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
        match epilogue {
            Epilogue::Store => {
                int8_rows::<false>(x, weight, scale, bias, rows, n_in, n_out, y, group)
            }
            Epilogue::Gelu => {
                int8_rows::<true>(x, weight, scale, bias, rows, n_in, n_out, y, group)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_rows<const GELU: bool>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
        group: usize,
    ) {
        for i in 0..rows {
            for column in 0..n_out {
                let mut sum = 0.0;
                let mut p = 0;
                while p < n_in {
                    let end = (p / group * group + group).min(n_in);
                    let mut partial = 0.0;
                    for q in p..end {
                        partial += x[i * n_in + q] * weight[q * n_out + column] as f32;
                    }
                    sum += partial * scale[p / group * n_out + column];
                    p = end;
                }
                let value = sum + bias[column];
                y[i * n_out + column] = if GELU { gelu(value) } else { value };
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn int8_linear_columns(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
        epilogue: Epilogue,
        group: usize,
    ) {
        match epilogue {
            Epilogue::Store => {
                int8_columns::<false>(x, weight, scale, bias, n_in, n_out, columns, y, group)
            }
            Epilogue::Gelu => {
                int8_columns::<true>(x, weight, scale, bias, n_in, n_out, columns, y, group)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn int8_columns<const GELU: bool>(
        x: &[f32],
        weight: &[i8],
        scale: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
        group: usize,
    ) {
        for (offset, column) in columns.enumerate() {
            let mut sum = 0.0;
            let mut p = 0;
            while p < n_in {
                let end = (p / group * group + group).min(n_in);
                let mut partial = 0.0;
                for q in p..end {
                    partial += x[q] * weight[q * n_out + column] as f32;
                }
                sum += partial * scale[p / group * n_out + column];
                p = end;
            }
            let value = sum + bias[column];
            y[offset] = if GELU { gelu(value) } else { value };
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
    dot, int8_dot, int8_linear_columns, int8_linear_rows, linear_columns, linear_rows, weighted_sum,
};

#[cfg(not(target_arch = "aarch64"))]
pub(crate) use scalar::{
    dot, int8_dot, int8_linear_columns, int8_linear_rows, linear_columns, linear_rows, weighted_sum,
};

#[cfg(test)]
mod int8_tests {
    use super::{gelu, int8_dot, int8_linear_columns, int8_linear_rows, Epilogue};

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
            int8_linear_columns(
                x,
                weight,
                scale,
                bias,
                n_in,
                n_out,
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
        for i in 0..rows {
            for j in 0..n_out {
                let mut sum = 0.0f64;
                for (g, _) in (0..n_in).step_by(group).enumerate() {
                    let end = (g * group + group).min(n_in);
                    let mut partial = 0.0f64;
                    for p in g * group..end {
                        partial += x[i * n_in + p] as f64 * weight[p * n_out + j] as f64;
                    }
                    sum += partial * scale[g * n_out + j] as f64;
                }
                let value = sum as f32 + bias[j];
                y[i * n_out + j] = if gelu_epilogue { gelu(value) } else { value };
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
