#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;
    use std::ops::Range;

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
        unsafe fn block<const R: usize, const V: usize>(&self, i: usize, j: usize, y: &mut [f32]) {
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
                let out = y
                    .as_mut_ptr()
                    .add((i + r) * self.columns + j - self.first_column);
                for (c, value) in acc[r].iter().enumerate() {
                    let b = vld1q_f32(bias.add(j + c * 4));
                    vst1q_f32(out.add(c * 4), vaddq_f32(*value, b));
                }
            }
        }

        fn block_rows<const R: usize, const V: usize>(&self, i: usize, y: &mut [f32]) {
            let end = self.first_column + self.columns;
            let mut j = self.first_column;
            while j + V * 4 <= end {
                // SAFETY: the loop condition guarantees `i + R <= rows`.
                unsafe { self.block::<R, V>(i, j, y) };
                j += V * 4;
            }
            if j < end {
                self.edge_columns::<R>(i, j, y);
            }
        }

        fn edge_columns<const R: usize>(&self, i: usize, j: usize, y: &mut [f32]) {
            for r in 0..R {
                for column in j..self.first_column + self.columns {
                    let mut sum = 0.0;
                    for p in 0..self.n_in {
                        sum +=
                            self.x[(i + r) * self.n_in + p] * self.weight[p * self.n_out + column];
                    }
                    y[(i + r) * self.columns + column - self.first_column] =
                        sum + self.bias[column];
                }
            }
        }
    }

    fn tiled<const R: usize, const V: usize>(
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
                kernel.block_rows::<R, V>(i, y);
                i += R;
            } else if remaining >= 4 {
                kernel.block_rows::<4, V>(i, y);
                i += 4;
            } else if remaining >= 2 {
                kernel.block_rows::<2, V>(i, y);
                i += 2;
            } else {
                kernel.block_rows::<1, V>(i, y);
                i += 1;
            }
        }
    }

    // Row slices of a multi-row matrix must stay on the 16-column kernel; the
    // rows == 1 kernel widens to 32 columns and rounds differently on the edge.
    pub(crate) fn linear_rows(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
        tiled::<4, 4>(x, weight, bias, rows, n_in, n_out, y);
    }

    pub(crate) fn linear_columns(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
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
        kernel.block_rows::<1, 8>(0, y);
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
        use super::{dot, linear_columns, linear_rows, weighted_sum};

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
                linear_columns(x, weight, bias, n_in, n_out, 0..n_out, &mut y);
            } else {
                linear_rows(x, weight, bias, rows, n_in, n_out, &mut y);
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

    const ROW_TILE: usize = 8;

    pub(crate) fn linear_rows(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        rows: usize,
        n_in: usize,
        n_out: usize,
        y: &mut [f32],
    ) {
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
                    *value += b;
                }
            }
        }
    }

    pub(crate) fn linear_columns(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        n_in: usize,
        n_out: usize,
        columns: Range<usize>,
        y: &mut [f32],
    ) {
        let (start, end) = (columns.start, columns.end);
        for p in 0..n_in {
            let a = x[p];
            let w_row = &weight[p * n_out + start..p * n_out + end];
            for (sum, &w) in y.iter_mut().zip(w_row) {
                *sum += a * w;
            }
        }
        for (value, &b) in y.iter_mut().zip(&bias[start..end]) {
            *value += b;
        }
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
pub(crate) use neon::{dot, linear_columns, linear_rows, weighted_sum};

#[cfg(not(target_arch = "aarch64"))]
pub(crate) use scalar::{dot, linear_columns, linear_rows, weighted_sum};
