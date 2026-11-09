pub(crate) const GROUP_SIZE: usize = 64;

pub(crate) struct Columns {
    group: usize,
    n_out: usize,
    scale: Vec<f32>,
    data: Vec<i8>,
}

impl Columns {
    pub(crate) fn new(n_in: usize, n_out: usize, group: usize) -> Columns {
        let groups = n_in.div_ceil(group);
        Columns {
            group,
            n_out,
            scale: vec![0.0; groups * n_out],
            data: Vec::with_capacity(n_in * n_out),
        }
    }

    pub(crate) fn observe(&mut self, row: usize, values: &[f32]) {
        let offset = row / self.group * self.n_out;
        for (scale, &value) in self.scale[offset..offset + self.n_out]
            .iter_mut()
            .zip(values)
        {
            *scale = scale.max(value.abs());
        }
    }

    pub(crate) fn finish(&mut self) {
        for scale in &mut self.scale {
            *scale /= 127.0;
        }
    }

    pub(crate) fn push(&mut self, row: usize, values: &[f32]) {
        let offset = row / self.group * self.n_out;
        self.data.extend(
            values
                .iter()
                .zip(&self.scale[offset..offset + self.n_out])
                .map(|(&value, &scale)| round(value, scale)),
        );
    }

    pub(crate) fn into_parts(self) -> (Vec<i8>, Vec<f32>) {
        (self.data, self.scale)
    }
}

pub(crate) struct Rows {
    group: usize,
    scale: Vec<f32>,
    data: Vec<i8>,
}

impl Rows {
    pub(crate) fn new(rows: usize, row_len: usize, group: usize) -> Rows {
        let groups = row_len.div_ceil(group);
        Rows {
            group,
            scale: Vec::with_capacity(rows * groups),
            data: Vec::with_capacity(rows * row_len),
        }
    }

    pub(crate) fn push(&mut self, row: &[f32]) {
        for values in row.chunks(self.group) {
            let max = values
                .iter()
                .fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = max / 127.0;
            self.scale.push(scale);
            self.data
                .extend(values.iter().map(|&value| round(value, scale)));
        }
    }

    pub(crate) fn into_parts(self) -> (Vec<i8>, Vec<f32>) {
        (self.data, self.scale)
    }
}

fn round(value: f32, scale: f32) -> i8 {
    if scale == 0.0 {
        return 0;
    }
    (value / scale).round().clamp(-127.0, 127.0) as i8
}

#[cfg(test)]
mod tests {
    use super::{Columns, Rows};

    fn check_round_trip(
        data: &[i8],
        scale: &[f32],
        original: &[f32],
        group_of: impl Fn(usize) -> usize,
    ) {
        for index in 0..original.len() {
            let group = group_of(index);
            let restored = data[index] as f32 * scale[group];
            let error = (restored - original[index]).abs();
            assert!(
                error <= scale[group] / 2.0 * (1.0 + 1e-6),
                "element {index}: value {}, restored {restored}, scale {}",
                original[index],
                scale[group]
            );
        }
    }

    #[test]
    fn columns_keep_one_scale_per_group_and_output_channel_within_half_a_scale() {
        let (rows, cols, group) = (9, 7, 4);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 23.0 - 2.0)
            .collect();
        let mut quantizer = Columns::new(rows, cols, group);
        for (row, values) in values.chunks(cols).enumerate() {
            quantizer.observe(row, values);
        }
        quantizer.finish();
        for (row, values) in values.chunks(cols).enumerate() {
            quantizer.push(row, values);
        }
        let (data, scale) = quantizer.into_parts();

        let groups = rows.div_ceil(group);
        assert_eq!(scale.len(), groups * cols);
        for g in 0..groups {
            for column in 0..cols {
                let max = (g * group..rows.min((g + 1) * group))
                    .map(|row| values[row * cols + column].abs())
                    .fold(0.0f32, f32::max);
                assert_eq!(scale[g * cols + column], max / 127.0);
            }
        }
        check_round_trip(&data, &scale, &values, |index| {
            index / cols / group * cols + index % cols
        });
    }

    #[test]
    fn rows_keep_one_scale_per_row_and_group_and_handle_zero_rows() {
        let (rows, row_len, group) = (5, 11, 4);
        let mut values: Vec<f32> = (0..rows * row_len)
            .map(|i| ((i * 41 + 3) % 89) as f32 / 17.0 - 2.5)
            .collect();
        values[row_len..2 * row_len].fill(0.0);

        let mut quantizer = Rows::new(rows, row_len, group);
        for row in values.chunks(row_len) {
            quantizer.push(row);
        }
        let (data, scale) = quantizer.into_parts();

        let groups = row_len.div_ceil(group);
        assert_eq!(scale.len(), rows * groups);
        for g in 0..groups {
            assert_eq!(scale[groups + g], 0.0);
        }
        check_round_trip(&data, &scale, &values, |index| {
            index / row_len * groups + index % row_len / group
        });
    }
}
