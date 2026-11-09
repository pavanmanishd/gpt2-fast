pub(crate) struct Columns {
    scale: Vec<f32>,
    data: Vec<i8>,
}

impl Columns {
    pub(crate) fn new(n_out: usize, capacity: usize) -> Columns {
        Columns {
            scale: vec![0.0; n_out],
            data: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn observe(&mut self, row: &[f32]) {
        for (scale, &value) in self.scale.iter_mut().zip(row) {
            *scale = scale.max(value.abs());
        }
    }

    pub(crate) fn finish(&mut self) {
        for scale in &mut self.scale {
            *scale /= 127.0;
        }
    }

    pub(crate) fn push(&mut self, row: &[f32]) {
        self.data.extend(
            row.iter()
                .zip(&self.scale)
                .map(|(&value, &scale)| round(value, scale)),
        );
    }

    pub(crate) fn into_parts(self) -> (Vec<i8>, Vec<f32>) {
        (self.data, self.scale)
    }
}

pub(crate) struct Rows {
    scale: Vec<f32>,
    data: Vec<i8>,
}

impl Rows {
    pub(crate) fn with_capacity(rows: usize, capacity: usize) -> Rows {
        Rows {
            scale: Vec::with_capacity(rows),
            data: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, row: &[f32]) {
        let max = row.iter().fold(0.0f32, |max, value| max.max(value.abs()));
        let scale = max / 127.0;
        self.data
            .extend(row.iter().map(|&value| round(value, scale)));
        self.scale.push(scale);
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
    fn columns_keep_one_scale_per_output_channel_within_half_a_scale() {
        let (rows, cols) = (9, 7);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 37 + 11) % 101) as f32 / 23.0 - 2.0)
            .collect();
        let mut quantizer = Columns::new(cols, rows * cols);
        for row in values.chunks(cols) {
            quantizer.observe(row);
        }
        quantizer.finish();
        for row in values.chunks(cols) {
            quantizer.push(row);
        }
        let (data, scale) = quantizer.into_parts();

        assert_eq!(scale.len(), cols);
        for column in 0..cols {
            let max = (0..rows)
                .map(|row| values[row * cols + column].abs())
                .fold(0.0f32, f32::max);
            assert_eq!(scale[column], max / 127.0);
        }
        check_round_trip(&data, &scale, &values, |index| index % cols);
    }

    #[test]
    fn rows_keep_one_scale_per_row_and_handle_zero_rows() {
        let (rows, row_len) = (5, 11);
        let mut values: Vec<f32> = (0..rows * row_len)
            .map(|i| ((i * 41 + 3) % 89) as f32 / 17.0 - 2.5)
            .collect();
        values[row_len..2 * row_len].fill(0.0);

        let mut quantizer = Rows::with_capacity(rows, rows * row_len);
        for row in values.chunks(row_len) {
            quantizer.push(row);
        }
        let (data, scale) = quantizer.into_parts();

        assert_eq!(scale.len(), rows);
        assert_eq!(scale[1], 0.0);
        check_round_trip(&data, &scale, &values, |index| index / row_len);
    }
}
