pub(crate) const GROUP_SIZE: usize = 64;

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
    use super::Rows;

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
