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

pub(crate) fn quantize_activations(
    x: &[f32],
    rows: usize,
    n_in: usize,
    group: usize,
    values: &mut [i8],
    scales: &mut [f32],
) {
    assert!(group > 0, "activation group must be positive");
    let groups = n_in.div_ceil(group);
    assert!(x.len() >= rows * n_in, "activation input is too short");
    assert!(
        values.len() >= rows * n_in,
        "activation output is too short"
    );
    assert!(
        scales.len() >= rows * groups,
        "activation scales are too short"
    );

    let mut index = 0;
    let mut scale_index = 0;
    for row in x[..rows * n_in].chunks(n_in) {
        for chunk in row.chunks(group) {
            let max = chunk.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = max / 127.0;
            scales[scale_index] = scale;
            scale_index += 1;
            for &value in chunk {
                values[index] = round(value, scale);
                index += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{quantize_activations, Rows};

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

    #[test]
    fn activations_round_trip_within_half_a_scale_and_clamp() {
        for (rows, n_in, group) in [
            (1usize, 16usize, 16usize),
            (1, 37, 16),
            (3, 65, 32),
            (2, 96, 64),
        ] {
            let mut values: Vec<f32> = (0..rows * n_in)
                .map(|i| ((i * 41 + 13) % 157) as f32 / 23.0 - 3.0)
                .collect();
            values[0] = 40.0;
            if rows > 1 {
                values[n_in] = 0.0;
            }

            let groups = n_in.div_ceil(group);
            let mut data = vec![0i8; rows * n_in];
            let mut scale = vec![0.0f32; rows * groups];
            quantize_activations(&values, rows, n_in, group, &mut data, &mut scale);

            assert_eq!(scale.len(), rows * groups);
            for (index, &value) in values.iter().enumerate() {
                let row = index / n_in;
                let column = index % n_in;
                let group_index = row * groups + column / group;
                assert!(data[index] >= -127);
                let restored = data[index] as f32 * scale[group_index];
                let error = (restored - value).abs();
                assert!(
                    error <= scale[group_index] / 2.0 * (1.0 + 1e-6),
                    "element {index}: value {value}, restored {restored}, scale {}",
                    scale[group_index]
                );
            }
        }
    }

    #[test]
    fn all_zero_activation_groups_quantize_to_zero_scale() {
        let mut values = vec![0.0f32; 48];
        values[20] = -2.0;
        let mut data = vec![0i8; values.len()];
        let mut scale = vec![0.0f32; 3];
        quantize_activations(&values, 1, values.len(), 16, &mut data, &mut scale);

        assert_eq!(scale[0], 0.0);
        assert_eq!(scale[2], 0.0);
        assert!(data[..16].iter().all(|&value| value == 0));
        assert!(data[32..].iter().all(|&value| value == 0));
        assert_eq!(data[20], -127);
    }
}
