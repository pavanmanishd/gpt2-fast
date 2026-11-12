pub(crate) const GROUP_SIZE: usize = 64;

// Zero selects one group per row.
#[inline]
pub(crate) fn group_size(group: usize, n_in: usize) -> usize {
    if group == 0 {
        n_in
    } else {
        group
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

// Same quantization and scale bits as `quantize_activations`, written into the
// pair-interleaved layout the int8 prompt kernels read.
pub(crate) fn quantize_activations_interleaved(
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
        values.len() >= crate::int8_layout::storage_len(rows, n_in),
        "activation output is too short"
    );
    assert!(
        scales.len() >= rows * groups,
        "activation scales are too short"
    );

    let aligned = group.is_multiple_of(crate::int8_layout::BLOCK);
    for (row, values_row) in x[..rows * n_in].chunks(n_in).enumerate() {
        for (group_index, chunk) in values_row.chunks(group).enumerate() {
            let max = chunk.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let scale = max / 127.0;
            scales[row * groups + group_index] = scale;
            let group_start = group_index * group;
            if aligned {
                for (block_index, block) in chunk.chunks(crate::int8_layout::BLOCK).enumerate() {
                    let column = group_start + block_index * crate::int8_layout::BLOCK;
                    let at = crate::int8_layout::block_offset(n_in, row, column);
                    let mut bytes = [0i8; crate::int8_layout::BLOCK];
                    for (byte, &value) in bytes.iter_mut().zip(block) {
                        *byte = round(value, scale);
                    }
                    values[at..at + block.len()].copy_from_slice(&bytes[..block.len()]);
                }
            } else {
                for (offset, &value) in chunk.iter().enumerate() {
                    values[crate::int8_layout::byte_offset(n_in, row, group_start + offset)] =
                        round(value, scale);
                }
            }
        }
    }
    crate::int8_layout::clear_padding(rows, n_in, values);
}

// The largest magnitudes are moved out of the vector so one value cannot flatten
// the resolution of the whole group it sits in; they are added back exactly.
fn select_outliers(x: &[f32], keep: usize, indices: &mut [u32], removed: &mut [f32]) -> usize {
    debug_assert!(keep <= indices.len());
    let mut count = 0;
    for (index, &value) in x.iter().enumerate() {
        let magnitude = value.abs();
        if count < keep {
            let mut at = count;
            while at > 0 && removed[at - 1].abs() > magnitude {
                removed[at] = removed[at - 1];
                indices[at] = indices[at - 1];
                at -= 1;
            }
            removed[at] = value;
            indices[at] = index as u32;
            count += 1;
        } else if keep > 0 && magnitude > removed[0].abs() {
            let mut at = 0;
            while at + 1 < keep && removed[at + 1].abs() < magnitude {
                removed[at] = removed[at + 1];
                indices[at] = indices[at + 1];
                at += 1;
            }
            removed[at] = value;
            indices[at] = index as u32;
        }
    }
    for at in 1..count {
        let index = indices[at];
        let value = removed[at];
        let mut slot = at;
        while slot > 0 && indices[slot - 1] > index {
            indices[slot] = indices[slot - 1];
            removed[slot] = removed[slot - 1];
            slot -= 1;
        }
        indices[slot] = index;
        removed[slot] = value;
    }
    count
}

pub(crate) fn quantize_activation_outliers(
    x: &[f32],
    group: usize,
    keep: usize,
    values: &mut [i8],
    scales: &mut [f32],
    indices: &mut [u32],
    removed: &mut [f32],
) -> usize {
    assert!(group > 0, "activation group must be positive");
    assert!(
        keep <= indices.len() && keep <= removed.len(),
        "outlier buffer is too short"
    );
    assert!(values.len() >= x.len(), "activation output is too short");
    assert!(
        scales.len() >= x.len().div_ceil(group),
        "activation scales are too short"
    );

    let count = select_outliers(x, keep, indices, removed);
    for (group_index, chunk) in x.chunks(group).enumerate() {
        let start = group_index * group;
        let excluded = &indices[..count];
        let max = chunk
            .iter()
            .enumerate()
            .filter(|(offset, _)| excluded.binary_search(&((start + offset) as u32)).is_err())
            .fold(0.0f32, |max, (_, value)| max.max(value.abs()));
        let scale = max / 127.0;
        scales[group_index] = scale;
        for (offset, &value) in chunk.iter().enumerate() {
            let index = start + offset;
            values[index] = if excluded.binary_search(&(index as u32)).is_ok() {
                0
            } else {
                round(value, scale)
            };
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::{
        quantize_activation_outliers, quantize_activations, quantize_activations_interleaved, Rows,
    };
    use crate::int8_layout;

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
    fn interleaved_quantization_matches_the_plain_quantizer() {
        for (rows, n_in, group) in [
            (1usize, 37usize, 16usize),
            (2, 37, 16),
            (3, 64, 32),
            (5, 130, 16),
            (4, 96, 64),
            (7, 7, 7),
            (5, 768, 64),
        ] {
            let values: Vec<f32> = (0..rows * n_in)
                .map(|i| ((i * 41 + 13) % 157) as f32 / 23.0 - 3.0)
                .collect();
            let groups = n_in.div_ceil(group);
            let mut plain = vec![0i8; rows * n_in];
            let mut plain_scale = vec![0.0f32; rows * groups];
            quantize_activations(&values, rows, n_in, group, &mut plain, &mut plain_scale);

            let mut interleaved = vec![1i8; int8_layout::storage_len(rows, n_in)];
            let mut scale = vec![0.0f32; rows * groups];
            quantize_activations_interleaved(
                &values,
                rows,
                n_in,
                group,
                &mut interleaved,
                &mut scale,
            );

            for index in 0..rows * groups {
                assert_eq!(
                    scale[index].to_bits(),
                    plain_scale[index].to_bits(),
                    "rows {rows}, n_in {n_in}, group {group}, scale {index}"
                );
            }
            for row in 0..rows {
                for column in 0..n_in {
                    assert_eq!(
                        int8_layout::byte(&interleaved, n_in, row, column),
                        plain[row * n_in + column],
                        "rows {rows}, n_in {n_in}, group {group}, row {row}, column {column}"
                    );
                }
            }
        }
    }

    #[test]
    fn outlier_quantization_without_outliers_matches_plain_activations() {
        for (len, group) in [(1usize, 16usize), (37, 16), (64, 32), (65, 64), (130, 32)] {
            let values: Vec<f32> = (0..len)
                .map(|i| ((i * 41 + 13) % 157) as f32 / 23.0 - 3.0)
                .collect();
            let groups = len.div_ceil(group);
            let mut plain = vec![0i8; len];
            let mut plain_scale = vec![0.0f32; groups];
            quantize_activations(&values, 1, len, group, &mut plain, &mut plain_scale);

            let mut data = vec![1i8; len];
            let mut scale = vec![1.0f32; groups];
            let mut indices = vec![0u32; len];
            let mut removed = vec![0.0f32; len];
            let count = quantize_activation_outliers(
                &values,
                group,
                0,
                &mut data,
                &mut scale,
                &mut indices,
                &mut removed,
            );

            assert_eq!(count, 0);
            for index in 0..len {
                assert_eq!(data[index], plain[index], "element {index}");
            }
            for group_index in 0..groups {
                assert_eq!(
                    scale[group_index].to_bits(),
                    plain_scale[group_index].to_bits(),
                    "group {group_index}"
                );
            }
        }
    }

    #[test]
    fn outliers_are_zeroed_and_do_not_set_the_group_scale() {
        let values = [1.0, 2.0, 500.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut data = vec![1i8; values.len()];
        let mut scale = vec![1.0f32; 3];
        let mut indices = vec![0u32; 4];
        let mut removed = vec![0.0f32; 4];
        let count = quantize_activation_outliers(
            &values,
            4,
            1,
            &mut data,
            &mut scale,
            &mut indices,
            &mut removed,
        );

        assert_eq!(count, 1);
        assert_eq!(&indices[..count], &[2]);
        assert_eq!(removed[0], 500.0);
        assert_eq!(data[2], 0);
        assert_eq!(scale[0], 3.0 / 127.0);
        assert_eq!(data[0], (1.0f32 / scale[0]).round() as i8);
        assert_eq!(data[3], 127);
        assert_eq!(scale[2], 9.0 / 127.0);
    }

    #[test]
    fn outlier_selection_takes_the_largest_magnitudes_in_index_order() {
        let values = [0.5, -9.0, 1.0, 7.0, -2.0, 8.0, 3.0, -6.0];
        let mut data = vec![0i8; values.len()];
        let mut scale = vec![0.0f32; 2];
        let mut indices = vec![0u32; 3];
        let mut removed = vec![0.0f32; 3];
        let count = quantize_activation_outliers(
            &values,
            4,
            3,
            &mut data,
            &mut scale,
            &mut indices,
            &mut removed,
        );

        assert_eq!(count, 3);
        assert_eq!(&indices[..count], &[1, 3, 5]);
        assert_eq!(&removed[..count], &[-9.0, 7.0, 8.0]);
        assert_eq!(data[1], 0);
        assert_eq!(data[3], 0);
        assert_eq!(data[5], 0);
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
