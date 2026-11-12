// Weights and quantized prompt rows are stored as pairs of rows: for rows
// `2p` and `2p+1`, each 16-byte block holds eight columns of row `2p`
// followed by the same eight columns of row `2p+1`.
//
// A 16-byte block is then a 2x8 int8 matrix for SMMLA, and the two rows of a
// decoded weight row stay in one cache line. Rows are padded to an even count
// and each row to whole eight-byte blocks, so every block is safe to load.

pub(crate) const BLOCK: usize = 8;

#[inline]
pub(crate) fn blocks(n_in: usize) -> usize {
    n_in.div_ceil(BLOCK)
}

#[inline]
pub(crate) fn pair_stride(n_in: usize) -> usize {
    blocks(n_in) * 2 * BLOCK
}

#[inline]
pub(crate) fn storage_len(rows: usize, n_in: usize) -> usize {
    rows.div_ceil(2) * pair_stride(n_in)
}

// Byte offset of one row's eight-byte block.
#[inline]
pub(crate) fn block_offset(n_in: usize, row: usize, column: usize) -> usize {
    (row / 2) * pair_stride(n_in) + (column / BLOCK) * 2 * BLOCK + (row % 2) * BLOCK
}

#[inline]
pub(crate) fn byte_offset(n_in: usize, row: usize, column: usize) -> usize {
    block_offset(n_in, row, column) + column % BLOCK
}

#[inline]
pub(crate) fn byte(data: &[i8], n_in: usize, row: usize, column: usize) -> i8 {
    data[byte_offset(n_in, row, column)]
}

// Writes one row into the pair layout. The destination block must already be
// zeroed where `plain` is shorter than eight bytes.
#[cfg(test)]
pub(crate) fn write_row(values: &mut [i8], n_in: usize, row: usize, plain: &[i8]) {
    for block in 0..blocks(n_in) {
        let start = block * BLOCK;
        let len = (n_in - start).min(BLOCK);
        let to = block_offset(n_in, row, start);
        values[to..to + len].copy_from_slice(&plain[start..start + len]);
    }
}

pub(crate) fn read_row(data: &[i8], n_in: usize, row: usize, out: &mut [i8]) {
    for block in 0..blocks(n_in) {
        let start = block * BLOCK;
        let len = (n_in - start).min(BLOCK);
        let from = block_offset(n_in, row, start);
        out[start..start + len].copy_from_slice(&data[from..from + len]);
    }
}

pub(crate) fn interleave_owned(rows: usize, n_in: usize, mut plain: Vec<i8>) -> Vec<i8> {
    let len = storage_len(rows, n_in);
    debug_assert!(plain.len() >= rows * n_in);
    plain.resize(len, 0);
    interleave_in_place(rows, n_in, &mut plain);
    plain
}

// Permutes `data` from row-major into pairs. `data` holds `rows * n_in` plain
// bytes followed by enough zeroed capacity for the padded layout. Pairs are
// moved from the last to the first so a destination never overwrites a source
// pair that has not been read yet.
pub(crate) fn interleave_in_place(rows: usize, n_in: usize, data: &mut [i8]) {
    let stride = pair_stride(n_in);
    let pairs = rows.div_ceil(2);
    debug_assert!(data.len() >= pairs * stride);
    let mut temp = vec![0i8; 2 * n_in];
    for pair in (0..pairs).rev() {
        let first = 2 * pair;
        temp[..n_in].copy_from_slice(&data[first * n_in..first * n_in + n_in]);
        if first + 1 < rows {
            temp[n_in..2 * n_in]
                .copy_from_slice(&data[(first + 1) * n_in..(first + 2) * n_in]);
        } else {
            temp[n_in..2 * n_in].fill(0);
        }
        let base = pair * stride;
        data[base..base + stride].fill(0);
        for block in 0..blocks(n_in) {
            let start = block * BLOCK;
            let len = (n_in - start).min(BLOCK);
            let to = base + block * 2 * BLOCK;
            data[to..to + len].copy_from_slice(&temp[start..start + len]);
            data[to + BLOCK..to + BLOCK + len]
                .copy_from_slice(&temp[n_in + start..n_in + start + len]);
        }
    }
}

// Clears the bytes that stand for columns at or past `n_in`, and the second
// row of a final unpaired row, so a padded load contributes nothing.
pub(crate) fn clear_padding(rows: usize, n_in: usize, values: &mut [i8]) {
    let padded = rows.div_ceil(2) * 2;
    for row in rows..padded {
        for block in 0..blocks(n_in) {
            let at = block_offset(n_in, row, block * BLOCK);
            values[at..at + BLOCK].fill(0);
        }
    }
    let tail = n_in % BLOCK;
    if tail != 0 {
        for row in 0..rows {
            let at = block_offset(n_in, row, n_in - tail);
            values[at + tail..at + BLOCK].fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        block_offset, byte, byte_offset, clear_padding, interleave_in_place, interleave_owned,
        read_row, storage_len, write_row,
    };

    fn plain(rows: usize, n_in: usize) -> Vec<i8> {
        (0..rows * n_in)
            .map(|i| ((i * 73 + 19) % 255) as i32 - 128)
            .map(|value| value as i8)
            .collect()
    }

    #[test]
    fn pairs_round_trip_on_odd_rows_and_short_blocks() {
        for (rows, n_in) in [
            (1usize, 1usize),
            (1, 8),
            (1, 13),
            (2, 7),
            (3, 15),
            (4, 16),
            (5, 37),
            (7, 64),
            (6, 96),
            (5, 768),
        ] {
            let source = plain(rows, n_in);
            let interleaved = interleave_owned(rows, n_in, source.clone());
            assert!(interleaved.len() >= storage_len(rows, n_in));
            for row in 0..rows {
                for column in 0..n_in {
                    assert_eq!(
                        byte(&interleaved, n_in, row, column),
                        source[row * n_in + column],
                        "rows {rows}, n_in {n_in}, row {row}, column {column}"
                    );
                }
                let mut row_bytes = vec![0i8; n_in];
                read_row(&interleaved, n_in, row, &mut row_bytes);
                assert_eq!(row_bytes, source[row * n_in..(row + 1) * n_in]);
            }
            assert!(block_offset(n_in, rows - 1, n_in - 1) < interleaved.len());
        }
    }

    #[test]
    fn in_place_interleave_matches_the_owned_copy() {
        for (rows, n_in) in [(5usize, 13usize), (7, 37), (4, 64), (6, 33), (3, 1)] {
            let source = plain(rows, n_in);
            let owned = interleave_owned(rows, n_in, source.clone());
            let mut in_place = source.clone();
            in_place.resize(storage_len(rows, n_in), 0);
            interleave_in_place(rows, n_in, &mut in_place);
            assert_eq!(in_place, owned, "rows {rows}, n_in {n_in}");
        }
    }

    #[test]
    fn padding_stands_for_zero() {
        let (rows, n_in) = (3usize, 13usize);
        let mut interleaved = interleave_owned(rows, n_in, plain(rows, n_in));
        clear_padding(rows, n_in, &mut interleaved);
        for column in n_in..16 {
            assert_eq!(interleaved[byte_offset(n_in, 0, column)], 0);
            assert_eq!(interleaved[byte_offset(n_in, 1, column)], 0);
        }
        for column in 0..n_in {
            assert_eq!(byte(&interleaved, n_in, 3, column), 0);
        }

        let mut values = vec![0i8; storage_len(2, 8)];
        write_row(&mut values, 8, 1, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(byte(&values, 8, 1, 7), 8);
        assert_eq!(byte(&values, 8, 0, 7), 0);
    }
}
