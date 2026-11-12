use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value;

use crate::quantize::{Rows, GROUP_SIZE};
use crate::Error;

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) struct SafeTensors<'a> {
    source: Source<'a>,
    data_start: usize,
    tensors: HashMap<String, Tensor>,
}

enum Source<'a> {
    File(File),
    Bytes(&'a [u8]),
}

struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl SafeTensors<'static> {
    pub(crate) fn open(path: &Path) -> Result<SafeTensors<'static>, Error> {
        let mut file = File::open(path)?;
        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)?;
        let header_length = header_length(&length_bytes)?;
        let mut header_bytes = vec![0u8; header_length as usize];
        file.read_exact(&mut header_bytes)?;

        Ok(SafeTensors {
            source: Source::File(file),
            data_start: 8 + header_length as usize,
            tensors: parse_header(&header_bytes)?,
        })
    }
}

impl<'a> SafeTensors<'a> {
    pub(crate) fn from_bytes(bytes: &'a [u8]) -> Result<SafeTensors<'a>, Error> {
        let (length_bytes, rest) = bytes.split_at_checked(8).ok_or_else(|| {
            Error::new("safetensors: buffer is shorter than the header length prefix")
        })?;
        let header_length = header_length(length_bytes)?;
        let (header_bytes, _) = rest
            .split_at_checked(header_length as usize)
            .ok_or_else(|| Error::new("safetensors: header extends past the end of the buffer"))?;

        Ok(SafeTensors {
            source: Source::Bytes(bytes),
            data_start: 8 + header_length as usize,
            tensors: parse_header(header_bytes)?,
        })
    }

    pub(crate) fn load_vector(&mut self, name: &str, len: usize) -> Result<Vec<f32>, Error> {
        let (shape, data) = self.load_f32(name)?;
        if shape != [len] {
            return Err(Error::new(format!(
                "tensor `{name}` has shape {shape:?}, expected [{len}]"
            )));
        }
        Ok(data)
    }

    pub(crate) fn load_matrix(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, Error> {
        let (shape, data) = self.load_f32(name)?;
        if shape != [rows, cols] {
            return Err(Error::new(format!(
                "tensor `{name}` has shape {shape:?}, expected [{rows}, {cols}]"
            )));
        }
        Ok(data)
    }

    pub(crate) fn load_matrix_transposed(
        &mut self,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<f32>, Error> {
        let input_major = self.load_matrix(name, n_in, n_out)?;
        let mut output_major = vec![0.0; n_in * n_out];
        for p in 0..n_in {
            for j in 0..n_out {
                output_major[j * n_in + p] = input_major[p * n_out + j];
            }
        }
        Ok(output_major)
    }

    pub(crate) fn load_matrix_f16(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<u16>, Error> {
        let mut output = vec![0u16; rows * cols];
        self.for_each_row(name, rows, cols, |row, values| {
            let target = &mut output[row * cols..(row + 1) * cols];
            for (value, &source) in target.iter_mut().zip(values) {
                *value = crate::half::f32_to_f16(source);
            }
        })?;
        Ok(output)
    }

    pub(crate) fn load_matrix_transposed_f16(
        &mut self,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Vec<u16>, Error> {
        let mut output = vec![0u16; n_in * n_out];
        self.for_each_row(name, n_in, n_out, |input, values| {
            for (output_row, &value) in values.iter().enumerate() {
                output[output_row * n_in + input] = crate::half::f32_to_f16(value);
            }
        })?;
        Ok(output)
    }

    pub(crate) fn load_int8(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Option<Vec<i8>>, Error> {
        let Some(tensor) = self.find(name) else {
            return Ok(None);
        };
        if tensor.dtype == "F32" {
            return Ok(None);
        }
        if tensor.dtype != "I8" {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has dtype {}, expected I8",
                tensor.dtype
            )));
        }
        if tensor.shape != [rows, cols] {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has shape {:?}, expected [{rows}, {cols}]",
                tensor.shape
            )));
        }
        let count = rows * cols;
        if tensor.end - tensor.start != count {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has inconsistent data offsets"
            )));
        }

        let start = tensor.start;
        let mut bytes = vec![0u8; count];
        self.read(start, &mut bytes)?;
        Ok(Some(bytes.into_iter().map(|byte| byte as i8).collect()))
    }

    fn load_f32(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), Error> {
        let tensor = self
            .find(name)
            .ok_or_else(|| Error::new(format!("safetensors: missing tensor `{name}`")))?;
        if tensor.dtype != "F32" {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has dtype {}, expected F32",
                tensor.dtype
            )));
        }

        let shape = tensor.shape.clone();
        let count: usize = shape.iter().product();
        let byte_len = count * 4;
        if tensor.end - tensor.start != byte_len {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has inconsistent data offsets"
            )));
        }

        let start = tensor.start;
        let mut bytes = vec![0u8; byte_len];
        self.read(start, &mut bytes)?;
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        Ok((shape, data))
    }

    fn read(&mut self, offset: usize, out: &mut [u8]) -> Result<(), Error> {
        let start = self
            .data_start
            .checked_add(offset)
            .ok_or_else(|| Error::new("safetensors: tensor offset overflows"))?;
        match &mut self.source {
            Source::File(file) => {
                file.seek(SeekFrom::Start(start as u64))?;
                file.read_exact(out)?;
            }
            Source::Bytes(bytes) => {
                let end = start
                    .checked_add(out.len())
                    .ok_or_else(|| Error::new("safetensors: tensor offset overflows"))?;
                let data = bytes.get(start..end).ok_or_else(|| {
                    Error::new("safetensors: tensor data extends past the end of the buffer")
                })?;
                out.copy_from_slice(data);
            }
        }
        Ok(())
    }

    pub(crate) fn quantize_transposed(
        &mut self,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), Error> {
        self.quantize_transposed_group(name, n_in, n_out, GROUP_SIZE)
    }

    pub(crate) fn quantize_transposed_group(
        &mut self,
        name: &str,
        n_in: usize,
        n_out: usize,
        group: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), Error> {
        let group = crate::quantize::group_size(group, n_in);
        let values = self.load_matrix_transposed(name, n_in, n_out)?;
        let mut quantizer = Rows::new(n_out, n_in, group);
        for row in values.chunks(n_in) {
            quantizer.push(row);
        }
        Ok(quantizer.into_parts())
    }

    pub(crate) fn quantize_rows(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), Error> {
        self.quantize_rows_group(name, rows, cols, GROUP_SIZE)
    }

    pub(crate) fn quantize_rows_group(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        group: usize,
    ) -> Result<(Vec<i8>, Vec<f32>), Error> {
        let group = crate::quantize::group_size(group, cols);
        let mut quantizer = Rows::new(rows, cols, group);
        self.for_each_row(name, rows, cols, |_, values| quantizer.push(values))?;
        Ok(quantizer.into_parts())
    }

    fn for_each_row(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        mut visit: impl FnMut(usize, &[f32]),
    ) -> Result<(), Error> {
        let start = self.f32_tensor_start(name, &[rows, cols])?;
        let row_bytes = cols * 4;
        let block_rows = (256 * 1024 / row_bytes).clamp(1, 64);
        let mut bytes = vec![0u8; block_rows * row_bytes];
        let mut values = vec![0.0f32; block_rows * cols];
        let mut row = 0;
        while row < rows {
            let count = (rows - row).min(block_rows);
            let length = count * row_bytes;
            self.read(start + row * row_bytes, &mut bytes[..length])?;
            let (chunks, _) = bytes[..length].as_chunks::<4>();
            for (value, chunk) in values[..count * cols].iter_mut().zip(chunks) {
                *value = f32::from_le_bytes(*chunk);
            }
            for (offset, values) in values[..count * cols].chunks(cols).enumerate() {
                visit(row + offset, values);
            }
            row += count;
        }
        Ok(())
    }

    fn f32_tensor_start(&self, name: &str, shape: &[usize]) -> Result<usize, Error> {
        let tensor = self
            .find(name)
            .ok_or_else(|| Error::new(format!("safetensors: missing tensor `{name}`")))?;
        if tensor.dtype != "F32" {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has dtype {}, expected F32",
                tensor.dtype
            )));
        }
        if tensor.shape != shape {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has shape {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
        if tensor.end - tensor.start != shape.iter().product::<usize>() * 4 {
            return Err(Error::new(format!(
                "safetensors: tensor `{name}` has inconsistent data offsets"
            )));
        }
        Ok(tensor.start)
    }

    fn find(&self, name: &str) -> Option<&Tensor> {
        self.tensors
            .get(name)
            .or_else(|| self.tensors.get(&format!("transformer.{name}")))
    }
}

fn header_length(bytes: &[u8]) -> Result<u64, Error> {
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| Error::new("safetensors: missing header length"))?;
    let length = u64::from_le_bytes(bytes);
    if length > MAX_HEADER_BYTES {
        return Err(Error::new("safetensors: implausible header length"));
    }
    Ok(length)
}

fn parse_header(bytes: &[u8]) -> Result<HashMap<String, Tensor>, Error> {
    let header: Value = serde_json::from_slice(bytes)?;
    let header = header
        .as_object()
        .ok_or_else(|| Error::new("safetensors: header is not a JSON object"))?;

    let mut tensors = HashMap::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        tensors.insert(name.clone(), parse_tensor(name, info)?);
    }
    Ok(tensors)
}

fn parse_tensor(name: &str, info: &Value) -> Result<Tensor, Error> {
    let dtype = info
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::new(format!("safetensors: tensor `{name}` has no dtype")))?
        .to_string();
    let shape = info
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new(format!("safetensors: tensor `{name}` has no shape")))?
        .iter()
        .map(|dim| {
            dim.as_u64().map(|n| n as usize).ok_or_else(|| {
                Error::new(format!("safetensors: tensor `{name}` has invalid shape"))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let offsets = info
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::new(format!("safetensors: tensor `{name}` has no data_offsets")))?;
    if offsets.len() != 2 {
        return Err(Error::new(format!(
            "safetensors: tensor `{name}` has invalid data_offsets"
        )));
    }
    let start = offsets[0]
        .as_u64()
        .ok_or_else(|| Error::new(format!("safetensors: tensor `{name}` has invalid offsets")))?
        as usize;
    let end = offsets[1]
        .as_u64()
        .ok_or_else(|| Error::new(format!("safetensors: tensor `{name}` has invalid offsets")))?
        as usize;
    if end < start {
        return Err(Error::new(format!(
            "safetensors: tensor `{name}` has invalid offsets"
        )));
    }

    Ok(Tensor {
        dtype,
        shape,
        start,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::SafeTensors;

    fn encode(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut data = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in entries {
            let start = data.len();
            data.extend_from_slice(bytes);
            let end = data.len();
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }

        let mut header =
            serde_json::to_vec(&serde_json::Value::Object(header)).expect("header is valid JSON");
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&data);
        bytes
    }

    fn sample() -> Vec<u8> {
        let floats: Vec<u8> = [1.0f32, -2.0, 3.5, 4.25]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let integers: Vec<u8> = [-1i8, 2, -3, 4].iter().map(|value| *value as u8).collect();
        encode(&[
            ("weight", "F32", &[2, 2], &floats),
            ("weight.scales", "F32", &[2], &floats[..8]),
            ("quantized", "I8", &[2, 2], &integers),
        ])
    }

    #[test]
    fn loads_tensors_from_memory() {
        let bytes = sample();
        let mut tensors = SafeTensors::from_bytes(&bytes).expect("parse");
        assert_eq!(
            tensors.load_matrix("weight", 2, 2).expect("f32 matrix"),
            vec![1.0, -2.0, 3.5, 4.25]
        );
        assert_eq!(
            tensors.load_vector("weight.scales", 2).expect("f32 vector"),
            vec![1.0, -2.0]
        );
        assert_eq!(
            tensors.load_int8("quantized", 2, 2).expect("int8 matrix"),
            Some(vec![-1, 2, -3, 4])
        );
    }

    #[test]
    fn transposes_an_input_major_matrix() {
        let bytes = sample();
        let mut tensors = SafeTensors::from_bytes(&bytes).expect("parse");
        assert_eq!(
            tensors
                .load_matrix_transposed("weight", 2, 2)
                .expect("transposed f32 matrix"),
            vec![1.0, 3.5, -2.0, 4.25]
        );
    }

    #[test]
    fn stores_matrices_as_output_major_halves() {
        let bytes = sample();
        let mut tensors = SafeTensors::from_bytes(&bytes).expect("parse");
        let half = |value: f32| crate::half::f32_to_f16(value);
        assert_eq!(
            tensors.load_matrix_f16("weight", 2, 2).expect("f16 matrix"),
            vec![half(1.0), half(-2.0), half(3.5), half(4.25)]
        );
        let mut tensors = SafeTensors::from_bytes(&bytes).expect("parse");
        assert_eq!(
            tensors
                .load_matrix_transposed_f16("weight", 2, 2)
                .expect("transposed f16 matrix"),
            vec![half(1.0), half(3.5), half(-2.0), half(4.25)]
        );
    }

    #[test]
    fn an_f32_tensor_is_not_mistaken_for_prequantized_data() {
        let bytes = sample();
        let mut tensors = SafeTensors::from_bytes(&bytes).expect("parse");
        assert_eq!(
            tensors.load_int8("weight", 2, 2).expect("f32 fallback"),
            None
        );
    }

    #[test]
    fn rejects_a_truncated_buffer() {
        let bytes = sample();
        for length in [0, 4, 8, 32] {
            assert!(SafeTensors::from_bytes(&bytes[..length.min(bytes.len())]).is_err());
        }
    }
}
