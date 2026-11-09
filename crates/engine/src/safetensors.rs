use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use serde_json::Value;

use crate::Error;

pub(crate) struct SafeTensors {
    file: File,
    data_start: u64,
    tensors: HashMap<String, Tensor>,
}

struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl SafeTensors {
    pub(crate) fn open(path: &Path) -> Result<SafeTensors, Error> {
        let mut file = File::open(path)?;

        let mut length_bytes = [0u8; 8];
        file.read_exact(&mut length_bytes)?;
        let header_length = u64::from_le_bytes(length_bytes);
        if header_length > 64 * 1024 * 1024 {
            return Err(Error::new("safetensors: implausible header length"));
        }

        let mut header_bytes = vec![0u8; header_length as usize];
        file.read_exact(&mut header_bytes)?;
        let header: Value = serde_json::from_slice(&header_bytes)?;
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

        Ok(SafeTensors {
            file,
            data_start: 8 + header_length,
            tensors,
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

    fn load_f32(&mut self, name: &str) -> Result<(Vec<usize>, Vec<f32>), Error> {
        let tensor = self
            .tensors
            .get(name)
            .or_else(|| self.tensors.get(&format!("transformer.{name}")))
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

        self.file
            .seek(SeekFrom::Start(self.data_start + tensor.start as u64))?;
        let mut bytes = vec![0u8; byte_len];
        self.file.read_exact(&mut bytes)?;
        let data = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        Ok((shape, data))
    }
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
