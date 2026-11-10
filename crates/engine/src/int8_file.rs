use std::path::Path;

use serde_json::{json, Map, Value};

use crate::quantize::GROUP_SIZE;
use crate::safetensors::SafeTensors;
use crate::{Config, Error};

pub fn write_int8(model_dir: &Path, out: &Path) -> Result<(), Error> {
    let config = Config::load(model_dir)?;
    let mut tensors = SafeTensors::open(&model_dir.join("model.safetensors"))?;
    let mut writer = Writer::default();

    let n_embd = config.n_embd;
    let n_inner = 4 * n_embd;

    let (data, scale) = tensors.quantize_rows("wte.weight", config.vocab_size, n_embd)?;
    writer.int8("wte.weight", &[config.vocab_size, n_embd], &data);
    writer.f32(
        "wte.weight.scales",
        &[config.vocab_size, n_embd.div_ceil(GROUP_SIZE)],
        &scale,
    );
    writer.f32_matrix(&mut tensors, "wpe.weight", config.n_ctx, n_embd)?;

    for layer in 0..config.n_layer {
        let prefix = format!("h.{layer}");
        writer.f32_vector(&mut tensors, &format!("{prefix}.ln_1.weight"), n_embd)?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.ln_1.bias"), n_embd)?;
        writer.int8_output_rows(
            &mut tensors,
            &format!("{prefix}.attn.c_attn.weight"),
            n_embd,
            3 * n_embd,
        )?;
        writer.f32_vector(
            &mut tensors,
            &format!("{prefix}.attn.c_attn.bias"),
            3 * n_embd,
        )?;
        writer.int8_output_rows(
            &mut tensors,
            &format!("{prefix}.attn.c_proj.weight"),
            n_embd,
            n_embd,
        )?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.attn.c_proj.bias"), n_embd)?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.ln_2.weight"), n_embd)?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.ln_2.bias"), n_embd)?;
        writer.int8_output_rows(
            &mut tensors,
            &format!("{prefix}.mlp.c_fc.weight"),
            n_embd,
            n_inner,
        )?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.mlp.c_fc.bias"), n_inner)?;
        writer.int8_output_rows(
            &mut tensors,
            &format!("{prefix}.mlp.c_proj.weight"),
            n_inner,
            n_embd,
        )?;
        writer.f32_vector(&mut tensors, &format!("{prefix}.mlp.c_proj.bias"), n_embd)?;
    }

    writer.f32_vector(&mut tensors, "ln_f.weight", n_embd)?;
    writer.f32_vector(&mut tensors, "ln_f.bias", n_embd)?;

    std::fs::write(out, writer.finish()?)?;
    Ok(())
}

#[derive(Default)]
struct Writer {
    header: Map<String, Value>,
    data: Vec<u8>,
    int8: Vec<(String, Vec<usize>, Vec<i8>)>,
}

impl Writer {
    fn f32_vector(
        &mut self,
        tensors: &mut SafeTensors,
        name: &str,
        len: usize,
    ) -> Result<(), Error> {
        let values = tensors.load_vector(name, len)?;
        self.f32(name, &[len], &values);
        Ok(())
    }

    fn f32_matrix(
        &mut self,
        tensors: &mut SafeTensors,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(), Error> {
        let values = tensors.load_matrix(name, rows, cols)?;
        self.f32(name, &[rows, cols], &values);
        Ok(())
    }

    fn int8_output_rows(
        &mut self,
        tensors: &mut SafeTensors,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<(), Error> {
        let (data, scale) = tensors.quantize_transposed(name, n_in, n_out)?;
        self.int8(name, &[n_out, n_in], &data);
        let groups = n_in.div_ceil(GROUP_SIZE);
        self.f32(&format!("{name}.scales"), &[n_out, groups], &scale);
        Ok(())
    }

    fn int8(&mut self, name: &str, shape: &[usize], values: &[i8]) {
        self.int8
            .push((name.to_string(), shape.to_vec(), values.to_vec()));
    }

    fn f32(&mut self, name: &str, shape: &[usize], values: &[f32]) {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.push(name, "F32", shape, &bytes);
    }

    // The int8 tensors are appended after every f32 tensor so that float data
    // stays 4-byte aligned.
    fn push(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
        let start = self.data.len();
        self.data.extend_from_slice(bytes);
        let end = self.data.len();
        self.header.insert(
            name.to_string(),
            json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }),
        );
    }

    fn finish(mut self) -> Result<Vec<u8>, Error> {
        for (name, shape, values) in std::mem::take(&mut self.int8) {
            let bytes: Vec<u8> = values.into_iter().map(|value| value as u8).collect();
            self.push(&name, "I8", &shape, &bytes);
        }

        let mut header = serde_json::to_vec(&Value::Object(self.header))?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::with_capacity(8 + header.len() + self.data.len());
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&self.data);
        Ok(bytes)
    }
}
