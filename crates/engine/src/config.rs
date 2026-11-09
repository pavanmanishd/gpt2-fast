use std::path::Path;

use serde_json::Value;

use crate::Error;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub layer_norm_epsilon: f32,
}

impl Config {
    pub(crate) fn load(model_dir: &Path) -> Result<Config, Error> {
        let text = std::fs::read_to_string(model_dir.join("config.json"))?;
        let value: Value = serde_json::from_str(&text)?;

        let n_ctx = usize_field(&value, "n_ctx")
            .or_else(|| usize_field(&value, "n_positions"))
            .ok_or_else(|| Error::new("config.json: missing n_ctx or n_positions"))?;
        let layer_norm_epsilon = value
            .get("layer_norm_epsilon")
            .and_then(Value::as_f64)
            .ok_or_else(|| Error::new("config.json: missing layer_norm_epsilon"))?
            as f32;

        let config = Config {
            n_layer: usize_field(&value, "n_layer")
                .ok_or_else(|| Error::new("config.json: missing n_layer"))?,
            n_head: usize_field(&value, "n_head")
                .ok_or_else(|| Error::new("config.json: missing n_head"))?,
            n_embd: usize_field(&value, "n_embd")
                .ok_or_else(|| Error::new("config.json: missing n_embd"))?,
            n_ctx,
            vocab_size: usize_field(&value, "vocab_size")
                .ok_or_else(|| Error::new("config.json: missing vocab_size"))?,
            layer_norm_epsilon,
        };

        if config.n_head == 0 || !config.n_embd.is_multiple_of(config.n_head) {
            return Err(Error::new("config.json: n_embd is not divisible by n_head"));
        }
        Ok(config)
    }
}

fn usize_field(value: &Value, key: &str) -> Option<usize> {
    value.get(key).and_then(Value::as_u64).map(|n| n as usize)
}
