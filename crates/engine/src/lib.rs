mod config;
mod error;
mod model;
mod safetensors;
mod simd;
mod tokenizer;

pub use config::Config;
pub use error::Error;
pub use model::Model;
pub use tokenizer::Tokenizer;

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &value) in logits.iter().enumerate().skip(1) {
        if value > logits[best] {
            best = i;
        }
    }
    best as u32
}
