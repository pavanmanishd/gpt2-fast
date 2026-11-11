#![cfg_attr(
    all(target_arch = "wasm32", target_feature = "atomics"),
    feature(stdarch_wasm_atomic_wait)
)]

mod config;
mod error;
mod int8_file;
mod model;
mod parallel;
mod quantize;
mod safetensors;
mod simd;
mod tokenizer;

pub use config::Config;
pub use error::Error;
pub use int8_file::write_int8;
pub use model::{Activations, Model, Weights};
#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
pub use parallel::{thread_pool_pointer, thread_pool_worker_main, thread_pool_workers};
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
