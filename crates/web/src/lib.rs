use wasm_bindgen::prelude::*;

use gpt2_engine::{Model, Tokenizer, Weights};

#[wasm_bindgen]
pub fn simd_enabled() -> bool {
    cfg!(all(target_arch = "wasm32", target_feature = "simd128"))
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[wasm_bindgen]
pub fn shared_memory() -> JsValue {
    wasm_bindgen::memory()
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[wasm_bindgen]
pub fn thread_pool_pointer() -> u32 {
    gpt2_engine::thread_pool_pointer() as u32
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[wasm_bindgen]
pub fn thread_pool_workers() -> usize {
    gpt2_engine::thread_pool_workers()
}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
#[wasm_bindgen]
pub fn thread_pool_worker_entry(pointer: u32, index: usize) {
    gpt2_engine::thread_pool_worker_main(pointer as usize, index);
}

#[wasm_bindgen]
pub struct Engine {
    model: Model,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(
        config_json: &str,
        vocab_json: &str,
        merges_text: &str,
        weights_bytes: &[u8],
        weights: &str,
        threads: usize,
    ) -> Result<Engine, JsValue> {
        let weights = Weights::parse(weights)
            .ok_or_else(|| JsValue::from_str("weights must be f32 or int8"))?;
        let mut model = Model::from_bytes(config_json, weights_bytes, weights)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        if threads > 1 {
            model
                .set_threads(threads)
                .map_err(|error| JsValue::from_str(&error.to_string()))?;
        }
        let tokenizer = Tokenizer::from_text(vocab_json, merges_text)
            .map_err(|error| JsValue::from_str(&error.to_string()))?;
        Ok(Engine { model, tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    pub fn logits(&self, ids: &[u32]) -> Vec<f32> {
        self.model.logits(ids)
    }

    pub fn generate(
        &self,
        ids: &[u32],
        max_new_tokens: usize,
        on_token: &js_sys::Function,
    ) -> Result<(), JsValue> {
        let mut callback_error = None;
        self.model.generate(ids, max_new_tokens, |token| {
            if callback_error.is_none() {
                if let Err(error) = on_token.call1(&JsValue::NULL, &JsValue::from(token)) {
                    callback_error = Some(error);
                }
            }
        });
        match callback_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub fn weight_manifest(&self) -> String {
        self.model.weight_manifest()
    }

    pub fn weight(&self, name: &str) -> Option<Vec<f32>> {
        self.model.weight(name)
    }

    pub fn weight_int8(&self, name: &str) -> Option<Vec<u32>> {
        self.model.weight_int8(name)
    }

    pub fn weight_scales(&self, name: &str) -> Option<Vec<f32>> {
        self.model.weight_scales(name)
    }

    pub fn release_weights(&mut self) {
        self.model.release_weights();
    }
}
