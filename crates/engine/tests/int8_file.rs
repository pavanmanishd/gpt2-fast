use std::fs;
use std::path::PathBuf;

use gpt2_engine::{write_int8, Model, Tokenizer, Weights};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gpt2")
}

#[test]
fn the_prequantized_file_matches_load_time_quantization() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping int8 file test: {} is missing", dir.display());
        return;
    }
    let tokenizer = Tokenizer::load(&dir).expect("tokenizer loads");
    let ids = tokenizer.encode("Hello, my name is");

    let quantized = Model::load_with(&dir, Weights::Int8).expect("int8 model loads");
    let expected = quantized.logits(&ids);

    let path = std::env::temp_dir().join(format!("gpt2-int8-{}.safetensors", std::process::id()));
    write_int8(&dir, &path).expect("int8 file is written");
    let config = fs::read_to_string(dir.join("config.json")).expect("config is readable");
    let bytes = fs::read(&path).expect("int8 file is readable");
    let _ = fs::remove_file(&path);

    let from_file =
        Model::from_bytes(&config, &bytes, Weights::Int8).expect("int8 file model loads");
    let actual = from_file.logits(&ids);
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "element {index}: loaded {a}, quantized at load {e}"
        );
    }
}
