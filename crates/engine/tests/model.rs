use std::path::PathBuf;

use gpt2_engine::{argmax, Model, Tokenizer};

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gpt2")
}

fn load_model() -> Option<Model> {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping model test: {} is missing", dir.display());
        return None;
    }
    Some(Model::load(&dir).expect("model loads"))
}

fn assert_close(actual: f32, expected: f32) {
    let difference = (actual - expected).abs();
    assert!(
        difference <= 1e-3,
        "expected {expected}, got {actual} (difference {difference})"
    );
}

#[test]
fn argmax_prefers_the_lowest_index_on_ties() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
    assert_eq!(argmax(&[5.0]), 0);
    assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
}

#[test]
fn logits_match_transformers() {
    let Some(model) = load_model() else {
        return;
    };
    let tokenizer = Tokenizer::load(&model_dir()).expect("tokenizer loads");
    let ids = tokenizer.encode("Hello, my name is");
    assert_eq!(ids, [15496, 11, 616, 1438, 318]);

    let logits = model.logits(&ids);
    assert_eq!(logits.len(), model.config().vocab_size);
    assert_eq!(argmax(&logits), 1757);
    assert_close(logits[1757], -64.4433);
    assert_close(logits[509], -64.6331);
    assert_close(logits[0], -68.4289);
    assert_close(logits[100], -77.7155);
    assert_close(logits[5000], -71.9565);
    assert_close(logits[50000], -76.9355);
    assert_close(logits[50256], -70.0094);
}

#[test]
fn generates_the_same_tokens_as_transformers() {
    let Some(model) = load_model() else {
        return;
    };
    let tokenizer = Tokenizer::load(&model_dir()).expect("tokenizer loads");
    let ids = tokenizer.encode("Hello, my name is");

    let mut generated = Vec::new();
    model.generate(&ids, 8, |token| generated.push(token));
    assert_eq!(generated, [1757, 13, 314, 1101, 257, 6260, 11, 290]);
    assert_eq!(tokenizer.decode(&generated), " John. I'm a writer, and");
}
