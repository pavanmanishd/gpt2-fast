use std::path::{Path, PathBuf};

use gpt2_engine::{Activations, Model, Tokenizer, Weights};
use serde_json::Value;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/gpt2")
}

fn prompt_ids(tokenizer: &Tokenizer) -> Option<Vec<Vec<u32>>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts.json");
    let text = std::fs::read_to_string(path).ok()?;
    let prompts: Value = serde_json::from_str(&text).ok()?;
    Some(
        prompts["prompts"]
            .as_array()?
            .iter()
            .map(|prompt| tokenizer.encode(prompt["text"].as_str().unwrap()))
            .collect(),
    )
}

fn single(model: &Model, ids: &[u32], max_new_tokens: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    model.generate(ids, max_new_tokens, |token| tokens.push(token));
    tokens
}

fn batched(model: &Model, ids: &[&[u32]], batch: usize, max_new_tokens: usize) -> Vec<Vec<u32>> {
    let prompts: Vec<&[u32]> = (0..batch).map(|index| ids[index % ids.len()]).collect();
    let mut generated = vec![Vec::new(); batch];
    model.generate_batch(&prompts, max_new_tokens, |tokens| {
        assert_eq!(tokens.len(), batch);
        for (sequence, &token) in tokens.iter().enumerate() {
            generated[sequence].push(token);
        }
    });
    for sequence in &generated {
        assert_eq!(sequence.len(), max_new_tokens);
    }
    generated
}

#[test]
fn batched_generation_matches_single_sequence_generation() {
    let dir = model_dir();
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping model test: {} is missing", dir.display());
        return;
    }
    let tokenizer = Tokenizer::load(Path::new(&dir)).expect("tokenizer loads");
    let prompts = prompt_ids(&tokenizer).expect("prompts.json loads");
    let ids: Vec<&[u32]> = prompts.iter().map(Vec::as_slice).collect();
    assert_eq!(ids.len(), 5);

    for (weights, activations) in [
        (Weights::F32, Activations::F32),
        (Weights::Int8, Activations::F32),
        (Weights::Int8, Activations::Int8),
    ] {
        let mut model =
            Model::load_with_activations(&dir, weights, activations).expect("model loads");
        for threads in [1, 4] {
            model.set_threads(threads).expect("thread pool");
            let reference: Vec<Vec<u32>> = ids
                .iter()
                .map(|prompt| single(&model, prompt, 128))
                .collect();
            for batch in [1, 2, 5, 8, 16] {
                let generated = batched(&model, &ids, batch, 128);
                for (sequence, tokens) in generated.iter().enumerate() {
                    assert_eq!(
                        tokens,
                        &reference[sequence % ids.len()],
                        "{:?} {:?}, threads {threads}, batch {batch}, sequence {sequence}",
                        weights.name(),
                        activations.name()
                    );
                }
            }
        }
    }
}
