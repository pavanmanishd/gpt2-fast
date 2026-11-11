use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::process;
use std::time::Instant;

use gpt2_engine::{Activations, Model, Weights};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Request {
    #[serde(default)]
    ids: Option<Vec<u32>>,
    #[serde(default)]
    batch: Option<Vec<Vec<u32>>>,
    max_new_tokens: usize,
    #[serde(default)]
    return_logits: bool,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_dir = None;
    let mut weights = Weights::F32;
    let mut activations = Activations::F32;
    let mut logits_activations = None;
    let mut logits_outliers = None;
    let mut activation_group = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => match args.next() {
                Some(value) => model_dir = Some(value),
                None => fail("--model-dir needs a value"),
            },
            "--weights" => match args.next() {
                Some(value) => match Weights::parse(&value) {
                    Some(parsed) => weights = parsed,
                    None => fail(&format!("--weights must be f32 or int8, got `{value}`")),
                },
                None => fail("--weights needs a value"),
            },
            "--activations" => match args.next() {
                Some(value) => match Activations::parse(&value) {
                    Some(parsed) => activations = parsed,
                    None => fail(&format!("--activations must be f32 or int8, got `{value}`")),
                },
                None => fail("--activations needs a value"),
            },
            "--logits-activations" => match args.next() {
                Some(value) => match Activations::parse(&value) {
                    Some(parsed) => logits_activations = Some(parsed),
                    None => fail(&format!(
                        "--logits-activations must be f32 or int8, got `{value}`"
                    )),
                },
                None => fail("--logits-activations needs a value"),
            },
            "--logits-outliers" => match args.next() {
                Some(value) => match value.parse::<usize>() {
                    Ok(parsed) => logits_outliers = Some(parsed),
                    Err(_) => fail(&format!(
                        "--logits-outliers must be an integer, got `{value}`"
                    )),
                },
                None => fail("--logits-outliers needs a value"),
            },
            "--activation-group" => match args.next() {
                Some(value) => match value.parse::<usize>() {
                    Ok(parsed) => activation_group = Some(parsed),
                    Err(_) => fail(&format!(
                        "--activation-group must be an integer, got `{value}`"
                    )),
                },
                None => fail("--activation-group needs a value"),
            },
            other => fail(&format!("unknown argument `{other}`")),
        }
    }
    let Some(model_dir) = model_dir else {
        fail("usage: worker --model-dir <dir> [--weights f32|int8] [--activations f32|int8] [--activation-group 16|32|64] [--logits-activations f32|int8] [--logits-outliers <n>]");
    };

    let started = Instant::now();
    let mut model = match Model::load_with_activations(Path::new(&model_dir), weights, activations)
    {
        Ok(model) => model,
        Err(error) => fail(&format!("failed to load model: {error}")),
    };
    if let Some(activation_group) = activation_group {
        if let Err(error) = model.set_activation_group(activation_group) {
            fail(&error.to_string());
        }
    }
    if let Some(logits_activations) = logits_activations {
        model.set_logits_activations(logits_activations);
    }
    if let Some(logits_outliers) = logits_outliers {
        if let Err(error) = model.set_logits_outliers(logits_outliers) {
            fail(&error.to_string());
        }
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    eprintln!("weights: {}", weights.name());
    eprintln!("activations: {}", activations.name());
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    write_line(&mut out, &json!({"ready": true, "load_ms": load_ms}));

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => fail(&format!("cannot read stdin: {error}")),
        };
        if line.trim().is_empty() {
            continue;
        }
        let request: Request = match serde_json::from_str(&line) {
            Ok(request) => request,
            Err(error) => fail(&format!("invalid request: {error}")),
        };
        match (request.ids, request.batch) {
            (Some(ids), None) => {
                if ids.is_empty() {
                    fail("request has no ids");
                }
                if request.return_logits {
                    let logits = model.logits(&ids);
                    write_line(&mut out, &json!({"logits": logits}));
                }
                model.generate(&ids, request.max_new_tokens, |token| {
                    write_line(&mut out, &json!({"token": token}));
                });
            }
            (None, Some(batch)) => {
                if batch.is_empty() || batch.iter().any(Vec::is_empty) {
                    fail("batch request has an empty prompt");
                }
                if request.return_logits {
                    fail("batch request cannot return logits");
                }
                let prompts: Vec<&[u32]> = batch.iter().map(Vec::as_slice).collect();
                model.generate_batch(&prompts, request.max_new_tokens, |tokens| {
                    write_line(&mut out, &json!({"tokens": tokens}));
                });
            }
            (Some(_), Some(_)) => fail("request has both ids and batch"),
            (None, None) => fail("request has neither ids nor batch"),
        }
        write_line(&mut out, &json!({"done": true}));
    }
}

fn write_line(out: &mut impl Write, value: &Value) {
    if serde_json::to_writer(&mut *out, value).is_err() {
        fail("cannot write to stdout");
    }
    if out.write_all(b"\n").and_then(|_| out.flush()).is_err() {
        fail("cannot write to stdout");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("worker: {message}");
    process::exit(1);
}
