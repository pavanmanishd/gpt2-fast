use std::io::{self, BufRead, BufWriter, Write};
use std::path::Path;
use std::process;
use std::time::Instant;

use gpt2_engine::Model;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Request {
    ids: Vec<u32>,
    max_new_tokens: usize,
    #[serde(default)]
    return_logits: bool,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-dir" => match args.next() {
                Some(value) => model_dir = Some(value),
                None => fail("--model-dir needs a value"),
            },
            other => fail(&format!("unknown argument `{other}`")),
        }
    }
    let Some(model_dir) = model_dir else {
        fail("usage: worker --model-dir <dir>");
    };

    let started = Instant::now();
    let model = match Model::load(Path::new(&model_dir)) {
        Ok(model) => model,
        Err(error) => fail(&format!("failed to load model: {error}")),
    };
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

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
        if request.ids.is_empty() {
            fail("request has no ids");
        }

        if request.return_logits {
            let logits = model.logits(&request.ids);
            write_line(&mut out, &json!({"logits": logits}));
        }
        model.generate(&request.ids, request.max_new_tokens, |token| {
            write_line(&mut out, &json!({"token": token}));
        });
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
