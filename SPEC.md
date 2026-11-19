# gpt2-fast: spec

## Goal

Write GPT-2 inference from scratch in Rust, make it as fast as possible on a CPU one measured step at a time, then run the same engine in the browser, so anyone who opens the website generates text on their own laptop: no server, no API, no install.

Every step is a public post: what changed, why, the before and after numbers, and a profile showing where the time went. The repo is the source of truth for those posts.

## Model

- GPT-2 from OpenAI's 2019 release, Hugging Face `openai-community/gpt2` (124M parameters) first, then `gpt2-medium` (355M), `gpt2-large` (774M) and `gpt2-xl` (1.5B).
- Weights in safetensors format, downloaded by a script, never committed.
- GPT-3 was never released, so it is out of scope. OpenAI's open-weight `gpt-oss-20b` is a possible final boss after GPT-2 XL, not part of this spec.

## Rules

1. **Correctness before speed.** Greedy decoding (always pick the most likely next token) is deterministic, so every engine version must produce exactly the same tokens as the reference for the standard prompts, and logits within a stated tolerance. A speedup that changes the output does not count.
   - Exception: quantized versions (8-bit or smaller weights) cannot match exactly. They report the share of tokens that match the reference and perplexity on a fixed text, and are shown separately.
2. **From scratch.** The engine crate has no machine-learning or linear-algebra dependencies (no candle, burn, ndarray, BLAS). Allowed: the standard library, `serde_json` for the safetensors header and tokenizer files, and a thread pool once threading is the step being measured. Benchmark and web tooling may use normal dependencies.
3. **One change per step.** Each milestone changes one idea, so the measured difference belongs to that idea.
4. **Explanations are tested.** If a post says why something got faster, there is a measurement or profile in the journal that shows it.

## Benchmark protocol

- **Environment:** a Linux container (`bench/Dockerfile`), limited to 4 cores and 8 GB. CPU only. The host machine is reported in every result.
- **Prompts:** `bench/prompts.json`, five fixed prompts of 16 to 256 tokens.
- **Run:** for each prompt, generate 128 tokens greedily. Warm up once, then time 5 runs.
- **Metrics:**
  - decode speed, tokens per second (the headline number)
  - prefill speed, prompt tokens per second
  - time to first token, ms
  - peak memory, MB
  - whether the output matches the reference
- **Sessions:** 5 sessions per milestone, in separate container runs; report the median of session medians. A laptop scheduler sometimes slows whole sessions, and the median absorbs that.
- **Profiles:** a flame graph per milestone (`perf` or `samply` inside the container), saved as SVG.

## Milestones

### Native track (CPU, Linux container)

| ID | Step | What it shows |
|---|---|---|
| R0 | Reference: Hugging Face transformers (PyTorch, CPU) and a from-scratch NumPy version | The numbers to beat and the outputs to match |
| N1 | Naive Rust: f32, single thread, recompute every token from scratch | Baseline for the engine |
| N2 | KV cache: keep the attention keys and values of past tokens | One idea, usually the biggest single speedup |
| N3 | Matrix multiplication loop order and tiling | Same maths, cache-friendly memory access |
| N4 | SIMD: process several numbers per instruction (NEON on ARM, AVX2 on x86) | Vector instructions |
| N5 | Threads: split matrix work across cores | Scaling from 1 to 4 cores |
| N6 | Fused operations and fewer allocations | Less memory traffic per token |
| N7 | 8-bit weights (int8, per-row scales) | Decode is limited by memory bandwidth; smaller weights mean faster tokens |
| N8 | Bigger models: medium, large, XL | How each optimization holds up as the model grows |
| N9 | Final race: gpt2-fast vs PyTorch vs llama.cpp (GGUF GPT-2) on the same prompts | The headline comparison |

### Web track (in the visitor's browser)

| ID | Step | Notes |
|---|---|---|
| W1 | Compile the engine to WebAssembly, run in a Web Worker | Weights downloaded once and kept in the browser's storage (Cache API or OPFS); f16 or int8 weights to keep the 124M download to 125-250 MB |
| W2 | WebAssembly SIMD (128-bit) | Most browsers support it |
| W3 | WebAssembly threads | Needs SharedArrayBuffer, so the site must send COOP/COEP headers; GitHub Pages can't, Cloudflare Pages or Netlify can |
| W4 | WebGPU compute shaders for the matrix multiplications | Falls back to W3 where WebGPU is missing |
| W5 | The website | A text box, live tokens-per-second meter, the device's own numbers, and the milestone story. Nothing leaves the visitor's device unless they opt in to share their speed |

Browser measurements use the same prompts and metrics, recorded from a fixed set of browsers on the dev machine, and labelled as browser results.

## Repository layout

```
crates/engine/     the inference engine (no ML dependencies)
crates/bench/      CLI that runs the protocol and writes results
crates/web/        wasm-bindgen wrapper for the browser
reference/         Python reference (transformers, NumPy) and expected outputs
bench/             Dockerfile, prompts.json, run scripts
web/               the website
results/log.jsonl  one line per measured milestone (the contract below)
results/runs/      raw session output, flame graphs
docs/journal/      one note per milestone: what changed, what failed, why it got faster
models/            downloaded weights (git-ignored)
```

## Results contract

Every measured milestone appends one JSON object per line to `results/log.jsonl`:

```json
{
  "date": "2026-11-17",
  "milestone": "N2",
  "title": "KV cache",
  "commit": "abc1234",
  "model": "gpt2-124m",
  "target": "cpu-linux",
  "decode_tok_s": 0.0,
  "prefill_tok_s": 0.0,
  "ttft_ms": 0.0,
  "peak_mb": 0.0,
  "matches_reference": true,
  "token_match_rate": 1.0,
  "sessions": 5,
  "host": "Apple M4 Pro, Linux container, 4 cores, 8 GB",
  "journal": "docs/journal/N2-kv-cache.md",
  "flamegraph": "results/runs/N2/flame.svg",
  "headline": "One sentence, true, in plain words, ready to adapt into a post."
}
```

`target` is `cpu-linux`, `wasm`, `wasm-simd`, `wasm-threads` or `webgpu`. Numbers come only from `crates/bench` output saved under `results/runs/`. The journal note explains the change for a reader who is not an ML engineer, including what didn't work.

## Definition of done

- N1 to N9 measured, journaled and matching the reference.
- The website runs GPT-2 124M on a normal laptop browser at a usable speed, with W5's meter showing the visitor's own tokens per second.
- Every post written from this repo cites a line in `results/log.jsonl`.
