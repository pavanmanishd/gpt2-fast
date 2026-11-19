# gpt2-fast

GPT-2 inference written from scratch in Rust, with the standard library and `serde_json` as its only dependencies. It was made fast one measured step at a time, and it runs in the browser on WebAssembly and WebGPU.

- **Paper:** [gpt2-fast: Dependency-Free GPT-2 Inference on Commodity CPUs and in the Web Browser](https://pavanmanishd.github.io/posts/gpt2-fast/)
- **Live demo:** [pavanmanishd.github.io/gpt2-fast](https://pavanmanishd.github.io/gpt2-fast/). The model runs in your tab, and no text leaves the browser.

## Results

GPT-2 small (124M), 128 greedy tokens per prompt, on 4 cores and 8 GB of an Apple M4 Pro in a Linux container. Each figure is the median of 5 session medians, and each session is bracketed by control runs.

| Weights | Engine | Decode (tok/s) | Prefill (tok/s) | First token (ms) | Peak memory (MB) |
|---|---|---:|---:|---:|---:|
| 8-bit | **gpt2-fast** | **942.8** | **5,322.9** | **11.8** | **183.5** |
| 8-bit | llama.cpp q8_0 | 676.3 | 4,222.5 | 14.8 | 281.4 |
| 8-bit | ONNX Runtime int8 | 463.3 | 4,419.2 | 14.5 | 365.6 |
| 8-bit | CTranslate2 int8 (2 threads) | 163.2 | 2,490.5 | 25.7 | 562.6 |
| 16-bit | **gpt2-fast** | **629.5** | 1,691.1 | 37.6 | 295.5 |
| 16-bit | llama.cpp f16 | 391.0 | 840.2 | 75.7 | 308.7 |
| 32-bit | **gpt2-fast** | **338.2** | 1,633.1 | 39.2 | 530.4 |
| 32-bit | llama.cpp f32 | 303.1 | 899.3 | 71.1 | 542.9 |
| 32-bit | ONNX Runtime f32 | 264.6 | 1,553.8 | 40.2 | 1,016.9 |
| 32-bit | CTranslate2 f32 (2 sessions) | 85.8 | 1,421.1 | 45.0 | 836.4 |

The 32-bit and 16-bit engines reproduce the reference tokens from Hugging Face transformers exactly. The 8-bit engine has a perplexity of 78.01, against 76.84 for the f32 reference.

In the browser, in Chromium on the same Mac, the WebGPU backend decodes 577.3 tok/s and the four-thread WebAssembly build decodes 262.4 tok/s.

## How it got fast

Each step changed one idea and was measured on its own. The paper explains each step and the evidence for it. The notes in [`docs/journal/`](docs/journal/) record what was tried, including the attempts that failed.

| Step | Change | Decode (tok/s) |
|---|---|---:|
| N1 | Naive f32, one thread, recompute every token | 0.123 |
| N2 | Key-value cache | 11.7 |
| N3 | Row-order loops and tiling | 43.2 |
| N4 | Hand-written NEON kernels | 61.3 |
| N5 | Thread pool over 4 cores | 126.0 |
| N6 | No allocations per token, GELU fused into the multiply | 132.6 |
| N7 | 8-bit weights, one scale per 64 inputs | 254.6 |
| N10 | Output-major weight layout, one dot product per output | 306.5 (f32), 410.2 (8-bit) |
| N11 | Cache blocking for prompts | 323.6 (f32) |
| N13 to N24 | 8-bit activations, SDOT and SMMLA, exact outliers in the scoring layer, row scales | 942.8 (8-bit) |
| N25 | 16-bit weights, widened to f32 inside the kernels | 629.5 |

Raw output for every step is in [`results/runs/`](results/runs/). Each measured milestone has one line in [`results/log.jsonl`](results/log.jsonl).

## Quick start

You need stable Rust and Python 3. The browser build also needs `wasm-bindgen-cli` (see [`web/README.md`](web/README.md)).

```sh
python3 reference/download.py gpt2      # weights into models/gpt2 (git-ignored)
cargo build --release

# 8-bit weights for the fast path and the browser
cargo run --release -p gpt2_bench --bin bench -- quantize \
  --model-dir models/gpt2 --out models/gpt2/model-int8.safetensors

# perplexity of the final 8-bit configuration
cargo run --release -p gpt2_bench --bin bench -- perplexity \
  --model-dir models/gpt2 --text reference/perplexity.txt \
  --expected-perplexity reference/perplexity.json \
  --weights int8 --activations int8 --weight-group row --activation-group 128
```

In the Linux benchmark container this prints 78.01. A macOS build of the same commit prints 77.86. The two platforms' math libraries probably account for the gap.

To run the model in the browser locally:

```sh
web/build.sh && web/build.sh --threads   # threads need the nightly pinned in web/rust-toolchain.toml
python3 web/serve.py --port 8000          # sends the COOP/COEP headers threads need
```

Then open <http://localhost:8000/web/>.

## Reproducing the benchmark

`bench/run.sh` builds the container image and runs the protocol in it, pinned to 4 cores and 8 GB. Every session checks its output against `reference/expected.json`.

```sh
bench/run.sh my-f32  5 -- worker --model-dir models/gpt2
bench/run.sh my-f16  5 -- worker --model-dir models/gpt2 --weights f16
bench/run.sh my-int8 5 --quantized -- worker --model-dir models/gpt2 \
  --weights int8 --activations int8 --weight-group row --activation-group 128
```

[`bench/README.md`](bench/README.md) shows how to build and run llama.cpp, ONNX Runtime and CTranslate2 under the same protocol. [`SPEC.md`](SPEC.md) defines the protocol and the results format.

## Layout

```
crates/engine/   the inference engine: std + serde_json only
crates/bench/    benchmark CLI, worker, quantizer, perplexity
crates/web/      wasm-bindgen wrapper for the browser
web/             browser worker, WebGPU backend (gpu/), local site and benchmark pages
bench/           Dockerfile, prompts, run scripts, rival-engine workers
reference/       PyTorch and NumPy references, expected outputs, perplexity text
results/         log.jsonl, raw session output, flame graphs, charts
docs/journal/    one note per step: what changed, what failed, why it got faster
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option. The GPT-2 weights are not part of this repository. `reference/download.py` fetches them from Hugging Face under their own license.
