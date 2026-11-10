# web

The browser build: the same engine, compiled to WebAssembly and run in a Web
Worker, with an optional WebGPU backend that runs the forward pass in compute
shaders. Weights are downloaded once and kept in the browser's Cache API.

## Website

`index.html` is the public site. It uses plain HTML, CSS and ES modules: no
framework, bundler, external JavaScript, web fonts or analytics.

| File | Purpose |
|---|---|
| `index.html` | The site: hero and live demo, the device benchmark, the milestone chart and cards, the measurement notes, and the footer. |
| `site.css` | Site styles, including the light and dark palettes. |
| `site.js` | Site behaviour: worker lifecycle, streaming meter, device benchmark, chart and story rendering. |
| `data/milestones.json` | Generated milestone data. Do not edit by hand. |
| `dev.html` | The old development page, driven by `main.js`. |
| `worker.js` | Shared Web Worker: loads weights, encodes text, generates and streams tokens. Used by the site, the dev page and the benchmark. |
| `gpu/engine.js` | The WebGPU backend: buffers, pipelines, weight upload, decode loop and readback. |
| `gpu/shaders.js` | WGSL compute shaders for embeddings, layer norm, linear layers, attention, residual adds, the tied head and argmax. |
| `thread.js` | The pool worker for the threaded build: joins the engine's shared memory and runs its worker loop. |
| `bench.html`, `bench.js` | The browser benchmark page. |
| `serve.py` | Development server that sends the COOP/COEP headers `SharedArrayBuffer` requires. |
| `sync_data.py` | Builds `data/milestones.json` from the repository. |

The site never downloads the model on page load. On a first visit it shows a
load button; after that the four model files are in the Cache API and later
visits load them from cache without showing the button. The device section runs prompt
`p064` for 64 new tokens, one warm-up and one timed run. Nothing from the page is
sent anywhere.

All data comes from the same origin: `data/milestones.json`, the worker, the
WebAssembly package and `/models/gpt2/`. The journal links use the `REPO_URL`
constant at the top of `site.js`.

## Refreshing the site data

After appending a milestone to `results/log.jsonl`:

```sh
python3 web/sync_data.py
```

The script uses only the Python standard library. It reads the `gpt2-124m`
entries in `results/log.jsonl`, `bench/prompts.json` and
`reference/expected.json`, and writes `web/data/milestones.json` in log order.
The output contains:

- `protocol`: host, cores, memory, sessions, runs per prompt, checked token
  count, and the near-tie margin.
- `device_bench`: the `p064` prompt text and the 64-token run parameters.
- `milestones`: the fields the page renders, including perplexity fields when a
  quantized run recorded them.

## Build

`web/build.sh` builds the module and runs `wasm-bindgen`:

```sh
web/build.sh            # WebAssembly SIMD, written to web/pkg
web/build.sh --no-simd  # scalar fallback, written to web/pkg-nosimd
web/build.sh --threads  # SIMD plus WebAssembly threads, written to web/pkg-threads
```

The SIMD build compiles with `-C target-feature=+simd128` and uses 128-bit
kernels for the matrix multiplications, dot products and the attention weighted
sum. The scalar fallback runs where SIMD is unavailable, and is also the
reference for checking the SIMD build.

`wasm-bindgen-cli` must match the `wasm-bindgen` crates.io release pinned in
`crates/web/Cargo.toml` (0.2.130).

The threaded build recompiles the standard library with atomics, which needs
nightly Rust. `web/rust-toolchain.toml` pins `nightly-2026-08-23` and the
`rust-src` and `wasm32-unknown-unknown` components; `build.sh` runs the build
from `web/`, so that file selects the compiler. It links an imported shared
memory and exports the TLS symbols the wasm-bindgen threads transform expects.

## Quantized weights

The page loads `models/gpt2/model-int8.safetensors`, which is not committed:

```sh
cargo run --release -p gpt2_bench --bin bench -- quantize \
  --model-dir models/gpt2 --out models/gpt2/model-int8.safetensors
```

## Run

Serve the repository root so `models/gpt2/` is reachable. `serve.py` sends
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, which turn on cross-origin
isolation and therefore `SharedArrayBuffer`:

```sh
python3 web/serve.py --port 8000
```

Open <http://localhost:8000/web/> for the site. The dev page is
<http://localhost:8000/web/dev.html>. The benchmark page is
<http://localhost:8000/web/bench.html>; add `?weights=f32` to load
`model.safetensors` and compare against the reference exactly, and
`?pkg=pkg-nosimd` to load the scalar build, or `?pkg=pkg-threads&threads=8` for
the threaded build. The site only picks the threaded package when
`crossOriginIsolated` is true and the worker falls back to `pkg` when the
threaded build or shared memory is unavailable, so both pages keep working
under a plain static server without the headers.

## WebGPU

Add `?backend=webgpu` to the worker or benchmark URL to run the forward pass
with WGSL compute shaders instead of WebAssembly. The worker still uses the
WebAssembly engine to parse the safetensors file and to tokenize; the weights
are uploaded to storage buffers once. Every layer runs on the GPU: the KV cache,
activations and the chosen token id stay there, and only the token id is copied
back after each step (the full logits are copied only for a `logits` request).

`wte` is 154 MB at f32 and can exceed the adapter's default 128 MB storage
binding limit, so the adapter's limits are requested and the table is split
across two bindings when needed; `maxBufferSize` is checked as well. The decode
loop submits up to `pipeline=4` steps before it waits for the first token, so
the argmax pass can feed the next step's embedding without a round trip. Use
`pipeline=1` for the simple wait-after-each-step version.

If `navigator.gpu` is missing, no adapter is returned, the adapter is a
software fallback, or the limits are too small, the worker builds the usual
CPU engine and the ready message reports `backend: "cpu"` with a `fallback`
reason. The benchmark session then records that the requested backend was
WebGPU but the CPU one ran.

```sh
uv run web/run_bench.py --label W4-chromium --browser chromium \
  --backend webgpu --weights int8 --sessions 1
uv run web/run_bench.py --label W4-chrome --browser chrome \
  --backend webgpu --weights f32 --sessions 1
```

`--browser chrome` launches the installed Google Chrome with Playwright's
temporary profile; `--pipeline` overrides the number of steps in flight.

## Benchmark sessions

`web/run_bench.py` serves the repository, opens a fresh browser context for
each session, waits for the page to finish and writes the session JSON to
`results/runs/<label>/`:

```sh
uv run web/run_bench.py --label W3-chromium --browser chromium \
  --sessions 1 --weights int8 --pkg pkg-threads --threads 8
```

`--warmup`, `--runs` and `--only` shorten a session while iterating; the
defaults match the benchmark protocol.

Install the browsers once with:

```sh
uv run --with playwright==1.48.0 playwright install chromium firefox webkit
```

`bench summarize results/runs/<label>` reads the written sessions.
