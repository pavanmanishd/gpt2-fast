# web

The browser build: the same engine, compiled to WebAssembly and run in a Web
Worker, with a WebGPU backend that runs the forward pass in compute shaders.
Weights are downloaded once and kept in the browser's Cache API.

## Website

`index.html` is the public site. It uses plain HTML, CSS and ES modules: no
framework, bundler, external JavaScript, web fonts or analytics.

| File | Purpose |
|---|---|
| `index.html` | The site: hero, live demo, device benchmark, the milestone story, the measurement notes, and the footer. |
| `site.css` | Site styles, including the light and dark palettes and the narrow-screen table layout. |
| `site.js` | Site behaviour: worker lifecycle, engine choice, streaming meter, device benchmark, charts and story rendering. |
| `data/milestones.json` | Generated milestone data. Do not edit by hand. |
| `dev.html` | The old development page, driven by `main.js`. |
| `worker.js` | Shared Web Worker: loads weights, encodes text, generates and streams tokens. Used by the site, the dev page and the benchmark. |
| `gpu/engine.js` | The WebGPU backend: buffers, pipelines, weight upload, decode loop and readback. |
| `gpu/shaders.js` | WGSL compute shaders for embeddings, layer norm, linear layers, attention, residual adds, the tied head and argmax. |
| `thread.js` | The pool worker for the threaded build: joins the engine's shared memory and runs its worker loop. |
| `bench.html`, `bench.js` | The browser benchmark page. |
| `build_site.py` | Assembles the deployable static site in `web/dist`, including the split weights and `_headers`. |
| `serve.py` | Development server that sends the COOP/COEP headers `SharedArrayBuffer` requires. |
| `sync_data.py` | Builds `data/milestones.json` from the results log, prompts and reference output. |

The site never downloads the model on page load. On a first visit it shows a
load button; after that the tokenizer files and the weight parts are in the
Cache API, keyed by their content-hashed names, and later visits load them from
cache without showing the button.

### Engine choice

The **Engine** control has three settings:

- **Auto** asks the worker for a WebGPU adapter. If the worker gets a real
  adapter it runs the WebGPU engine; otherwise it runs the CPU engine with
  WebAssembly threads when the page is cross-origin isolated, or SIMD when it
  is not. A browser without WebAssembly SIMD gets the scalar package.
- **WebGPU** requests the compute-shader backend. If no adapter or no usable
  device is available, the worker falls back to the CPU engine and the page
  says why.
- **CPU** uses the WebAssembly engine directly.

The line under the demo always shows what actually ran, for example
`WebGPU · apple metal-3`, `WebAssembly · 8 threads · SIMD` or
`WebAssembly · SIMD · 1 thread`, next to the weights format and size.

Changing the engine starts a fresh worker, but the weights stay in the Cache
API: the worker reads them from cache, so a switch does not download them again.
The weights are `models/gpt2/model-int8.safetensors`, int8, about 135 MB in
development and split into `model/*.part` files in the built site.

The device benchmark runs prompt `p064` for 64 new tokens with one warm-up and
one timed run on each engine the device supports: CPU first, then WebGPU if the
worker gets an adapter. It shows decode speed, prompt speed and time to first
token next to the project's own browser measurements for the same engine from
`data/milestones.json`. Results stay in the tab; nothing is uploaded.

### Story sections

The story is rendered entirely from `data/milestones.json`:

- **Native engine**: R0 references and the N1 to N11 steps for GPT-2 124M, with
  a log-scale decode chart and one card per milestone. `-threadsN` lines are
  shown in their parent's card. A line may carry `baseline`; when it does, the
  card compares against that milestone instead of the previous card. int8 lines
  are hatched and labelled, with token agreement and perplexity when recorded.
- **The race**: the original N9 llama.cpp race against N6/N7, and the N12
  rematch against N11/N11-int8 when those lines exist.
- **Bigger models**: GPT-2 124M, 355M, 774M and 1.5B. Later N11 rows replace the
  N8 gpt2-fast rows for the same model. A line with `status: "oom"` is marked
  `did not fit in 8 GB`.
- **In the browser**: the W1 to W4 headline runs and their variants, with a
  small chart and cards. These are browser numbers from the development laptop,
  not container numbers.

Sections with no data in the log are not rendered.

## Refreshing the site data

After appending a milestone to `results/log.jsonl`:

```sh
python3 web/sync_data.py
```

The script uses only the Python standard library:

```sh
python3 web/sync_data.py --log results/log.jsonl --out web/data/milestones.json
```

It reads `results/log.jsonl`, `bench/prompts.json` and
`reference/expected.json`, and writes one object with:

- `protocol`: host, cores, memory, sessions, runs per prompt, checked token
  count, the near-tie margin and the closest margin in the prompt set.
- `benchmark`: the `p064` prompt text and the 64-token device run parameters.
- `weights`: the int8 file name, size and format.
- `native`: the R0/N milestone cards in log order, including `baseline`,
  `scaling` and perplexity fields when present.
- `race`: race tables with the engines, decode, prompt speed, first token and
  tokens matching the reference.
- `sizes`: the four GPT-2 sizes with decode, prompt speed, peak memory,
  perplexity and OOM status.
- `web`: W1 to W4 with their variants, browser, target, weights, threads and
  match fields.

For local work, `?data=` selects another data file. It must be a same-origin
relative path; anything else falls back to `./data/milestones.json`.

```sh
python3 web/serve.py --port 8000
# http://localhost:8000/web/?data=./data/dev-milestones.json
```

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

## Deploying

Build the public site into `web/dist`:

```sh
python3 web/build_site.py
python3 web/build_site.py --out /tmp/site --model-dir models/gpt2 --part-mb 20
```

`web/dist` holds `index.html`, `site.js`, `site.css`, `worker.js`, `thread.js`,
`gpu/`, `pkg/`, `pkg-nosimd/`, `pkg-threads/`, a freshly generated
`data/milestones.json`, and `model/`: `config.json`, `vocab.json`, `merges.txt`,
`manifest.json` and the int8 weights cut into content-hashed `.part` files.
Every part is at most `--part-mb` MiB (20 by default), under Cloudflare Pages'
25 MB file limit. The weights are not in git, so build where
`models/gpt2/model-int8.safetensors` exists. `_headers` carries the COOP/COEP
headers for cross-origin isolation and marks the hashed parts immutable. The
worker keeps the manifest in the Cache API and rechecks it after five minutes,
so a new build reaches visitors without ever mixing old and new parts.

Hosting cases:

- **A host that applies `_headers`** (Cloudflare Pages, Netlify): upload the
  contents of `web/dist`. COOP/COEP turn on cross-origin isolation, so the
  threaded WebAssembly build works next to WebGPU.
- **A host without custom headers** (GitHub Pages, plain object storage):
  upload `web/dist` as is; `_headers` is ignored. The page still runs on WebGPU
  or on single-threaded SIMD WebAssembly; WebAssembly threads are unavailable
  because `SharedArrayBuffer` needs cross-origin isolation.
- **Weights on a different origin**: set the `gpt2-fast-model-base` meta tag in
  `web/index.html` to that origin, or pass `--model-base
  https://cdn.example.com/gpt2/` to the build. The worker fetches
  `manifest.json` and the parts with CORS, so the other origin must send
  `Access-Control-Allow-Origin` for the site; a CORS response also satisfies
  cross-origin isolation on the page. The rest of the site stays on its own
  origin.

Check the built site locally with the headers a real host would send:

```sh
python3 web/serve.py --root web/dist --port 8000
```

Nothing is published by these commands. The site goes live only when the owner
decides to deploy `web/dist`.

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

The worker uses `?backend=webgpu` to run the forward pass with WGSL compute
shaders instead of WebAssembly. The site's Engine control sets it; the page
falls back to the CPU engine and reports the reason when no usable adapter is
available. The worker still uses the WebAssembly engine to parse the
safetensors file and to tokenize; the weights are uploaded to storage buffers
once. Every layer runs on the GPU: the KV cache, activations and the chosen
token id stay there, and only the token id is copied back after each step (the
full logits are copied only for a `logits` request).

`wte` is 154 MB at f32 and can exceed the adapter's default 128 MB storage
binding limit, so the adapter's limits are requested and the table is split
across two bindings when needed; `maxBufferSize` is checked as well. The decode
loop submits up to `pipeline=4` steps before it waits for the first token, so
the argmax pass can feed the next step's embedding without a round trip. Use
`pipeline=1` for the simple wait-after-each-step version.

If `navigator.gpu` is missing, no adapter is returned, the adapter is a
software fallback, or the limits are too small, the worker builds the usual
CPU engine and the ready message reports `backend: "cpu"` with a `fallback`
reason. The site then shows the CPU engine and a plain explanation in place of
the requested WebGPU run.

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
