# web

The browser build: the same engine, compiled to WebAssembly and run in a Web
Worker. Weights are downloaded once and kept in the browser's Cache API.

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
| `bench.html`, `bench.js` | The browser benchmark page. |
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
```

The SIMD build compiles with `-C target-feature=+simd128` and uses 128-bit
kernels for the matrix multiplications, dot products and the attention weighted
sum. The scalar fallback runs where SIMD is unavailable, and is also the
reference for checking the SIMD build.

`wasm-bindgen-cli` must match the `wasm-bindgen` crates.io release pinned in
`crates/web/Cargo.toml` (0.2.130).

## Quantized weights

The page loads `models/gpt2/model-int8.safetensors`, which is not committed:

```sh
cargo run --release -p gpt2_bench --bin bench -- quantize \
  --model-dir models/gpt2 --out models/gpt2/model-int8.safetensors
```

## Run

Serve the repository root so `models/gpt2/` is reachable:

```sh
python3 -m http.server 8000
```

Open <http://localhost:8000/web/> for the site. The dev page is
<http://localhost:8000/web/dev.html>. The benchmark page is
<http://localhost:8000/web/bench.html>; add `?weights=f32` to load
`model.safetensors` and compare against the reference exactly, and
`?pkg=pkg-nosimd` to load the scalar build.

## Benchmark sessions

`web/run_bench.py` serves the repository, opens a fresh browser context for
each session, waits for the page to finish and writes the session JSON to
`results/runs/<label>/`:

```sh
uv run web/run_bench.py --label W2-chromium --browser chromium \
  --sessions 5 --weights int8 --pkg pkg
```

Install the browsers once with:

```sh
uv run --with playwright==1.48.0 playwright install chromium firefox webkit
```

`bench summarize results/runs/<label>` reads the written sessions.
