# web

The browser build: the same engine, compiled to WebAssembly and run in a Web
Worker. Weights are downloaded once and kept in the browser's Cache API.

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

Open <http://localhost:8000/web/>. The benchmark page is
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
