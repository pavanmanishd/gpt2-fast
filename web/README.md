# web

The browser build: the same engine, compiled to WebAssembly and run in a Web
Worker. Weights are downloaded once and kept in the browser's Cache API.

## Build

```sh
cargo build -p gpt2_web --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg --out-name gpt2_web \
  target/wasm32-unknown-unknown/release/gpt2_web.wasm
```

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
`model.safetensors` and compare against the reference exactly.
