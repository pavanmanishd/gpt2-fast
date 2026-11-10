#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="pkg"
features="+simd128"
rustflags=""

case "${1:-}" in
  "") ;;
  --no-simd)
    out="pkg-nosimd"
    features="-simd128"
    ;;
  --threads)
    out="pkg-threads"
    features="+simd128,+atomics,+bulk-memory,+mutable-globals"
    rustflags="-C link-arg=--import-memory \
-C link-arg=--shared-memory \
-C link-arg=--max-memory=2147483648 \
-C link-arg=--export=__wasm_init_tls \
-C link-arg=--export=__tls_base \
-C link-arg=--export=__tls_size \
-C link-arg=--export=__tls_align"
    ;;
  *)
    echo "usage: web/build.sh [--no-simd|--threads]" >&2
    exit 2
    ;;
esac

target_dir="$(
  cd "$root"
  cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
)"

if [[ "$out" == "pkg-threads" ]]; then
  # web/rust-toolchain.toml supplies the nightly compiler, rust-src and the
  # wasm target that build-std needs.
  cd "$root/web"
  RUSTFLAGS="-C target-feature=${features} ${rustflags}" \
    cargo build --manifest-path "$root/Cargo.toml" -p gpt2_web --release --locked \
      --target wasm32-unknown-unknown -Z build-std=panic_abort,std
else
  cd "$root"
  RUSTFLAGS="-C target-feature=${features}" \
    cargo build -p gpt2_web --release --locked --target wasm32-unknown-unknown
fi

wasm-bindgen --target web --out-dir "$root/web/${out}" --out-name gpt2_web \
  "${target_dir}/wasm32-unknown-unknown/release/gpt2_web.wasm"
echo "built web/${out}"
