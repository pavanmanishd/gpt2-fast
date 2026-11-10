#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="pkg"
features="+simd128"

case "${1:-}" in
  "") ;;
  --no-simd)
    out="pkg-nosimd"
    features="-simd128"
    ;;
  *)
    echo "usage: web/build.sh [--no-simd]" >&2
    exit 2
    ;;
esac

cd "$root"
target_dir="$(
  cargo metadata --format-version 1 --no-deps \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])'
)"
RUSTFLAGS="-C target-feature=${features}" \
  cargo build -p gpt2_web --release --locked --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir "web/${out}" --out-name gpt2_web \
  "${target_dir}/wasm32-unknown-unknown/release/gpt2_web.wasm"
echo "built web/${out}"
