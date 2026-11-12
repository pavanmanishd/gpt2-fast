#!/usr/bin/env bash
set -euo pipefail

name=${1:-gpt2}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
model_dir=$root/models/$name
out_dir=$root/models/$name-onnx

if [[ ! -f $model_dir/config.json ]]; then
    echo "convert_onnx: no HF model in $model_dir" >&2
    exit 1
fi

mkdir -p "$out_dir"
python3 "$here/export.py" --model-dir "$model_dir" --out-dir "$out_dir"
