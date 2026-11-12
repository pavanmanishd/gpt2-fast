#!/usr/bin/env bash
set -euo pipefail

name=${1:-gpt2}
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../../.." && pwd)
model_dir=$root/models/$name
out_dir=$root/models/$name-ct2

if [[ ! -f $model_dir/config.json ]]; then
    echo "convert_ct2: no HF model in $model_dir" >&2
    exit 1
fi

for compute_type in float32 int8; do
    ct2-transformers-converter \
        --model "$model_dir" \
        --output_dir "$out_dir/$compute_type" \
        --quantization "$compute_type" \
        --force
done
