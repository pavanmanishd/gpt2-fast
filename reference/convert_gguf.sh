#!/usr/bin/env bash
set -euo pipefail

name=${1:-gpt2}
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
model_dir=$root/models/$name
converter=/opt/llama.cpp/convert_hf_to_gguf.py

if [[ ! -f $converter ]]; then
    echo "convert_gguf: $converter is missing; run this inside the bench image" >&2
    exit 1
fi
if [[ ! -f $model_dir/config.json ]]; then
    echo "convert_gguf: no HF model in $model_dir" >&2
    exit 1
fi

# the converter indexes every *.safetensors in the directory, including derived
# files such as model-int8.safetensors, and its GPT-2 path rejects the HF
# h.N.attn.bias buffers, so stage a checkpoint without either
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
for file in config.json tokenizer.json vocab.json merges.txt; do
    if [[ -e $model_dir/$file ]]; then
        ln -s "$model_dir/$file" "$stage/$file"
    fi
done
python3 - "$model_dir/model.safetensors" "$stage/model.safetensors" <<'PY'
import sys

from safetensors.torch import load_file, save_file

source, destination = sys.argv[1], sys.argv[2]
tensors = load_file(source)
kept = {
    name: value
    for name, value in tensors.items()
    if not name.endswith((".attn.bias", ".attn.masked_bias"))
}
save_file(kept, destination)
print(f"convert_gguf: dropped {len(tensors) - len(kept)} attention mask buffers", file=sys.stderr)
PY

python3 "$converter" "$stage" --outtype f32 --outfile "$model_dir/$name-f32.gguf"
python3 "$converter" "$stage" --outtype f16 --outfile "$model_dir/$name-f16.gguf"
llama-quantize "$model_dir/$name-f32.gguf" "$model_dir/$name-q8_0.gguf" q8_0
