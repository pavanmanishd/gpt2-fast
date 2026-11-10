#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: bench/profile.sh <label> <prompt-id> <max_new_tokens> [--expected <file>] [--quantized] -- <worker command...>" >&2
    exit 1
}

[[ $# -ge 5 ]] || usage
label=$1
prompt_id=$2
max_new_tokens=$3
shift 3

expected=reference/expected.json
quantized=
while [[ $# -gt 0 ]]; do
    case $1 in
        --expected)
            [[ $# -ge 2 ]] || usage
            expected=$2
            shift 2
            ;;
        --quantized)
            quantized=--quantized
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            usage
            ;;
    esac
done
[[ $# -gt 0 ]] || usage
worker=("$@")

[[ $max_new_tokens =~ ^[0-9]+$ ]] && (( max_new_tokens >= 1 )) || {
    echo "max_new_tokens must be a positive integer" >&2
    exit 1
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -n "$(git status --porcelain -- . ':(exclude)results')" ]]; then
    if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
        echo "working tree is dirty; commit first or set ALLOW_DIRTY=1" >&2
        exit 1
    fi
    commit="$(git rev-parse --short HEAD)-dirty"
else
    commit="$(git rev-parse --short HEAD)"
fi
image="gpt2-fast-bench:${commit}"

docker build --platform linux/arm64 -t "$image" -f bench/Dockerfile .
mkdir -p "results/runs/$label"

inner="perf record -e cpu-clock -F 999 -g --call-graph dwarf -o /tmp/perf.data -- bench run --label $(printf '%q' "$label") --prompts bench/prompts.json --expected $(printf '%q' "$expected") --only $(printf '%q' "$prompt_id") --max-new-tokens $max_new_tokens --warmup 0 --runs 1 $quantized --out $(printf '%q' "results/runs/$label/profile.json") --"
for argument in "${worker[@]}"; do
    inner+=" $(printf '%q' "$argument")"
done
inner+=" && perf script -i /tmp/perf.data > /tmp/perf.script"
inner+=" && inferno-collapse-perf < /tmp/perf.script > $(printf '%q' "results/runs/$label/flame.folded")"
inner+=" && inferno-flamegraph < $(printf '%q' "results/runs/$label/flame.folded") > $(printf '%q' "results/runs/$label/flame.svg")"

docker run --rm --privileged --platform linux/arm64 --cpuset-cpus=0-3 --memory=8g \
    -e "GPT2_FAST_COMMIT=$commit" \
    -v "$PWD/models:/work/models:ro" \
    -v "$PWD/results/runs:/work/results/runs" \
    "$image" bash -c "$inner"
