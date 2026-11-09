#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: bench/run.sh <label> <sessions> [--warmup N] [--runs N] [--expected <file>] [--quantized] -- <worker command...>" >&2
    exit 1
}

[[ $# -ge 4 ]] || usage
label=$1
sessions=$2
shift 2

warmup=1
runs=5
expected=reference/expected.json
quantized=
while [[ $# -gt 0 ]]; do
    case $1 in
        --warmup)
            [[ $# -ge 2 ]] || usage
            warmup=$2
            shift 2
            ;;
        --runs)
            [[ $# -ge 2 ]] || usage
            runs=$2
            shift 2
            ;;
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

[[ $sessions =~ ^[0-9]+$ ]] && (( sessions >= 1 )) || {
    echo "sessions must be a positive integer" >&2
    exit 1
}
[[ $warmup =~ ^[0-9]+$ ]] || {
    echo "--warmup must be a non-negative integer" >&2
    exit 1
}
[[ $runs =~ ^[0-9]+$ ]] && (( runs >= 1 )) || {
    echo "--runs must be a positive integer" >&2
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

for session in $(seq 1 "$sessions"); do
    docker run --rm --platform linux/arm64 --cpuset-cpus=0-3 --memory=8g \
        -e "GPT2_FAST_COMMIT=$commit" \
        -v "$PWD/models:/work/models:ro" \
        -v "$PWD/results/runs:/work/results/runs" \
        "$image" \
        bench run --label "$label" --prompts bench/prompts.json --expected "$expected" \
        --warmup "$warmup" --runs "$runs" $quantized \
        --out "results/runs/$label/session-$session.json" \
        -- "${worker[@]}"
done

docker run --rm --platform linux/arm64 --cpuset-cpus=0-3 --memory=8g \
    -e "GPT2_FAST_COMMIT=$commit" \
    -v "$PWD/models:/work/models:ro" \
    -v "$PWD/results/runs:/work/results/runs" \
    "$image" \
    bench summarize "results/runs/$label" 2>&1 | tee "results/runs/$label/run.log"
