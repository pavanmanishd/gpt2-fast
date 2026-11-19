# Contributing

Read [SPEC.md](SPEC.md) first. It defines the benchmark protocol and the results format.

## Rules

- **Correctness first.** Every engine change must match the reference output for `bench/prompts.json`: exact tokens for f32 and f16, token match rate and perplexity for 8-bit. A speedup that fails this does not count.
- **No ML or linear-algebra dependencies in the engine.** `crates/engine` uses the standard library and `serde_json` only. Benchmark and web tooling may use normal dependencies.
- **One idea per change.** Each change does one thing, so the measured difference belongs to that idea.
- **Numbers come from the benchmark.** Report numbers only from `crates/bench` runs in the `bench/` container: 5 sessions, median of session medians. Save raw output under `results/runs/<label>/`.
- **Explanations are tested.** If you say why something got faster, include the profile or measurement that shows it. If a cause was not measured, say "probably".
- **Never commit model weights.** `models/` is git-ignored.

## Checks

```sh
cargo test --release
bench/run.sh <label> 5 -- worker --model-dir models/gpt2
```

The second command fails if any prompt's output differs from `reference/expected.json`.
