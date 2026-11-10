#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = ROOT / "results" / "log.jsonl"
DEFAULT_OUT = ROOT / "web" / "data" / "milestones.json"
PROMPTS_PATH = ROOT / "bench" / "prompts.json"
EXPECTED_PATH = ROOT / "reference" / "expected.json"
WEIGHTS_PATH = ROOT / "models" / "gpt2" / "model-int8.safetensors"
WEIGHTS_FALLBACK_BYTES = 134901520

DEFAULT_MODEL = "gpt2-124m"
DEFAULT_ENGINE = "gpt2-fast"
DEVICE_MAX_NEW_TOKENS = 64
THREADS_RE = re.compile(r"^(?P<parent>.+)-threads(?P<count>\d+)$")
WEB_RE = re.compile(r"^(?P<parent>W\d+)(?:-(?P<variant>.+))?$")
NATIVE_RE = re.compile(r"^(?:R0(?:-[a-z]+)?|N(?:[1-9]|1[01])(?:-int8)?)$")

MILESTONE_FIELDS = (
    "milestone",
    "title",
    "date",
    "commit",
    "model",
    "target",
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_ms",
    "peak_mb",
    "matches_reference",
    "token_match_rate",
    "sessions",
    "baseline",
    "headline",
    "journal",
    "status",
)

WEB_FIELDS = (
    "milestone",
    "title",
    "date",
    "commit",
    "model",
    "target",
    "weights",
    "browser",
    "threads",
    "baseline",
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_ms",
    "sessions",
    "matches_reference",
    "token_match_rate",
    "matches_native",
    "headline",
    "journal",
)


def display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def engine(entry):
    return entry.get("engine") or DEFAULT_ENGINE


def model_name(entry):
    return entry.get("model") or DEFAULT_MODEL


def is_native(entry):
    return (
        model_name(entry) == DEFAULT_MODEL
        and entry.get("target") == "cpu-linux"
        and engine(entry) == DEFAULT_ENGINE
    )


def is_web(entry):
    return entry.get("target") in {"wasm", "wasm-simd", "wasm-threads", "webgpu"}


def is_int8(entry):
    milestone = str(entry.get("milestone", ""))
    return milestone.endswith("-int8") or milestone == "N7"


def copy_fields(entry, fields):
    return {
        field: entry[field]
        for field in fields
        if field in entry and entry[field] is not None
    }


def copy_perplexity(entry):
    return {
        key: value
        for key, value in entry.items()
        if key.startswith("perplexity") and value is not None
    }


def milestone_entry(entry):
    item = copy_fields(entry, MILESTONE_FIELDS)
    item.update(copy_perplexity(entry))
    item["quantized"] = is_int8(entry)
    return item


def web_entry(entry):
    item = copy_fields(entry, WEB_FIELDS)
    item.update(copy_perplexity(entry))
    item["quantized"] = entry.get("weights") == "int8"
    return item


def native_section(entries):
    main = []
    scaling = {}
    for entry in entries:
        if not is_native(entry):
            continue
        milestone = str(entry.get("milestone", ""))
        match = THREADS_RE.match(milestone)
        if match:
            scaling.setdefault(match.group("parent"), []).append(entry)
        elif NATIVE_RE.match(milestone):
            main.append(entry)

    section = []
    for entry in main:
        item = milestone_entry(entry)
        rows = scaling.get(str(entry["milestone"]), [])
        if rows:
            item["scaling"] = sorted(
                (
                    {
                        "milestone": row["milestone"],
                        "threads": int(
                            THREADS_RE.match(str(row["milestone"])).group("count")
                        ),
                        "decode_tok_s": row.get("decode_tok_s"),
                        "prefill_tok_s": row.get("prefill_tok_s"),
                        "ttft_ms": row.get("ttft_ms"),
                    }
                    for row in rows
                ),
                key=lambda row: row["threads"],
            )
        section.append(item)
    return section


def race_row(entry, label, tokens_total):
    rate = entry.get("token_match_rate")
    matched = round(rate * tokens_total) if rate is not None else None
    return {
        "label": label,
        "milestone": entry["milestone"],
        "decode_tok_s": entry.get("decode_tok_s"),
        "prefill_tok_s": entry.get("prefill_tok_s"),
        "ttft_ms": entry.get("ttft_ms"),
        "token_match_rate": rate,
        "matched_tokens": matched,
        "tokens_total": tokens_total,
        "quantized": is_int8(entry) or "q8" in str(entry["milestone"]),
    }


def race_table(by_milestone, ident, label, note, f32_id, int8_id, prefix, total):
    llama = [
        (f"{prefix}-f32", "llama.cpp f32"),
        (f"{prefix}-f16", "llama.cpp f16"),
        (f"{prefix}-q8_0", "llama.cpp q8_0"),
    ]
    if not any(milestone in by_milestone for milestone, _ in llama):
        return None

    rows = []
    for milestone, caption in (
        ("R0-torch", "PyTorch"),
        ("R0-numpy", "NumPy"),
        (f32_id, "gpt2-fast f32"),
        (int8_id, "gpt2-fast int8"),
    ):
        entry = by_milestone.get(milestone)
        if entry is not None:
            rows.append(race_row(entry, caption, total))
    for milestone, caption in llama:
        entry = by_milestone.get(milestone)
        if entry is not None:
            rows.append(race_row(entry, caption, total))

    return {"id": ident, "label": label, "note": note, "rows": rows}


def race_section(entries, tokens_total):
    by_milestone = {
        entry["milestone"]: entry
        for entry in entries
        if model_name(entry) == DEFAULT_MODEL
        and entry.get("target") == "cpu-linux"
    }
    tables = [
        race_table(
            by_milestone,
            "N9",
            "Original race",
            "N6 f32 and N7 int8 against llama.cpp on the same four cores.",
            "N6",
            "N7",
            "N9-llamacpp",
            tokens_total,
        ),
        race_table(
            by_milestone,
            "N12",
            "Rematch",
            "N11 f32 and N11-int8 against llama.cpp. PyTorch and NumPy are the "
            "R0 references, not re-run for this race.",
            "N11",
            "N11-int8",
            "N12-llamacpp",
            tokens_total,
        ),
    ]
    return [table for table in tables if table is not None]


SIZE_GROUPS = (
    ("124M", ("R0-torch",), ("N6",), ("N7",)),
    (
        "355M (medium)",
        ("N8-gpt2-medium-torch",),
        ("N11-gpt2-medium-f32", "N8-gpt2-medium-f32"),
        ("N11-gpt2-medium-int8", "N8-gpt2-medium-int8"),
    ),
    (
        "774M (large)",
        ("N8-gpt2-large-torch",),
        ("N11-gpt2-large-f32", "N8-gpt2-large-f32"),
        ("N11-gpt2-large-int8", "N8-gpt2-large-int8"),
    ),
    (
        "1.5B (xl)",
        ("N8-gpt2-xl-torch",),
        ("N11-gpt2-xl-f32", "N8-gpt2-xl-f32"),
        ("N11-gpt2-xl-int8", "N8-gpt2-xl-int8"),
    ),
)


def pick(by_milestone, candidates):
    for milestone in candidates:
        entry = by_milestone.get(milestone)
        if entry is not None:
            return entry
    return None


def size_row(entry, label):
    row = {
        "label": label,
        "milestone": entry["milestone"],
        "decode_tok_s": entry.get("decode_tok_s"),
        "prefill_tok_s": entry.get("prefill_tok_s"),
        "peak_mb": entry.get("peak_mb"),
        "matches_reference": entry.get("matches_reference"),
        "token_match_rate": entry.get("token_match_rate"),
    }
    if entry.get("status") is not None:
        row["status"] = entry["status"]
    if entry.get("perplexity") is not None:
        row["perplexity"] = entry["perplexity"]
    return row


def sizes_section(entries):
    by_milestone = {entry["milestone"]: entry for entry in entries}
    groups = []
    for label, torch_ids, f32_ids, int8_ids in SIZE_GROUPS:
        rows = []
        torch_entry = pick(by_milestone, torch_ids)
        f32_entry = pick(by_milestone, f32_ids)
        int8_entry = pick(by_milestone, int8_ids)
        if torch_entry is not None:
            rows.append(size_row(torch_entry, "PyTorch"))
        if f32_entry is not None:
            rows.append(size_row(f32_entry, "gpt2-fast f32"))
        if int8_entry is not None:
            rows.append(size_row(int8_entry, "gpt2-fast int8"))
        if rows:
            groups.append({"label": label, "rows": rows})
    return groups


def web_section(entries):
    main = {}
    variants = {}
    for entry in entries:
        if not is_web(entry):
            continue
        match = WEB_RE.match(str(entry.get("milestone", "")))
        if match is None:
            continue
        parent = match.group("parent")
        variant = match.group("variant")
        if variant is None:
            main[parent] = entry
        else:
            variants.setdefault(parent, []).append(entry)

    section = []
    for parent in sorted(main, key=lambda name: int(name[1:])):
        item = web_entry(main[parent])
        item["variants"] = [web_entry(entry) for entry in variants.get(parent, [])]
        section.append(item)
    return section


def read_protocol(entries, prompts, expected):
    host = next(
        (
            entry["host"]
            for entry in reversed(entries)
            if entry.get("target") == "cpu-linux" and entry.get("host")
        ),
        "",
    )
    cores = re.search(r"(\d+)\s+cores", host)
    memory = re.search(r"(\d+)\s*GB", host)
    sessions = max(
        (
            entry.get("sessions", 0)
            for entry in entries
            if entry.get("target") == "cpu-linux"
        ),
        default=0,
    )
    browser_host = next(
        (
            entry["host"]
            for entry in reversed(entries)
            if is_web(entry) and entry.get("host")
        ),
        "",
    )
    return {
        "host": host,
        "browser_host": browser_host,
        "cores": int(cores.group(1)) if cores else None,
        "memory_gb": int(memory.group(1)) if memory else None,
        "sessions": sessions,
        "timed_runs": 5,
        "warmups": 1,
        "prompt_count": len(prompts["prompts"]),
        "max_new_tokens": prompts["max_new_tokens"],
        "tokens_checked": prompts["max_new_tokens"] * len(prompts["prompts"]),
        "near_tie_min": 0.05,
        "closest_gap": round(
            min(float(prompt["min_gap"]) for prompt in expected["prompts"]), 3
        ),
    }


def weights_bytes():
    try:
        return WEIGHTS_PATH.stat().st_size
    except OSError:
        return WEIGHTS_FALLBACK_BYTES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build web/data/milestones.json from the results log."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="results log to read (default: results/log.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="data file to write (default: web/data/milestones.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    entries = load_jsonl(args.log)
    prompts = json.loads(PROMPTS_PATH.read_text())
    expected = json.loads(EXPECTED_PATH.read_text())
    prompt = next(prompt for prompt in prompts["prompts"] if prompt["id"] == "p064")

    protocol = read_protocol(entries, prompts, expected)
    web = web_section(entries)
    payload = {
        "source": display_path(args.log),
        "protocol": protocol,
        "benchmark": {
            "prompt_id": prompt["id"],
            "prompt": prompt["text"],
            "max_new_tokens": DEVICE_MAX_NEW_TOKENS,
            "warmup": 1,
            "runs": 1,
        },
        "weights": {
            "format": "int8",
            "file": WEIGHTS_PATH.name,
            "bytes": weights_bytes(),
        },
        "native": native_section(entries),
        "race": race_section(entries, protocol["tokens_checked"]),
        "sizes": sizes_section(entries),
        "web": web,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {display_path(args.out)} "
        f"({len(payload['native'])} native, {len(payload['race'])} race, "
        f"{len(payload['sizes'])} sizes, {len(payload['web'])} web)"
    )


if __name__ == "__main__":
    main()
