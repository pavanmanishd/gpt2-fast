#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "results" / "log.jsonl"
PROMPTS_PATH = ROOT / "bench" / "prompts.json"
EXPECTED_PATH = ROOT / "reference" / "expected.json"
OUT_PATH = ROOT / "web" / "data" / "milestones.json"

MILESTONE_FIELDS = (
    "milestone",
    "title",
    "date",
    "commit",
    "target",
    "decode_tok_s",
    "prefill_tok_s",
    "ttft_ms",
    "peak_mb",
    "matches_reference",
    "token_match_rate",
    "sessions",
    "headline",
    "journal",
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def model(entry):
    return entry.get("model") or "gpt2-124m"


def main():
    entries = [
        entry for entry in load_jsonl(LOG_PATH)
        if model(entry) == "gpt2-124m"
    ]
    prompts = json.loads(PROMPTS_PATH.read_text())
    expected = json.loads(EXPECTED_PATH.read_text())

    max_new_tokens = prompts["max_new_tokens"]
    prompt_count = len(prompts["prompts"])
    p064 = next(prompt for prompt in prompts["prompts"] if prompt["id"] == "p064")
    closest_gap = min(prompt["min_gap"] for prompt in expected["prompts"])

    host = next((entry["host"] for entry in reversed(entries) if entry.get("host")), "")
    cores_match = re.search(r"(\d+)\s+cores", host)
    memory_match = re.search(r"(\d+)\s*GB", host)
    sessions = max(entry.get("sessions", 0) for entry in entries)

    milestones = []
    for entry in entries:
        milestone = {
            field: entry.get(field) for field in MILESTONE_FIELDS if field in entry
        }
        for key, value in entry.items():
            if key.startswith("perplexity"):
                milestone[key] = value
        milestones.append(milestone)

    payload = {
        "source": "results/log.jsonl",
        "protocol": {
            "host": host,
            "cores": int(cores_match.group(1)) if cores_match else None,
            "memory_gb": int(memory_match.group(1)) if memory_match else None,
            "sessions": sessions,
            "timed_runs": 5,
            "warmups": 1,
            "prompt_count": prompt_count,
            "max_new_tokens": max_new_tokens,
            "tokens_checked": max_new_tokens * prompt_count,
            "near_tie_min": 0.05,
            "closest_gap": round(closest_gap, 3),
        },
        "device_bench": {
            "prompt_id": p064["id"],
            "prompt": p064["text"],
            "max_new_tokens": 64,
            "warmup": 1,
        },
        "milestones": milestones,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({len(milestones)} milestones)")


if __name__ == "__main__":
    main()
