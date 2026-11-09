#!/usr/bin/env python3
import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

FILES = ("config.json", "vocab.json", "merges.txt", "tokenizer.json", "model.safetensors")
MODELS = ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
BASE_URL = "https://huggingface.co/openai-community/{model}/resolve/main/{file}"


def remote_size(url):
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HEAD {url} failed: {error.code} {error.reason}") from error
    return int(length) if length is not None else None


def download(url, destination, expected_size):
    temporary = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url) as response, open(temporary, "wb") as out:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else expected_size
        done = 0
        next_report = 10
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                percent = 100.0 * done / total
                if percent >= next_report:
                    print(f"  {percent:5.1f}%  {done / 1e6:.1f} MB", file=sys.stderr)
                    next_report = int(percent // 10) * 10 + 10
    size = temporary.stat().st_size
    if expected_size is not None and size != expected_size:
        temporary.unlink()
        raise SystemExit(f"{destination}: got {size} bytes, expected {expected_size}")
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 model files from Hugging Face.")
    parser.add_argument("model", choices=MODELS, help="model name, for example gpt2")
    args = parser.parse_args()

    model_dir = Path("models") / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        url = BASE_URL.format(model=args.model, file=name)
        destination = model_dir / name
        size = remote_size(url)
        if destination.exists() and size is not None and destination.stat().st_size == size:
            print(f"{destination} already present ({size} bytes)")
            continue
        print(f"downloading {url} -> {destination}")
        download(url, destination, size)


if __name__ == "__main__":
    main()
