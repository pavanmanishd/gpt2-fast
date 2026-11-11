#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DEFAULT_OUT = WEB / "dist"
DEFAULT_MODEL_DIR = ROOT / "models" / "gpt2"
DEV_MODEL_BASE = "/models/gpt2/"
BUILT_MODEL_BASE = "./model/"
MIB = 1024 * 1024

SITE_FILES = ("index.html", "site.js", "site.css", "worker.js", "thread.js")
SITE_DIRS = {"gpu": ("engine.js", "shaders.js")}
PACKAGE_DIRS = ("pkg", "pkg-nosimd", "pkg-threads")
PACKAGE_FILES = ("gpt2_web.js", "gpt2_web_bg.wasm")
MODEL_FILES = ("config.json", "vocab.json", "merges.txt")
WEIGHTS_FILE = "model-int8.safetensors"

META_PATTERN = re.compile(r'<meta[^>]*\bname="gpt2-fast-model-base"[^>]*>')
CONTENT_PATTERN = re.compile(r'content="([^"]*)"')

HEADERS = """/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp

/*.html
  Cache-Control: public, max-age=300

/*.css
  Cache-Control: public, max-age=300

/*.js
  Cache-Control: public, max-age=300

/*.ts
  Cache-Control: public, max-age=300

/*.json
  Cache-Control: public, max-age=300

/*.txt
  Cache-Control: public, max-age=300

/*.wasm
  Cache-Control: public, max-age=300

/model/*.part
  Cache-Control: public, max-age=31536000, immutable
"""


class BuildError(Exception):
    pass


def copy_file(source, destination):
    if not source.is_file():
        raise BuildError(f"missing {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(source, destination, required):
    if not source.is_dir():
        raise BuildError(f"missing {source}; run web/build.sh first")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise BuildError(
            f"{source} is incomplete (missing {', '.join(missing)}); run web/build.sh"
        )
    shutil.copytree(
        source, destination, ignore=shutil.ignore_patterns(".DS_Store")
    )


def current_model_base(html):
    tag = META_PATTERN.search(html)
    if tag is None:
        return None
    content = CONTENT_PATTERN.search(tag.group(0))
    return content.group(1).strip() if content else None


def index_with_model_base(html, model_base):
    tag = f'<meta name="gpt2-fast-model-base" content="{model_base}" />'
    match = META_PATTERN.search(html)
    if match is None:
        return html.replace("</head>", f"    {tag}\n  </head>", 1)
    return html[: match.start()] + tag + html[match.end() :]


def prepare_out(out):
    root = ROOT.resolve()
    if out == root or out == WEB.resolve() or out in root.parents:
        raise BuildError("--out must not be the repository root, web/, or a parent")
    if out.exists():
        if out.is_symlink() or not out.is_dir():
            raise BuildError(f"{out} exists and is not a directory")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def write_packages(out):
    for name in PACKAGE_DIRS:
        copy_directory(WEB / name, out / name, PACKAGE_FILES)


def write_data(out, weights):
    destination = out / "data" / "milestones.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(WEB / "sync_data.py"),
        "--out",
        str(destination),
        "--weights",
        str(weights),
    ]
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise BuildError(f"web/sync_data.py failed with exit code {error.returncode}")


def write_model(model_dir, model_out, weights, part_bytes):
    model_out.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        copy_file(model_dir / name, model_out / name)

    parts = []
    whole = hashlib.sha256()
    total = 0
    with weights.open("rb") as handle:
        while True:
            block = handle.read(part_bytes)
            if not block:
                break
            whole.update(block)
            digest = hashlib.sha256(block).hexdigest()
            name = f"{weights.stem}.{digest[:16]}.part"
            if any(part["name"] == name for part in parts):
                raise BuildError(f"two weight parts share the content hash in {name}")
            (model_out / name).write_bytes(block)
            parts.append({"name": name, "bytes": len(block), "sha256": digest})
            total += len(block)

    if not parts:
        raise BuildError(f"{weights} is empty")
    if total != weights.stat().st_size:
        raise BuildError(f"{weights} changed size while it was read")

    manifest = {
        "weights": {
            "name": weights.name,
            "bytes": total,
            "sha256": whole.hexdigest(),
            "parts": parts,
        }
    }
    (model_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return len(parts)


def report(out, part_bytes, part_count):
    files = [path for path in out.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    parts = [path for path in (out / "model").glob("*.part")]
    largest = max((path.stat().st_size for path in parts), default=0)
    print(
        f"wrote {out}: {len(files)} files, {total / 1e6:.1f} MB, "
        f"largest file {max(path.stat().st_size for path in files) / 1e6:.1f} MB"
    )
    print(
        f"model: {part_count} parts, largest {largest / MIB:.1f} MiB, "
        f"limit {part_bytes / MIB:.1f} MiB"
    )


def build(out, model_dir, part_bytes, model_base):
    prepare_out(out)

    html = (WEB / "index.html").read_text()
    source_base = current_model_base(html)
    if model_base:
        chosen_base = model_base
    elif source_base is None or source_base == DEV_MODEL_BASE:
        chosen_base = BUILT_MODEL_BASE
    else:
        chosen_base = source_base
    (out / "index.html").write_text(index_with_model_base(html, chosen_base))

    for name in SITE_FILES:
        if name != "index.html":
            copy_file(WEB / name, out / name)
    for name, required in SITE_DIRS.items():
        copy_directory(WEB / name, out / name, required)

    write_packages(out)

    weights = model_dir / WEIGHTS_FILE
    if not weights.is_file():
        raise BuildError(
            f"missing {weights}; quantize the model first (see web/README.md)"
        )
    write_data(out, weights)
    part_count = write_model(model_dir, out / "model", weights, part_bytes)
    (out / "_headers").write_text(HEADERS)

    report(out, part_bytes, part_count)


def main():
    parser = argparse.ArgumentParser(
        description="Assemble the public site in a directory that can be uploaded "
        "to a static host."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (default: web/dist)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="directory with config.json, vocab.json, merges.txt and the int8 "
        "weights (default: models/gpt2)",
    )
    parser.add_argument(
        "--part-mb",
        type=int,
        default=20,
        help="largest weight part in MiB (default: 20)",
    )
    parser.add_argument(
        "--model-base",
        help="value for the gpt2-fast-model-base meta tag in the built "
        "index.html (default: %s, or a custom value already in the file)"
        % BUILT_MODEL_BASE,
    )
    args = parser.parse_args()
    if args.part_mb < 1:
        parser.error("--part-mb must be at least 1")

    out = args.out.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    try:
        build(out, model_dir, args.part_mb * MIB, args.model_base)
    except BuildError as error:
        print(f"build_site: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
