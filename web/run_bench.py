# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "playwright==1.48.0",
# ]
# ///

import argparse
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

from serve import make_server

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_MS = 60 * 60 * 1000


def serve() -> tuple[ThreadingHTTPServer, int]:
    server = make_server(ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the browser benchmark in fresh Playwright sessions."
    )
    parser.add_argument("--label", required=True, help="results/runs subdirectory")
    parser.add_argument(
        "--browser",
        action="append",
        choices=["chromium", "chrome", "firefox", "webkit"],
        help="browser to run, repeatable (default: chromium)",
    )
    parser.add_argument("--sessions", type=positive, default=5)
    parser.add_argument("--backend", choices=["cpu", "webgpu"], default="cpu")
    parser.add_argument("--weights", choices=["f32", "int8"], default="int8")
    parser.add_argument(
        "--pkg", choices=["pkg", "pkg-nosimd", "pkg-threads"], default="pkg"
    )
    parser.add_argument(
        "--threads",
        type=positive,
        help="thread count sent to the page as a query parameter",
    )
    parser.add_argument(
        "--pipeline",
        type=positive,
        help="WebGPU decode steps submitted before the first token is read back",
    )
    parser.add_argument("--warmup", type=non_negative, default=1)
    parser.add_argument("--runs", type=non_negative, default=5)
    parser.add_argument(
        "--only", help="comma-separated prompt ids, for example p064,p256"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="output directory (default: results/runs/<label>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    browsers = args.browser or ["chromium"]
    out_dir = args.out or ROOT / "results" / "runs" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_args = [
        "--headless=new",
        "--enable-unsafe-webgpu",
        "--ignore-gpu-blocklist",
    ]
    if sys.platform == "darwin":
        gpu_args.append("--use-angle=metal")
    else:
        gpu_args.append("--enable-features=Vulkan")

    server, port = serve()
    session_index = 0
    try:
        with sync_playwright() as playwright:
            for name in browsers:
                if name == "chrome":
                    if args.backend == "webgpu":
                        browser = playwright.chromium.launch(
                            headless=False,
                            args=gpu_args,
                            channel="chrome",
                        )
                    else:
                        browser = playwright.chromium.launch(
                            headless=True, channel="chrome"
                        )
                elif name == "chromium" and args.backend == "webgpu":
                    browser = playwright.chromium.launch(headless=False, args=gpu_args)
                else:
                    browser = getattr(playwright, name).launch(headless=True)
                version = browser.version
                try:
                    for _ in range(args.sessions):
                        session_index += 1
                        context = browser.new_context()
                        try:
                            page = context.new_page()
                            query = {
                                "weights": args.weights,
                                "backend": args.backend,
                                "pkg": args.pkg,
                                "browser": name,
                                "browser_version": version,
                                "warmup": args.warmup,
                                "runs": args.runs,
                            }
                            if args.threads is not None:
                                query["threads"] = args.threads
                            if args.pipeline is not None:
                                query["pipeline"] = args.pipeline
                            if args.only:
                                query["only"] = args.only
                            page.goto(
                                f"http://127.0.0.1:{port}/web/bench.html?"
                                f"{urlencode(query)}"
                            )
                            page.wait_for_function(
                                "window.__benchDone === true || Boolean(window.__benchError)",
                                timeout=TIMEOUT_MS,
                            )
                            error = page.evaluate("window.__benchError")
                            if error:
                                raise RuntimeError(
                                    f"{name} session {session_index}: {error}"
                                )
                            session = page.evaluate("window.__benchResult")
                            session["label"] = args.label
                            path = out_dir / f"session-{session_index}.json"
                            path.write_text(json.dumps(session, indent=2) + "\n")
                            medians = session["session_medians"]
                            print(
                                f"{name} {version} session {session_index}: "
                                f"{session['target']} {session['pkg']} "
                                f"{session['threads']} thread(s), "
                                f"{medians['decode_tok_s']:.2f} tok/s decode, "
                                f"{medians['prefill_tok_s']:.1f} tok/s prefill, "
                                f"{medians['ttft_ms']:.0f} ms ttft, "
                                f"match {session['matches_reference']}",
                                flush=True,
                            )
                        finally:
                            context.close()
                finally:
                    browser.close()
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
