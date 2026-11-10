# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "playwright==1.48.0",
# ]
# ///

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT_MS = 60 * 60 * 1000


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def serve() -> tuple[ThreadingHTTPServer, int]:
    handler = functools.partial(Handler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the browser benchmark in fresh Playwright sessions."
    )
    parser.add_argument("--label", required=True, help="results/runs subdirectory")
    parser.add_argument(
        "--browser",
        action="append",
        choices=["chromium", "firefox", "webkit"],
        help="browser to run, repeatable (default: chromium)",
    )
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--weights", choices=["f32", "int8"], default="int8")
    parser.add_argument("--pkg", choices=["pkg", "pkg-nosimd"], default="pkg")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    browsers = args.browser or ["chromium"]
    out_dir = ROOT / "results" / "runs" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    server, port = serve()
    session_index = 0
    try:
        with sync_playwright() as playwright:
            for name in browsers:
                browser = getattr(playwright, name).launch(headless=True)
                version = browser.version
                try:
                    for _ in range(args.sessions):
                        session_index += 1
                        context = browser.new_context()
                        try:
                            page = context.new_page()
                            query = urlencode(
                                {
                                    "weights": args.weights,
                                    "pkg": args.pkg,
                                    "browser": name,
                                    "browser_version": version,
                                }
                            )
                            page.goto(
                                f"http://127.0.0.1:{port}/web/bench.html?{query}"
                            )
                            page.wait_for_function(
                                "window.__benchDone === true || Boolean(window.__benchError)",
                                timeout=TIMEOUT_MS,
                            )
                            error = page.evaluate("window.__benchError")
                            if error:
                                raise RuntimeError(f"{name} session {session_index}: {error}")
                            session = page.evaluate("window.__benchResult")
                            session["label"] = args.label
                            path = out_dir / f"session-{session_index}.json"
                            path.write_text(json.dumps(session, indent=2) + "\n")
                            medians = session["session_medians"]
                            print(
                                f"{name} {version} session {session_index}: "
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
