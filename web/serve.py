#!/usr/bin/env python3
import argparse
import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, static_headers=False, **kwargs):
        self.static_headers = static_headers
        super().__init__(*args, **kwargs)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if self.static_headers:
            if self.path.split("?", 1)[0].endswith(".part"):
                self.send_header(
                    "Cache-Control", "public, max-age=31536000, immutable"
                )
            else:
                self.send_header("Cache-Control", "public, max-age=300")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def make_server(root: Path, port: int = 0) -> http.server.ThreadingHTTPServer:
    handler = functools.partial(
        Handler,
        directory=str(root),
        static_headers=(root / "_headers").is_file(),
    )
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve a directory with the cross-origin isolation headers "
        "SharedArrayBuffer needs."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="directory to serve (default: the repository root)",
    )
    args = parser.parse_args()

    server = make_server(args.root, args.port)
    host, port = server.server_address
    path = "/web/" if (args.root / "web").is_dir() else "/"
    print(f"serving {args.root} at http://{host}:{port}{path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
