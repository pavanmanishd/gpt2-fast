#!/usr/bin/env python3
import argparse
import functools
import http.server
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
WATCH_SUFFIXES = {".html", ".css", ".js", ".json"}
POLL_SECONDS = 0.3
DEBOUNCE_SECONDS = 1.0
KEEPALIVE_SECONDS = 15
BUILD_SCRIPT = ROOT / "web" / "build.sh"
RUST_WATCH = (
    ROOT / "crates" / "engine" / "src",
    ROOT / "crates" / "web" / "src",
    ROOT / "Cargo.toml",
    ROOT / "Cargo.lock",
)

RELOAD_SCRIPT = """<script>
(() => {
  const source = new EventSource("/__reload");
  let banner = null;
  source.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "reload") location.reload();
    if (message.type === "error") {
      if (!banner) {
        banner = document.createElement("div");
        banner.id = "gpt2-build-error";
        banner.style.cssText = "position:fixed;left:12px;bottom:12px;z-index:2147483647;max-width:min(640px,90vw);padding:8px 10px;border-radius:6px;background:#3b1515;color:#ffd9d9;font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;box-shadow:0 2px 12px #0006";
        document.body.append(banner);
      }
      banner.textContent = "wasm build failed\\n" + message.lines.join("\\n");
    }
  };
})();
</script>
"""


def stat_key(path: Path):
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


def web_snapshot(web: Path) -> dict:
    state = {}
    for dirpath, dirnames, filenames in os.walk(web):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in ("dist", "__pycache__", "node_modules", ".git")
        ]
        relative = Path(dirpath).relative_to(web)
        in_package = bool(relative.parts) and relative.parts[0].startswith("pkg")
        if in_package:
            dirnames[:] = []
        for name in filenames:
            path = Path(dirpath) / name
            if in_package or path.suffix.lower() in WATCH_SUFFIXES:
                state[str(path)] = stat_key(path)
    return state


def rust_snapshot() -> dict:
    state = {}
    for base in RUST_WATCH:
        if base.is_file():
            state[str(base)] = stat_key(base)
        elif base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    state[str(path)] = stat_key(path)
    return state


def first_errors(lines, limit=8):
    errors = [
        line.strip()
        for line in lines
        if "error" in line.lower() or "panicked" in line.lower()
    ]
    if not errors:
        errors = [line.strip() for line in lines if line.strip()][-limit:]
    return [line[:300] for line in errors[:limit]]


def run_build():
    lines = []
    for args in ((), ("--threads",)):
        print("$ web/build.sh" + "".join(f" {arg}" for arg in args), flush=True)
        process = subprocess.Popen(
            ["bash", str(BUILD_SCRIPT), *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line.rstrip())
        if process.wait() != 0:
            return False, lines
    return True, lines


class ReloadHub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.building = threading.Event()

    def subscribe(self):
        client = queue.Queue(maxsize=8)
        with self.lock:
            self.clients.append(client)
        return client

    def unsubscribe(self, client):
        with self.lock:
            try:
                self.clients.remove(client)
            except ValueError:
                pass

    def publish(self, message):
        data = json.dumps(message)
        with self.lock:
            clients = list(self.clients)
        for client in clients:
            try:
                client.put_nowait(data)
            except queue.Full:
                pass


class Watcher(threading.Thread):
    def __init__(self, hub: ReloadHub, web: Path, rebuild: bool, trigger=None):
        super().__init__(name="reload-watcher", daemon=True)
        self.hub = hub
        self.web = web
        self.rebuild = rebuild
        self.trigger = trigger if trigger is not None else threading.Event()
        self.state = web_snapshot(web)
        self.rust_state = rust_snapshot() if rebuild else {}

    def run(self):
        while True:
            time.sleep(POLL_SECONDS)
            state = web_snapshot(self.web)
            if state != self.state:
                self.state = state
                if not self.hub.building.is_set():
                    self.hub.publish({"type": "reload"})
            if not self.rebuild:
                continue
            state = rust_snapshot()
            if state != self.rust_state:
                self.rust_state = state
                self.trigger.set()


class Builder(threading.Thread):
    def __init__(self, hub: ReloadHub, trigger: threading.Event):
        super().__init__(name="wasm-builder", daemon=True)
        self.hub = hub
        self.trigger = trigger

    def run(self):
        while True:
            self.trigger.wait()
            self.trigger.clear()
            while True:
                time.sleep(DEBOUNCE_SECONDS)
                if self.trigger.is_set():
                    self.trigger.clear()
                    continue
                break
            self.hub.building.set()
            try:
                ok, lines = run_build()
            finally:
                self.hub.building.clear()
            if ok:
                print("wasm build finished; reloading pages", flush=True)
                self.hub.publish({"type": "reload"})
            else:
                errors = first_errors(lines)
                print("wasm build failed", file=sys.stderr, flush=True)
                self.hub.publish({"type": "error", "lines": errors})


def inject_reload(body: bytes) -> bytes:
    script = RELOAD_SCRIPT.encode("utf-8")
    lowered = body.lower()
    for marker in (b"</body>", b"</html>"):
        index = lowered.rfind(marker)
        if index != -1:
            return body[:index] + script + body[index:]
    return body + script


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        static_headers=False,
        watch=False,
        hub=None,
        **kwargs,
    ):
        self.static_headers = static_headers
        self.watch = watch
        self.hub = hub
        if watch:
            self.protocol_version = "HTTP/1.1"
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

    def html_path(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            if not urlsplit(self.path).path.endswith("/"):
                return None
            for name in ("index.html", "index.htm"):
                candidate = path / name
                if candidate.is_file():
                    return candidate
            return None
        if path.is_file() and path.suffix.lower() in (".html", ".htm"):
            return path
        return None

    def send_injected_html(self, path: Path, body=True):
        content = inject_reload(path.read_bytes())
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        if body:
            self.wfile.write(content)

    def send_reload_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        client = self.hub.subscribe()
        try:
            while True:
                try:
                    data = client.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                else:
                    self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.unsubscribe(client)

    def do_GET(self):
        if self.watch:
            if urlsplit(self.path).path == "/__reload":
                self.send_reload_stream()
                return
            path = self.html_path()
            if path is not None:
                self.send_injected_html(path)
                return
        super().do_GET()

    def do_HEAD(self):
        if self.watch:
            path = self.html_path()
            if path is not None:
                self.send_injected_html(path, body=False)
                return
        super().do_HEAD()


def make_server(
    root: Path,
    port: int = 0,
    watch: bool = False,
    rebuild: bool = False,
) -> http.server.ThreadingHTTPServer:
    hub = ReloadHub() if watch else None
    handler = functools.partial(
        Handler,
        directory=str(root),
        static_headers=(root / "_headers").is_file(),
        watch=watch,
        hub=hub,
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.reload_hub = hub
    server.web_root = root / "web" if (root / "web").is_dir() else root
    server.rebuild = rebuild
    return server


def start_watch(server: http.server.ThreadingHTTPServer):
    if server.rebuild:
        trigger = threading.Event()
        builder = Builder(server.reload_hub, trigger)
        watcher = Watcher(server.reload_hub, server.web_root, True, trigger)
        builder.start()
    else:
        watcher = Watcher(server.reload_hub, server.web_root, False)
    watcher.start()
    return watcher


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
    parser.add_argument(
        "--watch",
        action="store_true",
        help="reload open pages when web/ changes",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="also rebuild the wasm packages when the Rust sources change "
        "(implies --watch)",
    )
    args = parser.parse_args()

    watch = args.watch or args.rebuild
    server = make_server(args.root, args.port, watch=watch, rebuild=args.rebuild)
    host, port = server.server_address
    path = "/web/" if (args.root / "web").is_dir() else "/"
    print(f"serving {args.root} at http://{host}:{port}{path}", flush=True)
    if watch:
        print(f"watching {server.web_root} for changes", flush=True)
    if args.rebuild:
        print("rebuilding on Rust changes", flush=True)
    if watch:
        start_watch(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
