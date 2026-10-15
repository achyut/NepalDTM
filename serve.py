"""Static server for the NepalDTM map in index.html.

python -m http.server speaks HTTP/1.0 (a new TCP connection per tile) with a
listen backlog of 5, which crawls over a long round trip. This keeps
connections open, sends Cache-Control so terrain and hillshade share one
download, and gzips text (the points GeoJSON is ~3 MB raw).

Run:
    python3 serve.py [--port 8000]
"""

from __future__ import annotations

import argparse
import gzip
import os
from functools import lru_cache, partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEXT = {".html", ".js", ".json", ".geojson", ".css"}


@lru_cache(maxsize=64)
def gzipped(path: str, mtime: float) -> bytes:
    return gzip.compress(Path(path).read_bytes(), 6)


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".webp": "image/webp", ".geojson": "application/geo+json"}

    def end_headers(self):
        p = self.path.split("?", 1)[0]
        if p.startswith("/tiles/") and p.endswith(".webp"):
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if (os.path.splitext(path)[1] in TEXT and os.path.isfile(path)
                and "gzip" in self.headers.get("Accept-Encoding", "")):
            body = gzipped(path, os.path.getmtime(path))
            self.send_response(200)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def list_directory(self, path):  # no directory listings
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="0.0.0.0")
    args = ap.parse_args()
    Server((args.bind, args.port), partial(Handler, directory=str(ROOT))).serve_forever()


if __name__ == "__main__":
    main()
