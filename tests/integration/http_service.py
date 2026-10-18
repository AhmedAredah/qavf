"""Tiny byte-range HTTP origin for real native file-reader integration tests."""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import urlsplit

from .contracts import corpus


@contextmanager
def http_corpus() -> Iterator[tuple[str, list[str]]]:
    files: dict[str, bytes] = {}
    for clean in (True, False):
        buffer = io.BytesIO()
        corpus(clean).write_parquet(buffer)
        files["/clean.parquet" if clean else "/invalid.parquet"] = buffer.getvalue()
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_HEAD(self):
            self.respond(body=False)

        def do_GET(self):
            requests.append(self.path)
            self.respond(body=True)

        def respond(self, *, body: bool) -> None:
            data = files.get(urlsplit(self.path).path)
            if data is None:
                self.send_error(404)
                return
            start, end, status = 0, len(data) - 1, 200
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                if match is None or not any(match.groups()):
                    self.send_error(416)
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), end) if right else end
                else:
                    start = max(len(data) - int(right), 0)
                if start > end:
                    self.send_error(416)
                    return
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.end_headers()
            if body:
                self.wfile.write(data[start : end + 1])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("HTTP fixture failed to terminate")
