"""
A static file server with HTTP Range request support, for serving the
review console's video files locally -- and a small write endpoint
(POST /api/trim) that lets review.html's trim UI move a clip's raw
start/end and trigger a re-render without leaving the browser.

Why this exists: Python's stdlib `http.server.SimpleHTTPRequestHandler` has
no Range-request handling at all (confirmed by reading its source directly --
`send_head` never touches the Range header, in Python 3.13). Without it, a
browser's <video> element cannot seek a large file without re-downloading
it from byte 0 every time -- confirmed directly: every request in a real
run showed as a full 200 OK, never a 206 Partial Content, and setting
`video.currentTime` silently reset back to 0 instead of seeking, because
the browser had no way to jump to a byte offset. `python -m http.server`
is fine for the review page's HTML/JS; it is not fine for the video file
sitting next to it.

This handles GET with an optional `Range: bytes=start-end` header, replies
206 with Content-Range/Content-Length when present, 200 with the whole file
otherwise -- enough for <video> seeking, nothing more (no directory
listings beyond the stdlib default, no caching headers beyond what's needed).

/api/trim (POST, JSON body {"project", "clip_index", "start", "end"}) is the
one write path: it edits that project's cutlist.json in place and shells out
to render.py (cache-aware -- only the changed segment gets re-encoded, a few
seconds not a full re-render) so a dragged boundary in the browser becomes a
real edit on disk, the same cutlist.json Claude reads, not a UI-only preview.
Local-only tool, but still a write endpoint -- project name is checked against
path traversal before being joined onto the served directory.

Usage:
    python range_server.py [port] [--directory DIR]
"""

import argparse
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "render.py")

sys.path.insert(0, SCRIPT_DIR)
import review_console  # noqa: E402 -- needs sys.path adjusted above first


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        if not os.path.exists(path):
            self.send_error(404, "File not found")
            return None

        file_size = os.path.getsize(path)
        range_header = self.headers.get("Range")
        ctype = self.guess_type(path)

        if range_header is None:
            f = open(path, "rb")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return f

        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if not match:
            self.send_error(416, "Invalid Range header")
            return None
        start_s, end_s = match.groups()
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return None

        length = end - start + 1
        f = open(path, "rb")
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._range_length = length
        return f

    def copyfile(self, source, outputfile):
        length = getattr(self, "_range_length", None)
        if length is None:
            return super().copyfile(source, outputfile)
        remaining = length
        chunk_size = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(chunk_size, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/trim":
            self._send_json(404, {"ok": False, "error": "no such endpoint"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"ok": False, "error": f"bad JSON body: {e}"})
            return

        project = body.get("project", "")
        clip_index = body.get("clip_index")
        new_start = body.get("start")
        new_end = body.get("end")

        # project must be a bare directory name under the served root, not a
        # path -- this is the one write-capable endpoint on this server, so
        # traversal here would mean writing/executing outside projects/.
        if not project or "/" in project or "\\" in project or project in (".", ".."):
            self._send_json(400, {"ok": False, "error": "invalid project name"})
            return
        project_dir = os.path.abspath(project)
        if not os.path.isdir(project_dir) or os.path.dirname(project_dir) != os.path.abspath("."):
            self._send_json(400, {"ok": False, "error": "project not found"})
            return

        cutlist_path = os.path.join(project_dir, "cutlist.json")
        try:
            with open(cutlist_path, encoding="utf-8") as f:
                cutlist = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._send_json(500, {"ok": False, "error": f"couldn't read cutlist.json: {e}"})
            return

        keep = cutlist.get("keep", [])
        if not isinstance(clip_index, int) or not (0 <= clip_index < len(keep)):
            self._send_json(400, {"ok": False, "error": f"clip_index out of range (0-{len(keep)-1})"})
            return
        try:
            new_start = float(new_start)
            new_end = float(new_end)
        except (TypeError, ValueError):
            self._send_json(400, {"ok": False, "error": "start/end must be numbers"})
            return
        if new_end <= new_start:
            self._send_json(400, {"ok": False, "error": "end must be after start"})
            return

        keep[clip_index]["start"] = round(new_start, 3)
        keep[clip_index]["end"] = round(new_end, 3)
        with open(cutlist_path, "w", encoding="utf-8") as f:
            json.dump(cutlist, f, indent=2)

        result = subprocess.run(
            [sys.executable, RENDER_SCRIPT, project_dir],
            capture_output=True, text=True, timeout=300,
        )
        log = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            self._send_json(500, {"ok": False, "error": "render.py failed", "log": log[-4000:]})
            return

        with open(os.path.join(project_dir, "timeline.json"), encoding="utf-8") as f:
            timeline = json.load(f)
        cutmap = review_console.build_cutmap(cutlist, timeline["cuts"])
        self._send_json(200, {
            "ok": True,
            "log": log[-4000:],
            "cutmap": cutmap,
            "total_duration": timeline["cuts"][-1],
        })


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, nargs="?", default=8420)
    parser.add_argument("--directory", default=".")
    args = parser.parse_args()

    os.chdir(args.directory)
    handler = RangeRequestHandler
    # Plain socketserver.TCPServer handles one request at a time -- a
    # browser's kept-alive connection (the default) then blocks every
    # subsequent request forever, which hung the whole page load when this
    # was first tried. Threading fixes it regardless of keep-alive behavior.
    with ThreadingHTTPServer(("", args.port), handler) as httpd:
        print(f"[range_server] serving {os.path.abspath('.')} on port {args.port} (Range requests supported)", file=sys.stderr)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
