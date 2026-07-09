"""
A static file server with HTTP Range request support, for serving the
review console's video files locally -- plus one write endpoint,
POST /api/delete-range, that lets review.html's mark-and-delete tool cut
an exact output-time range out of cutlist.json and re-render, with no
re-verification step (by explicit request: one-shot automatic pass is
Claude's job, manual precise deletion afterward is the human's, and nothing
should re-check or re-interpret a marked delete range once it's been cut).

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

from audio_boundaries import load_envelope  # noqa: E402 -- needs sys.path adjusted above first
import review_console  # noqa: E402


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
        if self.path != "/api/delete-range":
            self._send_json(404, {"ok": False, "error": "no such endpoint"})
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send_json(400, {"ok": False, "error": f"bad JSON body: {e}"})
            return

        project = body.get("project", "")
        try:
            del_start = float(body.get("start"))
            del_end = float(body.get("end"))
        except (TypeError, ValueError):
            self._send_json(400, {"ok": False, "error": "start/end must be numbers"})
            return
        if del_end <= del_start:
            self._send_json(400, {"ok": False, "error": "end must be after start"})
            return

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

        source_path = os.path.join(project_dir, cutlist["source"])
        segments_dir = os.path.join(project_dir, "segments")
        try:
            envelope = load_envelope(source_path, segments_dir)
        except Exception as e:
            self._send_json(500, {"ok": False, "error": f"couldn't load audio envelope: {e}"})
            return

        # Recompute the same snapped start/end + cumulative output-time
        # ranges render.py itself uses, so the delete range (given in
        # output time by the browser) maps onto the exact raw-footage
        # offsets actually in output.mp4 -- not the unsnapped cutlist
        # values. This is a coordinate transform, not a content judgment:
        # the mark is taken literally, nothing here second-guesses it.
        new_keep = []
        cum = 0.0
        MIN_DUR = 0.05
        for clip in cutlist["keep"]:
            s = envelope.snap_start(clip["start"], upper_bound=clip["end"])
            e = envelope.snap_end(clip["end"], lower_bound=s)
            dur = e - s
            out_start, out_end = cum, cum + dur
            cum = out_end

            overlap_start = max(del_start, out_start)
            overlap_end = min(del_end, out_end)
            if overlap_start >= overlap_end:
                new_keep.append(clip)  # no overlap with the delete range
                continue

            if overlap_start <= out_start and overlap_end >= out_end:
                continue  # entire entry deleted

            # offset within this entry maps 1:1 between raw and output time
            if overlap_start > out_start:
                pre_end = s + (overlap_start - out_start)
                if pre_end - clip["start"] > MIN_DUR:
                    new_keep.append({**clip, "end": round(pre_end, 3)})
            if overlap_end < out_end:
                post_start = s + (overlap_end - out_start)
                if clip["end"] - post_start > MIN_DUR:
                    new_keep.append({**clip, "start": round(post_start, 3)})

        if not new_keep:
            self._send_json(400, {"ok": False, "error": "that delete range would remove the entire video"})
            return

        cutlist["keep"] = new_keep
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
