"""
A static file server with HTTP Range request support, for serving the
review console's video files locally.

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
import os
import re
import socketserver
import sys


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
