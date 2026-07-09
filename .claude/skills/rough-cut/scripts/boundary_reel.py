"""
Extracts a short window around every internal cut join in a rendered output
and concatenates them into one short "boundary reel" with a join-number
overlay -- lets the user audit just the joins in a fraction of the time of
a full re-watch.

Why this is worth having: every reported problem in this project that
involved re-editing a boundary ("cut too early at 12s", a clipped word, a
leftover stutter fragment) lived at a cut join, never in the middle of an
untouched clip. A full re-watch spends most of its time confirming footage
that didn't change. A 10-join video produces a ~30s reel instead of a
70s+ re-watch, and concentrates attention exactly where problems actually
occur.

This is a review aid, not the deliverable -- like annotate_cuts.py's output,
don't ship this file.

Usage:
    python boundary_reel.py <project_dir> [--out boundary_reel.mp4] [--pad 1.5]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from ffmpeg_util import find_ffmpeg

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_PAD_S = 1.5


def _fontfile():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "captions", "assets", "Montserrat-SemiBold.ttf")
    return os.path.abspath(path).replace("\\", "/").replace(":", "\\:")


def _probe_duration(ffprobe, path):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("--out", default="boundary_reel.mp4")
    parser.add_argument("--pad", type=float, default=DEFAULT_PAD_S)
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    with open(os.path.join(project_dir, "timeline.json"), encoding="utf-8") as f:
        timeline = json.load(f)
    cuts = timeline["cuts"]
    joins = cuts[:-1]  # internal joins only -- the last cut is the video's own end, not a join

    if not joins:
        print("[boundary_reel] only one clip, no internal joins to review.", file=sys.stderr)
        return

    ffmpeg = find_ffmpeg()
    ffprobe = find_ffmpeg("ffprobe")
    input_video = os.path.join(project_dir, "output.mp4")
    total_duration = _probe_duration(ffprobe, input_video)
    fontfile = _fontfile()

    print(f"[boundary_reel] {len(joins)} internal join(s), pad={args.pad}s", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp_dir:
        seg_files = []
        for i, j in enumerate(joins):
            start = max(0.0, j - args.pad)
            end = min(total_duration, j + args.pad)
            label = f"join {i + 1}|{i + 2}"  # e.g. "join 4|5" = the cut between clip 4 and clip 5
            seg_path = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
            vf = (
                f"drawtext=fontfile='{fontfile}':text='{label}':x=10:y=10:fontsize=30:fontcolor=yellow:"
                f"box=1:boxcolor=black@0.6:boxborderw=8"
            )
            cmd = [
                ffmpeg, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", input_video,
                "-vf", vf, "-c:a", "aac", seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stderr[-3000:], file=sys.stderr)
                raise RuntimeError(f"ffmpeg failed on join {i + 1}")
            seg_files.append(seg_path)
            print(f"  join {i + 1:>2}|{i + 2:<2}  output ~{j:.2f}s  (reel span {start:.2f}-{end:.2f})", file=sys.stderr)

        concat_list = os.path.join(tmp_dir, "_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for seg in seg_files:
                escaped = seg.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        out_path = os.path.join(project_dir, args.out)
        cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", out_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-3000:], file=sys.stderr)
            raise RuntimeError("ffmpeg concat failed")

    print(f"[boundary_reel] wrote {out_path} -- {len(joins)} joins, ~{len(joins) * args.pad * 2:.0f}s total. Review copy only, do not deliver.", file=sys.stderr)


if __name__ == "__main__":
    main()
