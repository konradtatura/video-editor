"""
Render a cutlist into a final video, re-encoding only segments that changed
since the last render (content-hash cache) and stitching the rest instantly.

cutlist.json shape:
{
  "source": "raw.mp4",          // relative to the cutlist.json's directory
  "pad": 0.08,                   // seconds of padding added to each side of a cut
  "keep": [
    {"start": 0.451, "end": 6.555, "note": "..."},
    ...
  ]
}

Usage:
    python render.py <project_dir> [--out output.mp4]

project_dir must contain cutlist.json and the source video it references.
Segment cache lives in <project_dir>/segments/.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from audio_boundaries import load_envelope
from ffmpeg_util import find_ffmpeg

SETTINGS_VERSION = "v3-sustained-snap"


def seg_hash(source_path, start, end, pad):
    stat = os.stat(source_path)
    key = f"{source_path}|{stat.st_size}|{stat.st_mtime}|{round(start, 3)}|{round(end, 3)}|{pad}|{SETTINGS_VERSION}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def render_segment(ffmpeg, source_path, start, end, out_path):
    dur = end - start
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", source_path,
        "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed on segment {start}-{end}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("--out", default="output.mp4")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    cutlist_path = os.path.join(project_dir, "cutlist.json")
    segments_dir = os.path.join(project_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    with open(cutlist_path, encoding="utf-8") as f:
        cutlist = json.load(f)

    source_path = os.path.join(project_dir, cutlist["source"])
    pad = cutlist.get("pad", 0.08)
    keep = cutlist["keep"]

    ffmpeg = find_ffmpeg()
    envelope = load_envelope(source_path, segments_dir)

    seg_files = []
    hits, misses = 0, 0
    cum = 0.0
    cuts = []
    for i, clip in enumerate(keep):
        snapped_start = envelope.snap_start(clip["start"], upper_bound=clip["end"])
        snapped_end = envelope.snap_end(clip["end"], lower_bound=snapped_start)
        if abs(snapped_start - clip["start"]) > 0.02 or abs(snapped_end - clip["end"]) > 0.02:
            print(f"[render] clip {i+1}: snapped {clip['start']:.3f}-{clip['end']:.3f} -> {snapped_start:.3f}-{snapped_end:.3f}", file=sys.stderr)
        start = max(0.0, snapped_start)
        end = snapped_end
        h = seg_hash(source_path, start, end, pad)
        seg_path = os.path.join(segments_dir, f"{h}.mp4")
        if os.path.exists(seg_path):
            hits += 1
        else:
            misses += 1
            print(f"[render] encoding segment {i+1}/{len(keep)}: {start:.2f}-{end:.2f}  ({clip.get('note', '')})", file=sys.stderr)
            render_segment(ffmpeg, source_path, start, end, seg_path)
        seg_files.append(seg_path)
        cum += (end - start)
        cuts.append(round(cum, 3))

    print(f"[render] cache: {hits} hits, {misses} encoded", file=sys.stderr)

    # cross-skill contract: any consumer of output.mp4 (e.g. the captions
    # skill) can read this to know where the hard cuts are in the OUTPUT
    # timeline, without needing to know about cutlist.json or re-derive
    # snapped boundaries itself. "cuts" are cumulative end-times of each
    # kept clip -- a caption/subtitle card should never span across one.
    timeline_path = os.path.join(project_dir, "timeline.json")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({"cuts": cuts}, f, indent=2)

    concat_list_path = os.path.join(project_dir, "_concat.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for seg_path in seg_files:
            escaped = seg_path.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    out_path = os.path.join(project_dir, args.out)
    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list_path,
        "-c", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("ffmpeg concat failed")

    os.remove(concat_list_path)
    print(f"[render] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
