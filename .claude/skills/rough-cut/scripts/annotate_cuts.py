"""
Produces a REVIEW copy of a rendered output -- not the deliverable, a
throwaway with a clip-number + running-timecode overlay burned in -- plus a
numbered cut-map (clip N: output range, raw range, cutlist note) printed to
stdout and optionally written as JSON.

Why this exists: the review loop's actual bottleneck this project has never
been machine time (render is sub-second cached, verify is ~10s on Groq) --
it's converting what the user hears into a coordinate Claude can act on, and
back. A guessed "cut too early at ~12s" cost a wrong fix and a redo earlier
in this project. If the user can instead say "clip 4" or "the join between
6 and 7", locate.py becomes almost unnecessary and there is no coordinate to
get wrong in either direction.

This does not replace watching output.mp4 -- it's an alternate copy for
*giving feedback against*, since the burned-in overlay is not something
you'd want in the delivered file.

Usage:
    python annotate_cuts.py <project_dir> [--out output_review.mp4] [--cutmap-json cutmap.json]
"""

import argparse
import json
import os
import subprocess
import sys

from ffmpeg_util import find_ffmpeg

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def format_time(t):
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:05.2f}"


def build_cutmap(cutlist, cuts):
    starts = [0.0] + list(cuts[:-1])
    cutmap = []
    for i, (clip, start, end) in enumerate(zip(cutlist["keep"], starts, cuts)):
        note = clip.get("note", "")
        cutmap.append({
            "clip": i + 1,
            "output_start": start,
            "output_end": end,
            "raw_start": clip["start"],
            "raw_end": clip["end"],
            "note_preview": (note[:80] + "...") if len(note) > 80 else note,
        })
    return cutmap


def escape_drawtext(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")


def _fontfile():
    # drawtext needs an explicit fontfile on systems without a working
    # Fontconfig setup (confirmed: fails silently -- well, not silently,
    # loudly, "Cannot load default config file" -- on a fresh Windows ffmpeg
    # build with no Fontconfig installed). Reuse a font already bundled for
    # the captions skill rather than adding a new asset.
    path = os.path.join(os.path.dirname(__file__), "..", "..", "captions", "assets", "Montserrat-SemiBold.ttf")
    return os.path.abspath(path).replace("\\", "/").replace(":", "\\:")


def burn_overlay(ffmpeg, input_video, cutmap, output_video):
    fontfile = _fontfile()
    filters = [
        # running timecode, always visible, top-left
        f"drawtext=fontfile='{fontfile}':text='%{{pts\\:hms}}':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"
    ]
    for c in cutmap:
        label = escape_drawtext(f"clip {c['clip']}")
        filters.append(
            f"drawtext=fontfile='{fontfile}':text='{label}':x=10:y=44:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.55:boxborderw=6:"
            f"enable='between(t,{c['output_start']:.3f},{c['output_end']:.3f})'"
        )
    vf = ",".join(filters)
    # +faststart moves the moov atom to the front of the file -- without it
    # a browser can't even determine duration/seek points without
    # downloading the whole file first, which combined with no Range-request
    # support (see range_server.py) made video.currentTime silently reset to
    # 0 instead of seeking, confirmed directly in the review console.
    cmd = [ffmpeg, "-y", "-i", input_video, "-vf", vf, "-c:a", "copy", "-movflags", "+faststart", output_video]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise RuntimeError("ffmpeg overlay burn failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("--out", default="output_review.mp4")
    parser.add_argument("--cutmap-json", default=None)
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    with open(os.path.join(project_dir, "cutlist.json"), encoding="utf-8") as f:
        cutlist = json.load(f)
    with open(os.path.join(project_dir, "timeline.json"), encoding="utf-8") as f:
        timeline = json.load(f)

    cuts = timeline["cuts"]
    cutmap = build_cutmap(cutlist, cuts)

    print(f"[annotate_cuts] {len(cutmap)} clips\n")
    for c in cutmap:
        print(f"  clip {c['clip']:>2}  output [{format_time(c['output_start'])}-{format_time(c['output_end'])}]  "
              f"raw [{c['raw_start']:.2f}-{c['raw_end']:.2f}]  {c['note_preview']}")

    if args.cutmap_json:
        with open(args.cutmap_json, "w", encoding="utf-8") as f:
            json.dump(cutmap, f, indent=2, ensure_ascii=False)
        print(f"\n[annotate_cuts] wrote {args.cutmap_json}")

    ffmpeg = find_ffmpeg()
    input_video = os.path.join(project_dir, "output.mp4")
    output_video = os.path.join(project_dir, args.out)
    print(f"\n[annotate_cuts] burning clip-number + timecode overlay onto {output_video}...", file=sys.stderr)
    burn_overlay(ffmpeg, input_video, cutmap, output_video)
    print(f"[annotate_cuts] wrote {output_video} -- REVIEW COPY ONLY, do not deliver this file", file=sys.stderr)


if __name__ == "__main__":
    main()
