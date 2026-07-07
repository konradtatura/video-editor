"""
Maps a timestamp in the RENDERED output back to the corresponding clip and
timestamp in the RAW source footage. cutlist.json stores raw-footage times;
a human giving feedback naturally gives output-relative times ("the pause
at 0:45 is too long") -- this closes that gap without manual arithmetic.

Recomputes the same snapped boundaries render.py uses (so the cumulative
timeline matches what's actually in output.mp4, not the unsnapped cutlist
values) and reports which keep-entry a given output time falls in, plus the
exact corresponding raw-footage time.

Usage:
    python locate.py <project_dir> <output_time_seconds>
"""

import argparse
import json
import os
import sys

from audio_boundaries import load_envelope

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir")
    parser.add_argument("output_time", type=float, help="timestamp in output.mp4, in seconds")
    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    with open(os.path.join(project_dir, "cutlist.json"), encoding="utf-8") as f:
        cutlist = json.load(f)

    source_path = os.path.join(project_dir, cutlist["source"])
    segments_dir = os.path.join(project_dir, "segments")
    envelope = load_envelope(source_path, segments_dir)

    cum = 0.0
    for i, clip in enumerate(cutlist["keep"]):
        s = envelope.snap_start(clip["start"], upper_bound=clip["end"])
        e = envelope.snap_end(clip["end"], lower_bound=s)
        dur = e - s
        if cum <= args.output_time <= cum + dur:
            offset = args.output_time - cum
            raw_time = s + offset
            print(f"output {args.output_time:.3f}s falls in keep-entry {i+1}/{len(cutlist['keep'])}")
            print(f"  cutlist entry: start={clip['start']}, end={clip['end']}")
            print(f"  snapped raw range: {s:.3f}-{e:.3f}  (output {cum:.3f}-{cum+dur:.3f})")
            print(f"  note: {clip.get('note', '')}")
            print(f"  ---> corresponding RAW-footage time: {raw_time:.3f}s")
            return
        cum += dur

    print(f"output time {args.output_time:.3f}s is past the end of the rendered timeline (total {cum:.3f}s)", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
