"""
Prints/writes the acoustic burst inventory for a media file -- every voiced
span bounded by sustained silence, independent of anything Whisper
transcribed. This is the same energy envelope render.py already computes
and caches (audio_boundaries.VoiceEnvelope), just exposed directly instead
of being buried inside ad-hoc ffmpeg silencedetect calls.

Why this matters as its own artifact: the single most useful piece of
evidence in a hard-to-diagnose retake (see rough-cut project notes) was the
shape of the burst durations -- three short bursts of increasing length
followed by one long burst reads unambiguously as "three false starts, then
the take that finally got said in full", and that pattern is visible here
with zero ASR involved. Read this alongside the transcript, not instead of
it: bursts tell you "something happened here and here", the transcript (for
whatever it's worth) tells you what was probably said.

Works on any media file directly -- a project's raw.mp4, or an already-
rendered output.mp4 (e.g. what verify.py checks) -- not just rough-cut
projects, since the envelope only needs the audio itself.

Usage:
    python bursts.py <media_file> [--cache-dir DIR] [--start S] [--end E]

Prints each burst's start, end, duration. With --start/--end, restricts to
that time range (e.g. a span verify.py or a user flagged). --cache-dir
defaults to a sibling ".envelope_cache" folder next to the media file.
"""

import argparse
import json
import os
import sys

from audio_boundaries import load_envelope

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def default_cache_dir(media_path):
    return os.path.join(os.path.dirname(os.path.abspath(media_path)), ".envelope_cache")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media_file")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--json-out", default=None, help="optional path to write bursts as JSON")
    args = parser.parse_args()

    media_path = os.path.abspath(args.media_file)
    cache_dir = args.cache_dir or default_cache_dir(media_path)

    envelope = load_envelope(media_path, cache_dir)
    bursts = envelope.list_bursts(lo=args.start, hi=args.end)

    print(f"[bursts] {len(bursts)} voiced burst(s)" + (f" in {args.start}-{args.end}s" if args.start is not None else ""))
    for b in bursts:
        print(f"  {b['start']:8.3f} - {b['end']:8.3f}  (dur {b['duration']:.3f}s)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(bursts, f, indent=2)
        print(f"[bursts] wrote {args.json_out}")


if __name__ == "__main__":
    main()
