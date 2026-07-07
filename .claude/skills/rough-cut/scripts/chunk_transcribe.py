"""
Transcribes a source file in short, independently-decoded chunks split at
real pauses (measured from the audio energy envelope, not Whisper's own
VAD/segmentation), then stitches the results back into one global-time
transcript.

Why: a single long-context Whisper pass was empirically shown to silently
delete entire repeated utterances (verified by re-transcribing a 22s extract
of the same audio in isolation and getting materially more content back).
Keeping each decode call short and independent (fresh decoder state, no
cross-chunk conditioning) reduces that bias -- this isn't a total fix
(repeat-suppression is partly a trained decoder bias, not just an API
setting).

MODEL SIZE MATTERS MORE THAN CHUNKING: a rapid stutter with no acoustic
pause between repeats (e.g. "jeśli chcesz, jeśli chcesz, jeśli chcesz
pisać...") was silently collapsed by the "medium" model even within a
single ~6s chunk -- chunking alone did not fix it. Re-running the exact
same audio through "large-v3" surfaced the repeats correctly. Default here
is large-v3; do not downgrade to medium/small for anything but a quick
throwaway preview, and never trust medium's output for a final editorial
decision.

Chunking and model size only reduce the failure rate -- they do not
eliminate it. Always verify the RENDERED output with verify.py before
considering a cut final (see SKILL.md).

Usage:
    python chunk_transcribe.py <input_media> <output_json> [--language pl]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from faster_whisper import WhisperModel

from audio_boundaries import load_envelope
from ffmpeg_util import find_ffmpeg

MIN_PAUSE_S = 0.35
MIN_CHUNK_S = 3.0
MAX_CHUNK_S = 18.0


def find_split_points(envelope, total_duration):
    """Midpoints of sustained silent stretches, used as candidate chunk
    boundaries -- splitting on real measured pauses instead of Whisper's
    own VAD avoids baking the same model's judgment back into the fix."""
    voiced = envelope.voiced
    times = envelope.times
    splits = []
    i = 0
    n = len(voiced)
    while i < n:
        if not voiced[i]:
            j = i
            while j < n and not voiced[j]:
                j += 1
            duration = times[j - 1] - times[i] if j > i else 0
            if duration >= MIN_PAUSE_S:
                splits.append((times[i] + times[j - 1]) / 2)
            i = j
        else:
            i += 1
    return splits


def build_chunks(split_points, total_duration):
    chunks = []
    chunk_start = 0.0
    for sp in split_points:
        if sp - chunk_start >= MIN_CHUNK_S:
            if sp - chunk_start > MAX_CHUNK_S:
                # force-split evenly across the oversized span
                n_sub = int((sp - chunk_start) // MAX_CHUNK_S) + 1
                step = (sp - chunk_start) / n_sub
                pos = chunk_start
                for _ in range(n_sub):
                    chunks.append((pos, pos + step))
                    pos += step
            else:
                chunks.append((chunk_start, sp))
            chunk_start = sp
    if total_duration - chunk_start > 0.1:
        chunks.append((chunk_start, total_duration))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_media")
    parser.add_argument("output_json")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    cache_dir = os.path.join(os.path.dirname(args.output_json), "segments")
    envelope = load_envelope(args.input_media, cache_dir)
    total_duration = float(envelope.times[-1])

    splits = find_split_points(envelope, total_duration)
    chunks = build_chunks(splits, total_duration)
    print(f"[chunk_transcribe] {len(chunks)} chunks from {len(splits)} measured pauses", file=sys.stderr)

    ffmpeg = find_ffmpeg()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    all_segments = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, (c_start, c_end) in enumerate(chunks):
            chunk_wav = os.path.join(tmp_dir, f"chunk_{idx}.wav")
            cmd = [
                ffmpeg, "-y", "-ss", f"{c_start:.3f}", "-to", f"{c_end:.3f}",
                "-i", args.input_media, "-ac", "1", "-ar", "16000", chunk_wav,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            segments, info = model.transcribe(
                chunk_wav,
                language=args.language,
                word_timestamps=True,
                condition_on_previous_text=False,
                compression_ratio_threshold=4.0,
                log_prob_threshold=-2.0,
                no_speech_threshold=0.6,
                temperature=0.0,
                beam_size=5,
            )

            chunk_text_parts = []
            for seg in segments:
                words = [
                    {"word": w.word.strip(), "start": w.start + c_start, "end": w.end + c_start, "score": w.probability}
                    for w in (seg.words or [])
                ]
                all_segments.append({
                    "start": seg.start + c_start,
                    "end": seg.end + c_start,
                    "text": seg.text,
                    "words": words,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                })
                chunk_text_parts.append(seg.text)
            print(f"[chunk_transcribe] chunk {idx+1}/{len(chunks)} [{c_start:.2f}-{c_end:.2f}]: {' '.join(chunk_text_parts)}", file=sys.stderr)

    out = {"language": args.language, "segments": all_segments}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[chunk_transcribe] wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
