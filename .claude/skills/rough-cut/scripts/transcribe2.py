"""
Improved transcription pass using faster-whisper directly (not the full
whisperx package, so no pyannote-audio/pyav dependency -- runs natively on
Windows without the Smart App Control DLL block).

Why not whisperx.load_model(): whisperx's default decode settings, combined
with long-context conditioning, were found to silently delete entire
repeated utterances from the transcript (verified by isolated re-transcription
of the same audio producing materially different text -- see project notes).
This script disables the behaviors responsible:

- condition_on_previous_text=False: stops later decoding from being biased
  by a running summary of prior text, which was suppressing repeats.
- compression_ratio_threshold raised way up: Whisper's built-in anti-
  hallucination heuristic discards segments whose text looks "too
  repetitive" -- correct for true hallucination loops, wrong for a speaker
  genuinely repeating themselves. Raising it stops real stutters from being
  treated as failures.
- logprob_threshold lowered: same idea, don't discard low-confidence (but
  real) speech.
- vad_filter=True (Silero, bundled in faster-whisper, no pyav needed): still
  prevents the decoder from ever seeing pure silence/noise, which is what
  actually stops hallucinated loops -- decoupled from the two changes above.

Usage:
    python transcribe2.py <input_media> <output_json> [--language pl]
"""

import argparse
import json
import sys

from faster_whisper import WhisperModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_media")
    parser.add_argument("output_json")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    print(f"[transcribe2] loading model={args.model}", file=sys.stderr)
    model = WhisperModel(args.model, device="cpu", compute_type="int8")

    print(f"[transcribe2] transcribing {args.input_media}", file=sys.stderr)
    segments, info = model.transcribe(
        args.input_media,
        language=args.language,
        word_timestamps=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=4.0,
        log_prob_threshold=-2.0,
        no_speech_threshold=0.6,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        temperature=0.0,
        beam_size=5,
    )

    out_segments = []
    for seg in segments:
        words = [
            {"word": w.word.strip(), "start": w.start, "end": w.end, "score": w.probability}
            for w in (seg.words or [])
        ]
        out_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "words": words,
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
        })
        print(f"[transcribe2] [{seg.start:6.2f}-{seg.end:6.2f}] {seg.text}", file=sys.stderr)

    out = {"language": args.language, "segments": out_segments}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[transcribe2] wrote {args.output_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
