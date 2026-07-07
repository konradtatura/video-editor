"""
Verifies a RENDERED rough-cut output -- not the cutlist, the actual mp4 --
because every failure found in this pipeline's development was invisible in
the cutlist-construction-time transcript and only showed up when the
rendered file itself was re-transcribed fresh. The cutlist is a plan; this
checks what was actually built.

Checks performed:
1. Re-transcribes the whole output with large-v3 (independent of whatever
   transcript built the cutlist).
2. Flags repeated word n-grams (default n=4) -- a phrase repeating verbatim
   in the final cut is either a missed duplicate or a genuine stylistic
   repeat (e.g. "warto... warto" as two separate rhetorical beats); either
   way it needs a human look, this script cannot tell the two apart.
3. Flags anomalously long single-word timestamps (>1.2s for a word under
   10 characters) -- this pattern preceded every case where Whisper folded
   a hidden repeat into one word's duration instead of transcribing it.
4. Runs silencedetect on the rendered audio and flags any gap over
   0.4s (normal breath pauses in conversational speech are usually well
   under that; longer gaps are worth listening to).

This does not replace watching the video. It narrows down where to listen.

Usage:
    python verify.py <output.mp4>
"""

import argparse
import subprocess
import sys

from faster_whisper import WhisperModel

from ffmpeg_util import find_ffmpeg

NGRAM_SIZE = 4
LONG_WORD_THRESHOLD_S = 1.2
LONG_WORD_MAX_CHARS = 10
SILENCE_GAP_THRESHOLD_S = 0.4


def transcribe(path, model_size):
    print(f"[verify] transcribing {path} with {model_size}...", file=sys.stderr)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        path, language="pl", word_timestamps=True,
        condition_on_previous_text=False,
        compression_ratio_threshold=4.0, log_prob_threshold=-2.0,
        temperature=0.0, beam_size=5,
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})
    return words


def find_repeated_ngrams(words, n=NGRAM_SIZE):
    seen = {}
    findings = []
    for i in range(len(words) - n + 1):
        gram_words = words[i:i + n]
        key = " ".join(w["word"].lower().strip(",.?!") for w in gram_words)
        if len(key) < 8:
            continue
        if key in seen:
            prev = seen[key]
            findings.append({
                "phrase": key,
                "first_at": prev["start"],
                "second_at": gram_words[0]["start"],
            })
        else:
            seen[key] = gram_words[0]
    return findings


def find_bloated_words(words):
    findings = []
    for w in words:
        dur = w["end"] - w["start"]
        if dur > LONG_WORD_THRESHOLD_S and len(w["word"]) <= LONG_WORD_MAX_CHARS:
            findings.append({"word": w["word"], "start": w["start"], "end": w["end"], "duration": dur})
    return findings


def find_silence_gaps(path, threshold_s):
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-i", path, "-af", f"silencedetect=noise=-30dB:d={threshold_s}", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    gaps = []
    start = None
    for line in result.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].strip())
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split("|")[0].strip())
            gaps.append({"start": start, "end": end, "duration": end - start})
            start = None
    return gaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_media")
    parser.add_argument("--model", default="large-v3")
    args = parser.parse_args()

    words = transcribe(args.output_media, args.model)
    full_text = " ".join(w["word"] for w in words)

    print("\n=== full transcript of rendered output ===")
    print(full_text)

    ngram_findings = find_repeated_ngrams(words)
    print(f"\n=== repeated {NGRAM_SIZE}-word phrases ({len(ngram_findings)}) ===")
    if not ngram_findings:
        print("none found")
    for f in ngram_findings:
        print(f"  '{f['phrase']}'  first at {f['first_at']:.2f}s, again at {f['second_at']:.2f}s -- verify: missed duplicate, or genuine repeated rhetoric?")

    bloat_findings = find_bloated_words(words)
    print(f"\n=== anomalously long single-word timestamps ({len(bloat_findings)}) ===")
    if not bloat_findings:
        print("none found")
    for f in bloat_findings:
        print(f"  '{f['word']}'  {f['start']:.2f}-{f['end']:.2f}s (dur {f['duration']:.2f}s) -- likely hides collapsed repeated content, re-check this span in isolation")

    gaps = find_silence_gaps(args.output_media, SILENCE_GAP_THRESHOLD_S)
    print(f"\n=== silence gaps over {SILENCE_GAP_THRESHOLD_S}s ({len(gaps)}) ===")
    if not gaps:
        print("none found")
    for g in gaps:
        print(f"  {g['start']:.2f}-{g['end']:.2f}s (dur {g['duration']:.2f}s)")

    if not ngram_findings and not bloat_findings and not gaps:
        print("\n[verify] clean -- no automated red flags. Still watch the video before calling it done.")
    else:
        print("\n[verify] red flags found above -- investigate each before considering this cut final.")


if __name__ == "__main__":
    main()
