"""
Second-opinion repeat check: transcribes the raw file with CTC
(ctc_backend.py, wav2vec2) instead of Whisper, finds repeated word n-grams
in that output, and cross-references each one against the primary Whisper
transcript (transcript_a.json) to flag whether Whisper's transcript shows
the same repeat or not.

Why a second full pass, not just a targeted check on bloated words: a
bloated Whisper word-timestamp is a real signal but not the only shape this
failure takes -- Whisper can also fuse a repeat into fluent-sounding
*normal-length* words with nothing to flag it (verified: neither the
within-span self-similarity nor the front/back subsequence-match approach
could reliably separate a real fused repeat from ordinary speech using only
the suspect span's own audio -- both were tested and fell through on real
footage). CTC's own decode has no equivalent bias to correct for, so it
does not need a bloated-word trigger to work from -- it can be run as an
independent full pass and cross-referenced afterward.

This is NOT a transcript source (see ctc_backend.py's docstring) -- read
CTC's output only for repeat *structure* (does this n-gram appear twice),
never for wording. A CTC-only flag with no Whisper corroboration is the
highest-priority case: it's exactly the shape of a Whisper-suppressed
repeat this tool was built to catch.

Usage:
    python ctc_check.py <raw_media> [--start S] [--end E] [--ngram N]
        [--whisper-transcript transcript_a.json] [--chunk-s 25]
"""

import argparse
import json
import sys

from ctc_backend import transcribe_words

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_NGRAM = 3
DEFAULT_CHUNK_S = 25.0
CHUNK_OVERLAP_S = 1.0
CROSS_REFERENCE_WINDOW_S = 3.0


def _probe_duration(media_path):
    import subprocess
    from ffmpeg_util import find_ffmpeg
    ffprobe = find_ffmpeg("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def transcribe_full(media_path, start, end, chunk_s):
    """Chunks into fixed windows (CTC has no long-context suppression bias
    to avoid, unlike Whisper -- chunking here is purely for memory/runtime
    on CPU, not correctness) with a small overlap so a word landing right
    on a chunk boundary isn't split with too little context on either side."""
    total = _probe_duration(media_path)
    lo = start if start is not None else 0.0
    hi = end if end is not None else total

    all_words = []
    pos = lo
    while pos < hi:
        c_end = min(pos + chunk_s, hi)
        c_start = max(lo, pos - (CHUNK_OVERLAP_S if pos > lo else 0.0))
        words = transcribe_words(media_path, start=c_start, end=c_end)
        # drop words that start before this chunk's own (non-overlapping)
        # window start, except for the very first chunk -- avoids
        # double-counting the overlap region
        if c_start < pos:
            words = [w for w in words if w["start"] >= pos]
        all_words.extend(words)
        print(f"[ctc_check] chunk [{c_start:.2f}-{c_end:.2f}]: {' '.join(w['word'] for w in words)}", file=sys.stderr)
        pos = c_end
    return all_words


def find_repeated_ngrams(words, n):
    seen = {}
    findings = []
    for i in range(len(words) - n + 1):
        gram = words[i:i + n]
        key = " ".join(w["word"].lower() for w in gram)
        if len(key) < n * 2:  # skip trivially short/noisy grams
            continue
        if key in seen:
            prev = seen[key]
            findings.append({"phrase": key, "first_at": prev["start"], "second_at": gram[0]["start"]})
        else:
            seen[key] = gram[0]
    return findings


def find_adjacent_repeats(words, min_k=1, max_k=6):
    """Catches 'X X' where the two X's are IMMEDIATELY adjacent in the word
    stream (words[i:i+k] == words[i+k:i+2k]) -- this is the signature of a
    genuinely gapless fused repeat, which find_repeated_ngrams structurally
    cannot see when X is shorter than the n-gram size (a sliding window at
    n=3 never produces two identical 3-grams from a back-to-back 2-word
    repeat like 'im głupszy im głupszy' -- confirmed missing this exact
    case on real footage before this function was added). Checked from
    longest k down to shortest per position so a real multi-word phrase
    repeat isn't reported redundantly as several shorter overlapping ones."""
    findings = []
    lowered = [w["word"].lower() for w in words]
    i = 0
    n = len(words)
    while i < n:
        matched_k = None
        for k in range(min(max_k, (n - i) // 2), min_k - 1, -1):
            if lowered[i:i + k] == lowered[i + k:i + 2 * k]:
                matched_k = k
                break
        if matched_k:
            phrase = " ".join(lowered[i:i + matched_k])
            if len(phrase) >= matched_k * 2:  # skip trivially short/noisy matches
                findings.append({
                    "phrase": phrase,
                    "first_at": words[i]["start"],
                    "second_at": words[i + matched_k]["start"],
                })
            i += 2 * matched_k  # skip past the whole repeated span, don't re-match inside it
        else:
            i += 1
    return findings


def load_whisper_repeats(transcript_path, n):
    """Re-derive the primary transcript's own word list + repeated n-grams,
    so a CTC finding can be checked against what Whisper already flagged at
    that time, without needing verify.py's live re-transcription."""
    with open(transcript_path, encoding="utf-8") as f:
        data = json.load(f)
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            words.append({"word": w["word"].strip(",.?!…"), "start": w["start"]})
    return find_repeated_ngrams(words, n) + find_adjacent_repeats(words)


def dedupe_findings(findings, window_s=1.0):
    """find_repeated_ngrams and find_adjacent_repeats can both flag the same
    underlying repeat (e.g. a 3+ word phrase that's also immediately
    adjacent) -- collapse findings whose first_at/second_at are both within
    window_s of an already-kept one, preferring the adjacent-repeat version
    since its span is exact rather than n-gram-window-dependent."""
    kept = []
    for f in sorted(findings, key=lambda x: (x["first_at"], -len(x["phrase"]))):
        if any(abs(f["first_at"] - k["first_at"]) <= window_s and abs(f["second_at"] - k["second_at"]) <= window_s for k in kept):
            continue
        kept.append(f)
    return kept


def cross_reference(ctc_findings, whisper_findings, window_s):
    for f in ctc_findings:
        corroborated = any(
            abs(f["first_at"] - wf["first_at"]) <= window_s or abs(f["second_at"] - wf["second_at"]) <= window_s
            for wf in whisper_findings
        )
        f["whisper_corroborated"] = corroborated
    return ctc_findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media_path")
    parser.add_argument("--start", type=float, default=None)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM)
    parser.add_argument("--chunk-s", type=float, default=DEFAULT_CHUNK_S)
    parser.add_argument("--whisper-transcript", default=None,
                         help="path to transcript_a.json -- cross-references each CTC-flagged repeat against Whisper's own repeat findings")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    print("[ctc_check] transcribing via CTC (wav2vec2) -- this is a repeat-structure check, not a transcript source...", file=sys.stderr)
    words = transcribe_full(args.media_path, args.start, args.end, args.chunk_s)
    print(f"[ctc_check] {len(words)} CTC words total", file=sys.stderr)

    findings = find_repeated_ngrams(words, args.ngram) + find_adjacent_repeats(words)
    findings = dedupe_findings(findings)

    if args.whisper_transcript:
        whisper_findings = load_whisper_repeats(args.whisper_transcript, args.ngram)
        findings = cross_reference(findings, whisper_findings, CROSS_REFERENCE_WINDOW_S)

    print(f"\n[ctc_check] {len(findings)} repeated {args.ngram}-gram(s) found in CTC output")
    for f in findings:
        if "whisper_corroborated" in f:
            tag = "[ALSO in Whisper transcript]" if f["whisper_corroborated"] else "[**NOT in Whisper transcript -- investigate**]"
        else:
            tag = ""
        print(f"  '{f['phrase']}'  first at {f['first_at']:.2f}s, again at {f['second_at']:.2f}s  {tag}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(findings, f, indent=2)
        print(f"[ctc_check] wrote {args.json_out}")

    if not findings:
        print("\n[ctc_check] no repeated n-grams found by CTC in this range.")
    else:
        print("\n[ctc_check] CTC text is phonetically rough -- read these only as 'this n-gram repeated', "
              "never as ground truth wording. Cross-check each against the raw audio (bursts.py/find_repeats.py, "
              "or a listen) before treating it as a confirmed retake.")


if __name__ == "__main__":
    main()
