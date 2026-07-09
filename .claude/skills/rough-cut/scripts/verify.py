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
   repeat (e.g. "warto... warto" as two separate rhetorical beats), OR (a
   failure mode found the hard way) a Whisper decoder repetition-loop
   artifact from decoding this specific file, where the audio has no
   repeat at all. Text alone cannot tell these apart.
3. Flags anomalously long single-word timestamps (>1.2s for a word under
   10 characters) -- this pattern preceded every case where Whisper folded
   a hidden repeat into one word's duration instead of transcribing it.
4. Runs silencedetect on the rendered audio and flags any gap over
   0.4s (normal breath pauses in conversational speech are usually well
   under that; longer gaps are worth listening to).
5. NEW: cross-checks every finding from #2/#3 against the audio itself
   (bursts.py + find_repeats.py's DTW comparison) instead of leaving that
   entirely to a human re-listen. This exists because both failure
   directions were hit on real footage in the same project: a genuine
   repeat got waved off as a "transcription artifact" (wrong -- it was
   real, confirmed only after the user insisted), and separately a
   decoder repetition-loop produced 10 identical-looking text-repeat
   flags that were NOT real (confirmed by checking the actual audio
   waveform, which showed no anomaly at all). An acoustic check answers
   "is there really a second similar-sounding burst near here" directly
   from signal energy/spectral shape, which a decoder hallucination
   cannot fake and repeat-suppression cannot hide.

This does not replace watching the video. It narrows down where to listen,
and the acoustic check narrows it down further -- but it is still evidence
to weigh, not a verdict: a fused burst (see find_repeats.py's documented
limitation) can hide a real repeat from the acoustic check too, and if the
user reports something is off, that report outranks every automated signal
here.

Usage:
    python verify.py <output.mp4>
"""

import argparse
import os
import subprocess
import sys
import tempfile

import groq_backend
from audio_boundaries import load_envelope
from bursts import default_cache_dir
from dtw_features import compute_features, slice_features, dtw_distance
from ffmpeg_util import find_ffmpeg
from local_alignment import subsequence_match

TRANSCRIBE_BACKEND = os.environ.get("TRANSCRIBE_BACKEND", "local").lower()

NGRAM_SIZE = 4
LONG_WORD_THRESHOLD_S = 1.2
LONG_WORD_MAX_CHARS = 10
SILENCE_GAP_THRESHOLD_S = 0.4
ACOUSTIC_WINDOW_MARGIN_S = 4.0       # how far around a flagged span to look for bursts
ACOUSTIC_MAX_DISTANCE = 0.25         # see find_repeats.py's calibration note
DEEP_SCAN_WINDOW_MARGIN_S = 15.0     # wider net for deep-scan query bursts when the narrow window is inconclusive
DEEP_SCAN_MAX_QUERY_FRACTION = 0.6   # a query must be meaningfully shorter than the haystack it's tested against


def _transcribe_local(path, model_size):
    from faster_whisper import WhisperModel
    print(f"[verify] transcribing {path} with {model_size} (local)...", file=sys.stderr)
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


def transcribe(path, model_size):
    """Backend-selectable, same pattern as chunk_transcribe.py/transcribe2.py
    (TRANSCRIBE_BACKEND=groq, falls back to local automatically). This was
    hardcoded to local faster-whisper for a long time after Groq was added
    everywhere else in the pipeline -- measured directly on a 70s rendered
    clip: local took 4 minutes, making it by far the single slowest step in
    the whole rough-cut + captions workflow, an order of magnitude past
    every other stage combined. Whether Groq's transcription is as reliable
    for *this specific job* (re-transcribing an already-cut, already-
    normalized output looking for repeats) as local large-v3 has not been
    separately re-validated -- watch the first few real runs after this
    change a bit more closely than usual."""
    if TRANSCRIBE_BACKEND == "groq":
        try:
            client = groq_backend.get_client()
            print(f"[verify] transcribing {path} with whisper-large-v3 (Groq)...", file=sys.stderr)
            with tempfile.TemporaryDirectory() as tmp_dir:
                segments = groq_backend.transcribe_whole_file(client, path, "pl", tmp_dir)
            words = []
            for seg in segments:
                for w in seg.get("words", []):
                    words.append({"word": w["word"].strip(), "start": w["start"], "end": w["end"]})
            return words
        except groq_backend.GroqUnavailable as e:
            print(f"[verify] TRANSCRIBE_BACKEND=groq requested but unavailable ({e}) -- falling back to local faster-whisper", file=sys.stderr)
    return _transcribe_local(path, model_size)


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


def _deep_scan_attempt(envelope, feats, times, span_start, span_end):
    """Second attempt for a span that came back INCONCLUSIVE from whole-burst
    comparison (typically because it's one long fused burst) -- widen the
    search for short candidate bursts nearby and try each as a subsequence-
    DTW query against the long burst covering the flagged span, exactly the
    deep-scan technique find_repeats.py uses. Returns (verdict, detail) or
    None if there's no long burst here to scan into."""
    wide_lo = span_start - DEEP_SCAN_WINDOW_MARGIN_S
    wide_hi = span_end + DEEP_SCAN_WINDOW_MARGIN_S
    wide_bursts = envelope.list_bursts(lo=wide_lo, hi=wide_hi)
    if not wide_bursts:
        return None

    # the haystack is whichever burst actually covers (or is closest to) the flagged span
    haystack = min(wide_bursts, key=lambda b: abs((b["start"] + b["end"]) / 2 - (span_start + span_end) / 2))
    h_feats = slice_features(feats, times, haystack["start"], haystack["end"])
    h_times = times[(times >= haystack["start"]) & (times < haystack["end"])]

    best = None
    for q in wide_bursts:
        if q is haystack or q["duration"] > haystack["duration"] * DEEP_SCAN_MAX_QUERY_FRACTION:
            continue
        q_feats = slice_features(feats, times, q["start"], q["end"])
        match = subsequence_match(q_feats, h_feats, h_times)
        if match is not None and (best is None or match["cost"] < best["match"]["cost"]):
            best = {"query": q, "match": match}

    if best is not None and best["match"]["cost"] <= ACOUSTIC_MAX_DISTANCE:
        q, m = best["query"], best["match"]
        return "CONFIRMED", (f"subsequence-DTW match: burst {q['start']:.2f}-{q['end']:.2f}s found again inside "
                              f"{haystack['start']:.2f}-{haystack['end']:.2f}s at {m['start']:.2f}-{m['end']:.2f}s, "
                              f"cost {m['cost']:.3f} (deep scan -- fused-burst case)")
    return None


def acoustic_check(envelope, feats, times, span_start, span_end):
    """Look for at least one pair of acoustically-similar bursts overlapping
    or near [span_start, span_end]. Returns (verdict, detail):
    verdict is "CONFIRMED" (a close match found nearby -- likely a real
    repeat, either a direct whole-burst DTW pair or a deep-scan subsequence
    match inside a fused burst), "NO_MATCH" (bursts exist here but nothing
    matches, including after a deep-scan attempt -- likely a decoder
    artifact, not real duplicated audio), or "INCONCLUSIVE" (no long burst
    nearby to even attempt a deep scan against -- this is now rare; it used
    to be the default outcome for any fused burst before the deep-scan
    fallback was added)."""
    lo = span_start - ACOUSTIC_WINDOW_MARGIN_S
    hi = span_end + ACOUSTIC_WINDOW_MARGIN_S
    local_bursts = envelope.list_bursts(lo=lo, hi=hi)

    best = None
    if len(local_bursts) >= 2:
        for i in range(len(local_bursts)):
            a = local_bursts[i]
            a_feats = slice_features(feats, times, a["start"], a["end"])
            for j in range(i + 1, len(local_bursts)):
                b = local_bursts[j]
                b_feats = slice_features(feats, times, b["start"], b["end"])
                dist = dtw_distance(a_feats, b_feats)
                if best is None or dist < best["distance"]:
                    best = {"a": a, "b": b, "distance": dist}

    if best is not None and best["distance"] <= ACOUSTIC_MAX_DISTANCE:
        return "CONFIRMED", (f"acoustic match: {best['a']['start']:.2f}-{best['a']['end']:.2f}s <-> "
                              f"{best['b']['start']:.2f}-{best['b']['end']:.2f}s, DTW distance {best['distance']:.3f}")

    deep = _deep_scan_attempt(envelope, feats, times, span_start, span_end)
    if deep is not None:
        return deep

    if len(local_bursts) < 2:
        return "INCONCLUSIVE", f"{len(local_bursts)} burst(s) in range and no nearby short burst matched into the long one via deep scan either"
    closest = f"{best['distance']:.3f}" if best else "n/a"
    return "NO_MATCH", f"no burst pair below distance {ACOUSTIC_MAX_DISTANCE} nearby, and deep scan into any fused burst found nothing either (closest: {closest}) -- likely a transcription artifact, not real duplicated audio"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_media")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--no-acoustic-check", action="store_true", help="skip the DTW cross-check (faster, text-only)")
    args = parser.parse_args()

    words = transcribe(args.output_media, args.model)
    full_text = " ".join(w["word"] for w in words)

    print("\n=== full transcript of rendered output ===")
    print(full_text)

    ngram_findings = find_repeated_ngrams(words)
    bloat_findings = find_bloated_words(words)

    envelope = None
    feats = times = None
    if not args.no_acoustic_check and (ngram_findings or bloat_findings):
        print("\n[verify] cross-checking flags against the audio itself (bursts + DTW)...", file=sys.stderr)
        envelope = load_envelope(args.output_media, default_cache_dir(args.output_media))
        feats, times = compute_features(args.output_media)

    print(f"\n=== repeated {NGRAM_SIZE}-word phrases ({len(ngram_findings)}) ===")
    if not ngram_findings:
        print("none found")
    for f in ngram_findings:
        line = f"  '{f['phrase']}'  first at {f['first_at']:.2f}s, again at {f['second_at']:.2f}s"
        if envelope is not None:
            verdict, detail = acoustic_check(envelope, feats, times, f["first_at"], f["second_at"])
            line += f"  -- [{verdict}] {detail}"
        else:
            line += " -- verify: missed duplicate, or genuine repeated rhetoric?"
        print(line)

    print(f"\n=== anomalously long single-word timestamps ({len(bloat_findings)}) ===")
    if not bloat_findings:
        print("none found")
    for f in bloat_findings:
        line = f"  '{f['word']}'  {f['start']:.2f}-{f['end']:.2f}s (dur {f['duration']:.2f}s)"
        if envelope is not None:
            sub_bursts = envelope.list_bursts(lo=f["start"] - 0.2, hi=f["end"] + 0.2)
            verdict, detail = acoustic_check(envelope, feats, times, f["start"], f["end"])
            line += f"  -- {len(sub_bursts)} sub-burst(s) inside this word's span; [{verdict}] {detail}"
        else:
            line += " -- likely hides collapsed repeated content, re-check this span in isolation"
        print(line)

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
        if envelope is not None:
            print("[verify] [CONFIRMED] = acoustic evidence of a real repeat nearby, treat as high-priority.")
            print("[verify] [NO_MATCH] = likely a transcription artifact (e.g. a decoder repetition loop), but still worth a quick listen.")
            print("[verify] [INCONCLUSIVE] = can't acoustically rule in or out (e.g. fused burst) -- this is where a human ear still matters most.")


if __name__ == "__main__":
    main()
