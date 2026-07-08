"""
ASR-independent repeat detector: finds pairs of acoustic bursts that closely
resemble each other, using DTW (dynamic time warping) over log-mel features
-- not Whisper text. This is the safety net for the fact that no amount of
decode-parameter tuning or chunking fully stops Whisper from occasionally
deleting or mis-rendering genuinely repeated speech (see project notes):
repeats found here should be cross-checked against the transcript, and
anything the transcript missed/collapsed is a direct signal of a
transcription gap, not an editorial one.

Two complementary passes:

1. **Whole-burst DTW** (`find_candidate_pairs`): segment audio into voiced
   bursts (audio_boundaries.VoiceEnvelope.list_bursts -- the same energy
   envelope render.py already computes), then DTW-compare every pair of
   bursts within --window seconds of each other. Catches cleanly-separated
   retakes reliably, tempo-invariant by construction (unlike the original
   v1's fixed-time-lag matching, which required frame-for-frame alignment
   and missed the normal case where a retake is spoken at a different pace
   -- verified directly on real footage).

   Calibration note (from real data, not a guess): on a ~50s stretch
   containing 5 known genuine repeat pairs and several genuinely-different
   sentences from the same speaker, the 5 true repeats landed at DTW
   distance 0.085-0.242, and the first false positive (two different
   sentences, same speaker/genre) was 0.251. DEFAULT_MAX_DISTANCE is set
   just under that gap. This is a real gap, not a clean margin -- same-
   speaker prosody on similarly-structured sentences can land close to
   genuine repeats, so treat distance as *ranked evidence*, not a hard
   pass/fail signal.

   DEFAULT_WINDOW_S was tightened from an initial 60s to 20s after testing
   on a full ~168s video: a real retake is always local (spoken seconds
   after the original, not 50-100s later), and the wider window let in a
   lot of coincidental same-speaker-same-genre noise from distant,
   unrelated bursts without adding any true positives -- narrowing it to
   20s cut the whole-burst pair count roughly in half on that file while
   every previously-confirmed true match stayed found.

2. **Deep scan / subsequence search** (`deep_scan_long_bursts`): closes the
   whole-burst pass's real blind spot -- a retake spoken with zero breath
   between attempts fuses into one long burst together with whatever
   different content follows it, and that long fused burst won't DTW-match
   any other single burst as a whole (most of it is unrelated content).
   Confirmed directly on real footage: three documented false-start
   retakes fused into one 15.5s burst. This pass targets exactly that
   shape -- for any burst notably longer than its neighbors, treat each
   short burst nearby as a "query" and search for it *inside* the long one
   using open-begin/open-end subsequence DTW (local_alignment.py), which
   finds where in the long burst that shorter phrase best re-occurs without
   needing the whole burst to match.

   A from-scratch Smith-Waterman-style local aligner was tried first for
   this and abandoned: with a cheap enough gap penalty it found a spurious
   "alignment" spanning nearly an entire 15.5s span at only 0.42 average
   frame similarity (real repeats measured 0.67-0.75), because it could
   thread gaps through occasional coincidental high-similarity frames
   without genuinely matching content throughout. Subsequence DTW has no
   such free gap-penalty parameter -- it reuses the same calibrated
   per-frame cost as the whole-burst pass. It has its own failure mode
   (unconstrained open-begin/open-end DTW can "stall" on a single haystack
   frame and match many query frames to it almost for free -- confirmed:
   a false match at cost 0.227, inside the calibrated true-repeat range,
   against totally unrelated content, where the matched window was 5x
   shorter than the query), which is why `local_alignment.subsequence_match`
   rejects any match whose duration ratio to the query exceeds
   `DEFAULT_MAX_DURATION_RATIO`.

Works on any media file directly -- a project's raw.mp4, or an already-
rendered output.mp4 (e.g. what verify.py checks) -- not just rough-cut
projects.

Usage:
    python find_repeats.py <media_file> [--cache-dir DIR] [--start S] [--end E]
        [--window W] [--max-distance D] [--no-deep-scan]
"""

import argparse
import json
import os
import statistics
import sys

from audio_boundaries import load_envelope
from bursts import default_cache_dir
from dtw_features import compute_features, slice_features, dtw_distance
from local_alignment import subsequence_match

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_WINDOW_S = 20.0
DEFAULT_MAX_DISTANCE = 0.25
MIN_BURST_DURATION_S = 0.15  # ignore tiny blips/breaths, not real syllables

# a burst qualifies as a deep-scan "haystack" if it's this many times longer
# than the median burst in the file, and at least this many seconds long --
# both conditions guard against triggering on a video that's just generally
# long-winded (e.g. one continuous 8s sentence isn't "anomalous" on its own)
DEEP_SCAN_MEDIAN_RATIO = 2.5
DEEP_SCAN_MIN_DURATION_S = 3.0
# how many short bursts immediately before a long one to try as queries
DEEP_SCAN_MAX_QUERIES = 4
# a query must be no more than this fraction of the haystack's own duration
# -- otherwise it's not really a "short burst before the long one", it's
# comparable in size and belongs to the whole-burst pass instead
DEEP_SCAN_MAX_QUERY_FRACTION = 0.6


def find_candidate_pairs(bursts, feats, times, window_s, max_distance):
    candidates = []
    n = len(bursts)
    for i in range(n):
        a = bursts[i]
        if a["duration"] < MIN_BURST_DURATION_S:
            continue
        a_feats = slice_features(feats, times, a["start"], a["end"])
        for j in range(i + 1, n):
            b = bursts[j]
            if b["start"] - a["end"] > window_s:
                break
            if b["duration"] < MIN_BURST_DURATION_S:
                continue
            b_feats = slice_features(feats, times, b["start"], b["end"])
            dist = dtw_distance(a_feats, b_feats)
            if dist <= max_distance:
                dur_ratio = max(a["duration"], b["duration"]) / max(1e-6, min(a["duration"], b["duration"]))
                candidates.append({
                    "a_start": a["start"], "a_end": a["end"], "a_duration": a["duration"],
                    "b_start": b["start"], "b_end": b["end"], "b_duration": b["duration"],
                    "distance": dist, "duration_ratio": dur_ratio,
                })
    candidates.sort(key=lambda c: c["distance"])
    return candidates


def deep_scan_long_bursts(bursts, feats, times, window_s, max_distance):
    if len(bursts) < 2:
        return []
    median_dur = statistics.median(b["duration"] for b in bursts)
    threshold = max(DEEP_SCAN_MIN_DURATION_S, median_dur * DEEP_SCAN_MEDIAN_RATIO)

    results = []
    for idx, haystack in enumerate(bursts):
        if haystack["duration"] < threshold:
            continue
        h_feats = slice_features(feats, times, haystack["start"], haystack["end"])
        h_times = times[(times >= haystack["start"]) & (times < haystack["end"])]

        queried = 0
        for k in range(idx - 1, -1, -1):
            if queried >= DEEP_SCAN_MAX_QUERIES:
                break
            query = bursts[k]
            if haystack["start"] - query["end"] > window_s:
                break
            if query["duration"] < MIN_BURST_DURATION_S:
                continue
            if query["duration"] > haystack["duration"] * DEEP_SCAN_MAX_QUERY_FRACTION:
                continue
            queried += 1
            q_feats = slice_features(feats, times, query["start"], query["end"])
            match = subsequence_match(q_feats, h_feats, h_times)
            if match is not None and match["cost"] <= max_distance:
                results.append({
                    "query_start": query["start"], "query_end": query["end"], "query_duration": query["duration"],
                    "haystack_start": haystack["start"], "haystack_end": haystack["end"],
                    "match_start": match["start"], "match_end": match["end"],
                    "cost": match["cost"], "duration_ratio": match["duration_ratio"],
                })
    results.sort(key=lambda r: r["cost"])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media_file")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--start", type=float, default=None, help="restrict to this time range")
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                         help="only compare burst pairs within this many seconds of each other (a retake is always local, not a global search)")
    parser.add_argument("--max-distance", type=float, default=DEFAULT_MAX_DISTANCE,
                         help="DTW / subsequence-DTW cost threshold below which a match is flagged (lower = stricter)")
    parser.add_argument("--no-deep-scan", action="store_true",
                         help="skip the subsequence search inside anomalously long bursts (faster, whole-burst DTW only)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    media_path = os.path.abspath(args.media_file)
    cache_dir = args.cache_dir or default_cache_dir(media_path)

    print("[find_repeats] loading voice envelope...", file=sys.stderr)
    envelope = load_envelope(media_path, cache_dir)
    bursts = envelope.list_bursts(lo=args.start, hi=args.end)
    print(f"[find_repeats] {len(bursts)} voiced bursts in range", file=sys.stderr)

    print("[find_repeats] extracting spectral features (once, whole file)...", file=sys.stderr)
    feats, times = compute_features(media_path)

    print(f"[find_repeats] comparing burst pairs within {args.window}s of each other via DTW...", file=sys.stderr)
    candidates = find_candidate_pairs(bursts, feats, times, args.window, args.max_distance)

    print(f"\n[find_repeats] {len(candidates)} acoustically-similar burst pair(s) (DTW distance <= {args.max_distance})")
    for c in candidates:
        print(f"  {c['a_start']:8.3f}-{c['a_end']:8.3f} (dur {c['a_duration']:.2f}s)  <->  "
              f"{c['b_start']:8.3f}-{c['b_end']:8.3f} (dur {c['b_duration']:.2f}s)  "
              f"dist={c['distance']:.3f}  dur_ratio={c['duration_ratio']:.2f}")

    deep_results = []
    if not args.no_deep_scan:
        print(f"\n[find_repeats] deep-scanning anomalously long bursts for fused/hidden repeats...", file=sys.stderr)
        deep_results = deep_scan_long_bursts(bursts, feats, times, args.window, args.max_distance)
        print(f"\n[find_repeats] {len(deep_results)} hidden repeat(s) found inside long fused bursts (subsequence DTW)")
        for r in deep_results:
            print(f"  query {r['query_start']:8.3f}-{r['query_end']:8.3f} (dur {r['query_duration']:.2f}s)  found inside "
                  f"haystack {r['haystack_start']:8.3f}-{r['haystack_end']:8.3f}  at  "
                  f"{r['match_start']:8.3f}-{r['match_end']:8.3f}  cost={r['cost']:.3f}  dur_ratio={r['duration_ratio']:.2f}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"whole_burst_pairs": candidates, "deep_scan_matches": deep_results}, f, indent=2)
        print(f"[find_repeats] wrote {args.json_out}")

    if candidates or deep_results:
        print("\n[find_repeats] cross-check each finding against the transcript -- if the transcript "
              "only shows this phrase once, that's a Whisper repeat-suppression case, not a false alarm.")
    else:
        print("\n[find_repeats] no acoustically-similar bursts or hidden fused repeats found in range.")


if __name__ == "__main__":
    main()
