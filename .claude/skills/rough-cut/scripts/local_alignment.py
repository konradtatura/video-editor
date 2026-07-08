"""
Subsequence (open-begin/open-end) DTW, for finding where a KNOWN short burst
best re-occurs inside a longer span -- this closes the blind spot in
find_repeats.py's whole-burst DTW comparison, which requires two repeats to
already be separated into distinct bursts by a sustained silence. A retake
spoken with no breath at all between attempts produces one long fused burst
that whole-burst DTW cannot decompose, since most of that burst is
different follow-on content and doesn't resemble any other single burst as
a whole (confirmed on real footage -- see SKILL.md).

Why this and not free-form Smith-Waterman local alignment (tried first,
abandoned): a from-scratch local aligner needs a hand-tuned gap penalty and
match-reward threshold, and the first attempt at that failed in a very
concrete, measurable way -- given a cheap enough gap cost, the DP found a
spurious "alignment" spanning nearly the entire 15.5s test span at only
0.42 average frame similarity (a genuine repeat measured 0.67-0.75 average
similarity along its true DTW path on the same footage), because it could
accumulate a large score by threading gaps through occasional coincidental
high-similarity frames rather than by genuinely matching content
throughout. Subsequence DTW has no such free parameter to mistune: it
reuses the exact same per-frame cosine-distance cost as the already-
calibrated whole-burst DTW (dtw_features.dtw_distance), just with a
different boundary condition -- the query (a short, already-known burst)
must be consumed in full and in order (this is still full ordinary DTW
down the query axis), but is allowed to *start* matching at any point along
the longer haystack for free, and the algorithm finds where along the
haystack that match ends up cheapest. This is the standard formulation used
for spoken-term/keyword detection in a longer recording.

Usage pattern: when bursts.py flags one burst as anomalously long right
after several short ones, treat each short burst as a "query" (a candidate
false-start attempt) and search for it inside the long burst as the
"haystack" -- a low-cost match near the start of the haystack is direct
evidence that attempt was repeated again right before the long burst's
different, unrelated continuation content.
"""

import numpy as np

DEFAULT_MAX_DURATION_RATIO = 2.5


def subsequence_match(query_feats, haystack_feats, haystack_times, max_duration_ratio=DEFAULT_MAX_DURATION_RATIO):
    """
    query_feats: (n, d) unit-norm features for a short, already-known burst.
    haystack_feats/haystack_times: (m, d) / (m,) features and frame times
    for the longer span to search within (e.g. a suspiciously long fused
    burst).

    Returns {"start", "end", "cost", "duration_ratio"} for the cheapest
    matching sub-window of the haystack, where "cost" is the normalized (by
    query length) DTW cost of matching the query against that sub-window --
    directly comparable to dtw_features.dtw_distance's output and the same
    calibrated thresholds apply (real repeats: 0.085-0.24; first false
    match between unrelated content: 0.25+).

    Returns None if either sequence is too short to align, OR if the
    cheapest match degenerates to a sub-window wildly shorter/longer than
    the query. This filter is required, not optional: confirmed directly
    that unconstrained open-begin/open-end DTW can "stall" on a single
    haystack frame, matching many query frames to it near-for-free when
    that one frame happens to resemble several query frames reasonably
    well -- this produced a false match at cost 0.227 (inside the
    calibrated true-repeat range) against a totally unrelated span, where
    the matched sub-window was 0.12s for a 0.61s query (ratio ~5x). A
    genuine retake of the same phrase does not compress that much even at
    very different speaking paces; reject anything past `max_duration_ratio`.
    """
    n, m = len(query_feats), len(haystack_feats)
    if n < 2 or m < 2:
        return None

    cost = 1.0 - query_feats @ haystack_feats.T  # (n, m) cosine distance

    INF = np.inf
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, :] = 0.0  # the query may start matching at any haystack column for free
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i - 1, j - 1]
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])

    end_j = int(np.argmin(D[n, 1:])) + 1  # cheapest ending column
    best_cost = D[n, end_j]

    # trace back to find the start column: walk the same recurrence choices
    # backwards until we hit a cell whose value came from the free row 0
    i, j = n, end_j
    while i > 1:
        candidates = [
            (D[i - 1, j - 1], i - 1, j - 1),
            (D[i - 1, j], i - 1, j),
            (D[i, j - 1], i, j - 1),
        ]
        _, i, j = min(candidates, key=lambda t: t[0])
    start_j = j

    matched_frames = max(1, end_j - start_j)
    duration_ratio = max(n, matched_frames) / min(n, matched_frames)
    if duration_ratio > max_duration_ratio:
        return None

    return {
        "start": float(haystack_times[max(0, start_j - 1)]),
        "end": float(haystack_times[min(len(haystack_times) - 1, end_j - 1)]),
        "cost": float(best_cost / n),
        "duration_ratio": float(duration_ratio),
    }
