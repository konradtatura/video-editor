---
name: rough-cut
description: Turn a raw talking-head recording into a filler-free rough cut with word-level precision, using a cached-segment renderer for fast iteration. Use when the user has a raw video recording (talking head, interview, podcast) and wants filler words, dead air, and retakes removed automatically, keeping only the last/best take of each repeated line. Also use to re-render after editing a cutlist -- only changed segments re-encode.
---

# Rough Cut

Raw recording -> transcript -> editorial cutlist (read by Claude, not a rules engine) -> cached-segment render -> mandatory re-verification against the actual rendered file.

## Why this shape

Every step here exists because a simpler version of it failed during development, on real footage, in a reproducible way. The two failure classes that matter most:

1. **Word timestamps lie near pauses.** WhisperX/Whisper's forced alignment assigns unclaimed silent frames to the *last* word before a pause, so a word's `end` timestamp can be 0.3-1.5s later than the word actually stops being spoken. Cutting at the literal word-end timestamp bakes dead air into the clip. Fix: never trust ASR end-timestamps for a cut boundary -- measure the real audio energy instead (`audio_boundaries.py`).

2. **Whisper suppresses genuine repeats, sometimes completely -- and this cuts both ways, which is the more dangerous direction.** When a speaker stutters or restarts a line, Whisper's decoder (trained on clean captions, and carrying an anti-hallucination heuristic that discards "too repetitive" output) can silently delete an entire repetition from the transcript text -- not mistime it, delete it, with zero trace. This got worse the smaller the model: `medium` missed repeats that `large-v3` caught on the identical audio. Chunking into short independently-decoded pieces helps but does **not** fully fix it, especially for rapid single-word stutters with no acoustic pause between repeats (verified: a 4x "jeśli chcesz" stutter with zero silence between attempts survived medium-model chunked transcription intact, and was only caught by re-running `large-v3` on the same audio). **Multiple independent transcription passes agreeing is not proof a repeat doesn't exist** -- confirmed on real footage: `chunk_transcribe.py`, `transcribe2.py`, and a fresh wide-context re-transcription all agreed on a single clean sentence where the audio actually had 3 false-start stutters folded in, because all three suffered the identical suppression on the identical audio. The other direction is just as real: **isolating a short (~1s) audio clip for re-transcription without surrounding context makes Whisper hallucinate plausible text instead of transcribing accurately** -- confirmed directly: two *different* ~1s bursts both "transcribed" to the identical phrase. Neither failure mode is rare or contrived; both happened in the same session on the same video. This is why acoustic (non-ASR) verification exists now -- see `bursts.py` / `find_repeats.py` below -- text-only cross-checking cannot catch either direction.

**The practical consequence: no transcript, at any settings, is ground truth, and neither is a short isolated re-transcription used to "double check" a suspicious span.** The reliable checks are (a) re-transcribing the actual rendered output file with full context and looking for what shouldn't be there, and (b) measuring the actual audio -- burst durations and DTW acoustic similarity -- which cannot hallucinate and cannot suppress. This is why `verify.py` exists and why it is not optional. **If the user directly reports something is wrong at a specific moment, that outranks every automated signal here** -- use it to locate the fix (`locate.py`), don't re-litigate whether the problem is real by throwing more transcription at it.

## Setup (one-time, per machine)

Needs: ffmpeg on PATH, and a Python environment with `faster-whisper`, `numpy`, `scipy`, and CPU-only `torch`.

**macOS**
```
brew install ffmpeg
python3 -m venv venv-whisper
source venv-whisper/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Windows**
```
winget install Gyan.FFmpeg   # restart the shell afterward so ffmpeg is on PATH
python -m venv venv-whisper
venv-whisper\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

No WSL, no Smart App Control workarounds needed -- this skill deliberately uses `faster_whisper` directly (not the full `whisperx` package) specifically because that avoids a pyannote-audio/`av` dependency that can trip Windows' Smart App Control. Should run unmodified on Linux too (same install pattern as macOS).

Run every script in this skill through that venv's Python (e.g. `venv-whisper/bin/python scripts/chunk_transcribe.py ...` on macOS/Linux, `venv-whisper\Scripts\python.exe scripts\chunk_transcribe.py ...` on Windows).

## Pipeline

```
projects/<name>/
  raw.mp4                 <- input
  transcript_a.json       <- chunk_transcribe.py output (primary)
  transcript_b.json       <- transcribe2.py output, full-file (cross-check source)
  cutlist.json            <- editorial decisions, written by Claude reading the transcript(s)
  segments/                <- cached rendered segments (hash-named) + cached audio envelope
  output.mp4              <- rendered result
```

### 1. Set up the project

```
mkdir -p projects/<name>/segments
cp <source video> projects/<name>/raw.mp4
```

### 2. Transcribe

Both scripts below support an optional Groq API backend (`TRANSCRIBE_BACKEND=groq`,
same `whisper-large-v3` model, seconds instead of minutes, audio leaves the machine)
alongside the default local `faster-whisper` path -- see the repo [README](../../../README.md#transcription-backend-local-vs-groq)
for setup, cost, and the local/cloud tradeoff. Falls back to local automatically if
`GROQ_API_KEY` is missing or a Groq call fails.

Primary pass (chunked, splits at real measured pauses, independent decode per chunk):

```
python scripts/chunk_transcribe.py projects/<name>/raw.mp4 projects/<name>/transcript_a.json --language pl --model large-v3
```

Cross-check pass (single full-file decode, different failure modes than chunking -- catches some things chunking drops, e.g. a rare word chunking split badly on):

```
python scripts/transcribe2.py projects/<name>/raw.mp4 projects/<name>/transcript_b.json --language pl --model large-v3
```

Always use `large-v3`. `medium` is faster but has been directly observed to silently drop repeats that `large-v3` catches on identical audio -- it is not safe for a final editorial decision, only for a quick throwaway preview if speed is more important than correctness.

Both scripts run natively on Windows via `faster_whisper` directly (not the full `whisperx` package) specifically to avoid a Smart App Control DLL block on `pyannote-audio`'s `av` dependency. No WSL required.

### 3. Build the cutlist -- Claude reads the transcript(s), not a script

This is deliberately not automated. Read the transcript text and decide what to keep. Rules that held up across a full development session:

- **Keep-last-take rule**: when a phrase/clause is restarted, keep only the final instance that *continues into new content*. An earlier restart can sound perfectly complete in isolation and still be the wrong one to keep -- check whether a later instance restates the same opening and continues further; if so, the earlier one is a discarded take, not a separate thought.
- **Cross-reference when something looks off -- but not by re-transcribing a short isolated clip.** If a number, name, or clause reads strangely, check the other transcript pass first. If you still need to re-check a specific span, prefer `scripts/bursts.py` + `scripts/find_repeats.py` (acoustic, cannot hallucinate) over re-transcribing a tight isolated cut of that span (confirmed to hallucinate matching-but-wrong text when given too little context -- see "Why this shape" above). If you must re-transcribe a span in isolation, use a wide window (10s+) with natural surrounding context, never a ~1s crop.
- **Anomalously long single-word timestamp = red flag.** A short word (e.g. "pomija.", "hajtiketowe") reported with a >1s duration usually means a repeated utterance got folded into that word's span instead of being transcribed. Confirm with `scripts/bursts.py --start <t-2> --end <t+3>` on the raw file -- a >1.2s "word" that's actually 2-4 separate acoustic bursts is the same signature `verify.py` looks for post-render (see step 5); catching it before the cutlist is written is cheaper than catching it after.
- **When unsure whether something is a genuine stutter/retake, run `scripts/find_repeats.py` on the raw file for that span before deciding.** It DTW-matches acoustic bursts against each other independent of what Whisper transcribed, so it catches exactly the cases repeat-suppression hides from the transcript. Calibrated on real footage: genuine repeats landed at DTW distance 0.085-0.24, and the first false match between two different (but similarly-structured, same-speaker) sentences was 0.25+ -- so a low distance is strong evidence, but read it as *ranked* evidence and cross-check the transcript context, not a hard yes/no. It also runs a **deep scan** by default: any burst notably longer than its neighbors gets treated as a haystack, and short bursts nearby get tried against it as subsequence-DTW queries (`local_alignment.py`) -- this specifically catches a retake whose final successful attempt is acoustically fused (no full sustained-silence gap) with the different content that follows it, which whole-burst comparison alone cannot see. Validated on real footage: this independently rediscovered a hidden 3rd retake attempt that the original manual edit had needed an isolated re-transcription to find. It is not a complete fix -- it only works when there's *some* separate short burst nearby to use as a query; if every attempt in a cluster is fused together with zero separation at all, `bursts.py`'s duration pattern (several short bursts, then one much longer one) is still the fallback tell.
- **Stock CTA/outro phrases mark a natural ending.** A phrase like "follow for more [value]" is a strong signal the speaker is wrapping up. If it appears more than once, that usually means the speaker re-recorded the whole ending with more material -- ask the user whether they want the first (shorter) or last (more complete) ending; do not assume "more complete = better" without checking, since the user may prefer the punchier first take.
- **Flag, don't silently decide, when a retake changes the actual content** (e.g. different numbers/stats between takes) -- that's an editorial call belonging to the user, not a filler-word judgment call. If the user says to be autonomous, apply the standard last-take rule and document the decision in the cutlist note so it's a one-line edit to reverse.
- Every `keep` entry's `note` field should say *why*, and which transcript source was trusted if there was a discrepancy. Cutlist notes are the debugging trail for the next review pass.

`cutlist.json` schema:
```json
{
  "source": "raw.mp4",
  "pad": 0,
  "keep": [
    {"start": 7.9, "end": 11.13, "note": "why this range, why not the earlier attempt, which transcript source"}
  ]
}
```
`pad` is a legacy fallback; leave at `0` -- boundary padding is handled by the snapping logic in `render.py`, not a fixed offset.

### 4. Render

```
python scripts/render.py projects/<name> [--out output.mp4]
```

For each kept clip: snaps `start`/`end` to the true voice boundary measured from the audio's own energy envelope (not the ASR timestamp), hashes the snapped range, re-encodes only on a cache miss, then concatenates. Editing one cut in `cutlist.json` and re-rendering only re-encodes that one segment -- this is the "10x faster" iteration loop.

Boundary snapping requires a *sustained* silence (>=150ms) before accepting it as a real boundary, specifically to avoid stopping on a brief stop-consonant or breath mid-phrase (this truncated a real word in early testing before the fix).

**Gotcha: nudging a `start`/`end` value by a small amount can silently do nothing.** `snap_start`/`snap_end` (in `audio_boundaries.py`) search a fixed tolerance window around the value you give (`start`: -0.3s to +1.0s; `end`: -2.0s to +0.3s) and grab the *first* real voice-boundary edge they find in it -- not the edge closest to your new value. If the edge you're trying to move away from is still inside that window, the snapped result -- and therefore the segment-cache hash -- comes out identical, `render.py` silently reuses the old cached segment, and your edit had zero effect even though the command reported success. Confirmed on real footage: moving a `start` from 109.6 to 110.0 produced byte-identical output both times. If a boundary fix doesn't seem to take effect after re-rendering, don't just nudge the value further and hope -- query the actual candidate edges directly:
```
python scripts/bursts.py projects/<name>/raw.mp4 --start <t-1> --end <t+1>
```
and pick a new `start`/`end` value that pushes the *old* edge outside the tolerance window (e.g. for `start`, the new value minus 0.3s must be past the old edge) so the snap is forced onto the next real one.

### 5. Verify -- mandatory, not optional

```
python scripts/verify.py projects/<name>/output.mp4
```

Re-transcribes the actual rendered file with `large-v3` (independent of whatever built the cutlist) and reports: repeated word n-grams, anomalously long single-word timestamps, and silence gaps over 0.4s. This is a "narrow down where to listen" tool, not a pass/fail gate -- it will flag genuine parallel rhetoric as a false positive (verified: it correctly flagged three real repeated phrases in a clean approved cut, all of which turned out to be intentional parallel sentence structure, not missed duplicates).

**Every text-level flag is now cross-checked acoustically** (via `bursts.py` + `find_repeats.py`'s DTW comparison, run automatically unless `--no-acoustic-check` is passed) and labeled:
- `[CONFIRMED]` -- a close acoustic match exists nearby (either a direct whole-burst DTW pair, or a deep-scan subsequence match into a fused burst); treat as high-priority, this is real audio, not just repeated text. Note this confirms the *acoustic* question (was this really said twice), not the *editorial* one (mistake vs. intentional device) -- confirmed on real footage: a genuine `[CONFIRMED]` case turned out to be deliberate parallel rhetoric ("your ads will work if you don't have X... but will work much better if you have X"), correctly identified as a real repeat, still requiring a human/Claude judgment on whether it's a keeper.
- `[NO_MATCH]` -- no close acoustic match nearby, including after a deep-scan attempt; more likely a transcription artifact than a real duplicate. Confirmed necessary on real footage: a Whisper decoder repetition-loop produced 10 identical-looking text-repeat flags for a single clean sentence, and the acoustic check would have correctly dismissed all 10 (the audio waveform at that point showed no anomaly at all). Also confirmed correctly dismissing a genuine case of the same *term* (not the same *moment*) reused across unrelated sentences -- e.g. a recurring acronym central to the video's topic -- since that's expected vocabulary reuse, not a duplicated take.
- `[INCONCLUSIVE]` -- no long burst nearby to even attempt a deep scan against. This should now be rare (the deep-scan fallback handles most fused-burst cases); when it does happen, a human ear still matters most.

Even a `[NO_MATCH]` or a clean report is not proof of anything by itself -- see "Why this shape" above on why neither ASR nor acoustic matching alone is ground truth. Read every flag; do not treat a clean report as a substitute for actually watching the video, and never let any of these automated signals override a specific problem the user reports.

## Iterating after review

`cutlist.json` is the editable source of truth, and `render.py` only re-encodes changed segments -- this is what makes the review loop fast. When the user gives feedback:

- **If they give a raw-footage timestamp** (rare -- usually only if they're looking at the transcript), edit the matching `cutlist.json` entry directly.
- **If they give an output-relative timestamp** ("the pause at 0:45 is too long", "cut more around 1:12 in the video"), that time is in the *edited* video, not the raw source -- `cutlist.json` only understands raw-footage time. Run:
  ```
  python scripts/locate.py projects/<name> <output_time_seconds>
  ```
  This recomputes the same snapped boundaries `render.py` uses (so it matches what's actually in `output.mp4`, not the unsnapped cutlist values) and reports which `keep` entry the time falls in, plus the exact corresponding raw-footage time to edit.
- After editing, re-run `render.py`. Re-run `verify.py` afterward too -- a manual single-clip edit can introduce a new issue (e.g. a boundary that now clips a word) just as easily as the original cutlist construction could.

## Known limitations

- **`scripts/find_repeats.py`'s whole-burst DTW comparison has one blind spot, now mostly closed by a deep-scan pass, not fully.** If a retake's final successful attempt is acoustically fused (no full sustained-silence gap) into the same burst as the different content that follows it, the earlier false-start bursts won't DTW-match well against that one long fused burst -- most of it is different content. `deep_scan_long_bursts` (in `find_repeats.py`, using `local_alignment.subsequence_match`) treats any burst notably longer than its neighbors as a haystack and tries nearby short bursts as subsequence-DTW queries against it, which finds a repeat *inside* the long burst without needing the whole thing to match. Validated on real footage across two different projects: correctly found a hidden repeat inside a 15.5s fused burst that whole-burst DTW couldn't see, and independently rediscovered a hidden 3rd retake attempt that the original manual edit had needed an isolated re-transcription to catch. Remaining gap: this only works when there's some separate short burst nearby to use as a query -- if an entire cluster of repeats has zero separation anywhere (no burst boundary at all to anchor a query on), neither pass can find it, and `bursts.py`'s raw duration pattern is the only remaining signal.
  - A from-scratch Smith-Waterman-style local aligner was tried first for the deep-scan pass and abandoned: with a workable gap penalty it produced a spurious "alignment" spanning nearly an entire 15.5s span at only 0.42 average frame similarity (real repeats measured 0.67-0.75), because it could thread cheap gaps through occasional coincidental high-similarity frames without matching content throughout. Subsequence (open-begin/open-end) DTW was used instead -- no free gap-penalty parameter, reuses the same calibrated per-frame cost as whole-burst DTW -- but has its own failure mode: unconstrained, it can "stall" on a single haystack frame and match many query frames to it almost for free (confirmed: a false match at cost 0.227, inside the calibrated true-repeat range, where the matched window was ~5x shorter than the query). `subsequence_match` rejects any match whose duration ratio to the query exceeds 2.5x; this filter is required, not optional.
- DTW distance (whole-burst or subsequence) is *ranked* evidence, not a clean binary. Same-speaker, similarly-structured-but-different sentences can land close to the calibrated true-repeat range (observed floor: 0.25) -- a low distance is strong support, but cross-check the transcript/context before treating any single number as a verdict. Separately: a low distance confirms the audio really was repeated, not that repeating it was a *mistake* -- deliberate parallel rhetoric acoustically looks exactly like an accidental retake, and that distinction is still an editorial judgment call, not something these tools can make.
- **`find_repeats.py`'s default `--window` is 20s, tightened down from an initial 60s after testing on a full ~168s video** -- a real retake is always spoken within seconds of the original, and the wider window let in same-speaker/same-genre coincidental noise from bursts 50-100+ seconds apart without adding any true positives. If a retake genuinely happens further apart than 20s in some footage, widen `--window` for that specific check.
- No transcription pass, model size, or decode setting fully eliminates repeat-suppression -- and this isn't limited to short/degraded audio; it was confirmed on a full natural-context, pause-chunked pass too, with multiple independent transcription methods agreeing on the same wrong (over-suppressed) answer. Treat every transcript as a lossy hint. Acoustic checks (`bursts.py`, `find_repeats.py`, `verify.py`'s cross-check) are a second, independent line of evidence precisely because they cannot suppress or hallucinate the way a decoder can -- but actually watching the render is still the only complete check.
- Isolating a short (~1s) audio clip to re-transcribe it "for confirmation" is unreliable in both directions -- it can hallucinate matching text for genuinely different audio, and it provides no more protection against repeat-suppression than a full-context pass. Don't use it as evidence either way; use `bursts.py`/`find_repeats.py` instead for acoustic questions, and a wide-context (10s+) re-transcription if a text question specifically needs re-checking.
- This skill covers rough-cut only (dead air, fillers, retakes). Graphics, captions, and music are separate stages, not built here.

## File manifest

- `scripts/chunk_transcribe.py` -- primary transcription: chunks at measured real pauses, independent decode per chunk, `large-v3` default.
- `scripts/transcribe2.py` -- single full-file transcription pass, for cross-checking chunk_transcribe's output.
- `scripts/audio_boundaries.py` -- energy envelope + sustained-silence boundary snapping (`VoiceEnvelope.snap_start`/`snap_end`), plus acoustic burst segmentation (`VoiceEnvelope.list_bursts`). Shared by `render.py`, `chunk_transcribe.py`, `bursts.py`, and `find_repeats.py`.
- `scripts/bursts.py` -- CLI: prints/writes the acoustic burst inventory (start/end/duration of every voiced span) for any media file, raw or rendered. ASR-independent; a stutter/retake shows up here as a duration pattern even when Whisper's transcript hides it.
- `scripts/dtw_features.py` -- shared log-mel feature extraction + normalized whole-sequence DTW distance, used by `find_repeats.py`'s whole-burst pass and `verify.py`'s acoustic cross-check.
- `scripts/local_alignment.py` -- subsequence (open-begin/open-end) DTW: finds where a known short burst best re-occurs inside a longer span, without needing the whole span to match. Powers the deep-scan pass in `find_repeats.py` and `verify.py`'s fused-burst fallback.
- `scripts/find_repeats.py` -- ASR-independent repeat detector, two passes: whole-burst DTW (tempo-invariant, unlike the original fixed-time-lag v1) for cleanly-separated retakes, plus a deep scan (`local_alignment.py`) for retakes fused into a longer burst with no silence gap. Calibrated on real footage: see Known limitations for thresholds and remaining blind spot.
- `scripts/render.py` -- cutlist -> boundary-snapped, hash-cached segments -> concat.
- `scripts/verify.py` -- automated post-render sanity check: repeated n-grams, bloated word timestamps, and silence gaps, each cross-checked against `bursts.py`/`find_repeats.py`'s acoustic evidence (including the deep-scan fallback) and labeled CONFIRMED/NO_MATCH/INCONCLUSIVE.
- `scripts/locate.py` -- maps an output-relative timestamp back to the raw-footage time and cutlist entry it came from, for iterating on feedback given against the rendered video.
- `scripts/ffmpeg_util.py` -- shared ffmpeg binary resolution (PATH, with a winget-install fallback path).
- `scripts/groq_backend.py` -- optional cloud transcription backend (`TRANSCRIBE_BACKEND=groq`): extracts audio, calls Groq's `whisper-large-v3` API, normalizes the response to this pipeline's segment/word schema. See repo README for setup.
