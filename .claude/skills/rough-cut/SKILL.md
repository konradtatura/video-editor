---
name: rough-cut
description: Turn a raw talking-head recording into a filler-free rough cut with word-level precision, using a cached-segment renderer for fast iteration. Use when the user has a raw video recording (talking head, interview, podcast) and wants filler words, dead air, and retakes removed automatically, keeping only the last/best take of each repeated line. Also use to re-render after editing a cutlist -- only changed segments re-encode.
---

# Rough Cut

Raw recording -> transcript -> editorial cutlist (read by Claude, not a rules engine) -> cached-segment render -> mandatory re-verification against the actual rendered file.

## Why this shape

Every step here exists because a simpler version of it failed during development, on real footage, in a reproducible way. The two failure classes that matter most:

1. **Word timestamps lie near pauses.** WhisperX/Whisper's forced alignment assigns unclaimed silent frames to the *last* word before a pause, so a word's `end` timestamp can be 0.3-1.5s later than the word actually stops being spoken. Cutting at the literal word-end timestamp bakes dead air into the clip. Fix: never trust ASR end-timestamps for a cut boundary -- measure the real audio energy instead (`audio_boundaries.py`).

2. **Whisper suppresses genuine repeats, sometimes completely.** When a speaker stutters or restarts a line, Whisper's decoder (trained on clean captions, and carrying an anti-hallucination heuristic that discards "too repetitive" output) can silently delete an entire repetition from the transcript text -- not mistime it, delete it, with zero trace. This got worse the smaller the model: `medium` missed repeats that `large-v3` caught on the identical audio. Chunking into short independently-decoded pieces helps but does **not** fully fix it, especially for rapid single-word stutters with no acoustic pause between repeats (verified: a 4x "jeśli chcesz" stutter with zero silence between attempts survived medium-model chunked transcription intact, and was only caught by re-running `large-v3` on the same audio).

**The practical consequence: no transcript, at any settings, is ground truth.** The only reliable check is re-transcribing the actual rendered output file and looking for what shouldn't be there. This is why `verify.py` exists and why it is not optional.

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
- **Cross-reference when something looks off.** If a number, name, or clause reads strangely, check the other transcript pass and/or re-transcribe that specific span in isolation with `large-v3` before trusting either. Several real errors in this pipeline's development were transcript-only artifacts invisible until an isolated re-check.
- **Anomalously long single-word timestamp = red flag.** A short word (e.g. "pomija.", "hajtiketowe") reported with a >1s duration usually means a repeated utterance got folded into that word's span instead of being transcribed. Re-check that span in isolation.
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

### 5. Verify -- mandatory, not optional

```
python scripts/verify.py projects/<name>/output.mp4
```

Re-transcribes the actual rendered file with `large-v3` (independent of whatever built the cutlist) and reports: repeated word n-grams, anomalously long single-word timestamps, and silence gaps over 0.4s. This is a "narrow down where to listen" tool, not a pass/fail gate -- it will flag genuine parallel rhetoric as a false positive (verified: it correctly flagged three real repeated phrases in a clean approved cut, all of which turned out to be intentional parallel sentence structure, not missed duplicates). Read every flag; do not treat a clean report as a substitute for actually watching the video.

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

- **`scripts/find_repeats.py`** (audio self-similarity / spectral repeat detector) was built as an ASR-independent safety net but **does not work reliably** -- it uses fixed-time-lag matching, which fails whenever a retake is spoken at a slightly different pace than the original (the normal case). It missed the single most obvious repeat in the test corpus. A correct version needs dynamic time warping (DTW), not implemented. Do not use its output to inform a cutlist; kept in the repo only as a starting point if this is revisited.
- No transcription pass, model size, or decode setting fully eliminates repeat-suppression. Treat every transcript as a lossy hint, and `verify.py` + actually watching the render as the only real check.
- This skill covers rough-cut only (dead air, fillers, retakes). Graphics, captions, and music are separate stages, not built here.

## File manifest

- `scripts/chunk_transcribe.py` -- primary transcription: chunks at measured real pauses, independent decode per chunk, `large-v3` default.
- `scripts/transcribe2.py` -- single full-file transcription pass, for cross-checking chunk_transcribe's output.
- `scripts/audio_boundaries.py` -- energy envelope + sustained-silence boundary snapping. Shared by `render.py` and `chunk_transcribe.py`.
- `scripts/render.py` -- cutlist -> boundary-snapped, hash-cached segments -> concat.
- `scripts/verify.py` -- automated post-render sanity check (repeated n-grams, bloated word timestamps, silence gaps).
- `scripts/locate.py` -- maps an output-relative timestamp back to the raw-footage time and cutlist entry it came from, for iterating on feedback given against the rendered video.
- `scripts/ffmpeg_util.py` -- shared ffmpeg binary resolution (PATH, with a winget-install fallback path).
- `scripts/find_repeats.py` -- experimental, not reliable, see Known limitations.
