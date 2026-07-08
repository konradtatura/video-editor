"""
Groq-hosted whisper-large-v3 transcription backend -- an alternative to the
local faster-whisper path, selected via TRANSCRIBE_BACKEND=groq. Same model
architecture as the local `large-v3` default, run on Groq's hardware, returns
in seconds instead of minutes -- at the cost of audio leaving the machine.

Output is normalized to the exact same segment schema chunk_transcribe.py/
transcribe2.py already produce from faster-whisper:
    [{"start", "end", "text", "words": [{"word","start","end","score"}],
      "avg_logprob", "compression_ratio"}, ...]
so nothing downstream (cutlist building, repeat detection, captions) needs
to know which backend ran.

Groq's API caps uploads at 25MB. We only ever send extracted mono 16kHz
audio (never the source video), which keeps almost every real clip well
under that limit; the rare oversized case (a very long recording) falls
back to a lower bitrate, then to splitting into fixed-size chunks sent as
separate sequential requests.

Groq's word-level timestamps (OpenAI-compatible verbose_json format) don't
carry a per-word confidence score the way faster-whisper's probability
does -- there is nothing to substitute it with, so every Groq-sourced word
gets score=1.0. Downstream logic that flags low-confidence words (e.g.
captions' make_captions.py) will not fire for Groq transcripts; this is a
known gap, not a bug.
"""

import json
import os
import subprocess
import sys

from ffmpeg_util import find_ffmpeg

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
GROQ_MODEL = "whisper-large-v3"  # not turbo -- we want accuracy parity with local large-v3


class GroqUnavailable(Exception):
    """TRANSCRIBE_BACKEND=groq was requested but can't be honored right now
    (missing key, missing package, or an API-level failure such as an
    invalid key or rate limit) -- caller should fall back to local."""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise GroqUnavailable("GROQ_API_KEY is not set")
    try:
        from groq import Groq
    except ImportError:
        raise GroqUnavailable("the `groq` package is not installed (pip install groq)")
    return Groq(api_key=api_key)


def extract_audio(input_media, output_path, bitrate="64k"):
    """Whole-file audio extraction, mono 16kHz, per Groq's recommended
    preprocessing (keeps upload size well under the 25MB API limit)."""
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-i", input_media,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate,
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def extract_audio_range(input_media, output_path, start, end, bitrate="64k"):
    """Same as extract_audio but for one time range -- used for per-chunk
    extraction (chunk_transcribe.py) and for the fixed-size-chunk fallback
    when a whole file's audio is still too large after a bitrate drop."""
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", input_media, "-vn", "-ac", "1", "-ar", "16000", "-b:a", bitrate,
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def transcribe_audio_file(client, audio_path, language):
    """Sends one (already-small) audio file to Groq and returns the raw
    verbose_json response as a plain dict. Never logs the API key -- the
    client already holds it, nothing here touches it directly."""
    try:
        with open(audio_path, "rb") as f:
            response = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model=GROQ_MODEL,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
            )
    except Exception as e:
        # Covers auth failures, rate limits, network errors -- all treated
        # as "Groq isn't available right now", not a crash.
        raise GroqUnavailable(f"Groq API call failed: {type(e).__name__}: {e}")

    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return json.loads(response.json())


def _assign_words_to_segments(segments, words):
    """Groq/OpenAI verbose_json returns `words` as one flat top-level list,
    not nested under each segment the way faster-whisper's API does -- so
    we re-nest them by matching each word's start time against consecutive
    segment boundaries (both lists are chronological)."""
    result = [[] for _ in segments]
    wi = 0
    n = len(words)
    for i in range(len(segments)):
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else float("inf")
        while wi < n and words[wi]["start"] < next_start:
            result[i].append(words[wi])
            wi += 1
    if wi < n and segments:
        result[-1].extend(words[wi:])
    return result


def normalize_groq_response(raw, time_offset=0.0):
    """Groq's verbose_json -> this pipeline's segment schema. `time_offset`
    shifts a chunk's local timestamps back into the source file's global
    timeline (mirrors chunk_transcribe.py's own `+ c_start` offsetting)."""
    raw_segments = sorted(raw.get("segments", []), key=lambda s: s["start"])
    raw_words = sorted(raw.get("words", []), key=lambda w: w["start"])
    words_by_segment = _assign_words_to_segments(raw_segments, raw_words)

    # time_offset is often a numpy.float32 (chunk_transcribe.py's chunk
    # boundaries come from a numpy times array) -- numpy_scalar + python_float
    # silently produces another numpy scalar, which json.dumps then rejects.
    # Cast everything back to native float/str before it leaves this function.
    time_offset = float(time_offset)

    out_segments = []
    for seg, seg_words in zip(raw_segments, words_by_segment):
        out_segments.append({
            "start": float(seg["start"]) + time_offset,
            "end": float(seg["end"]) + time_offset,
            "text": seg.get("text", ""),
            "words": [
                {
                    "word": w["word"].strip(),
                    "start": float(w["start"]) + time_offset,
                    "end": float(w["end"]) + time_offset,
                    "score": 1.0,
                }
                for w in seg_words
            ],
            "avg_logprob": float(seg.get("avg_logprob", 0.0)),
            "compression_ratio": float(seg.get("compression_ratio", 0.0)),
        })
    return out_segments


def transcribe_chunk(client, input_media, start, end, language, tmp_dir):
    """Extracts one time range as audio and transcribes it via Groq,
    returning segments already offset into the input file's global time."""
    audio_path = os.path.join(tmp_dir, f"groq_chunk_{start:.3f}_{end:.3f}.mp3")
    extract_audio_range(input_media, audio_path, start, end)
    raw = transcribe_audio_file(client, audio_path, language)
    return normalize_groq_response(raw, time_offset=start)


def _probe_duration(input_media):
    ffprobe = find_ffmpeg("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", input_media],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def transcribe_whole_file(client, input_media, language, tmp_dir):
    """Whole-file transcription via Groq, for single-pass callers
    (transcribe2.py, make_captions.py). Extracts full audio at 64kbps;
    if that's still over Groq's 25MB cap, retries at 32kbps; if still too
    big (very long recordings only), falls back to splitting into
    fixed-size sequential chunks -- simpler than reusing
    chunk_transcribe.py's pause-based splitting since this path only
    triggers on rare oversized files, not the common case."""
    audio_path = os.path.join(tmp_dir, "groq_audio_64k.mp3")
    extract_audio(input_media, audio_path, bitrate="64k")
    size = os.path.getsize(audio_path)

    if size > MAX_UPLOAD_BYTES:
        print("[groq_backend] 64kbps audio exceeds 25MB, retrying at 32kbps", file=sys.stderr)
        audio_path = os.path.join(tmp_dir, "groq_audio_32k.mp3")
        extract_audio(input_media, audio_path, bitrate="32k")
        size = os.path.getsize(audio_path)

    if size <= MAX_UPLOAD_BYTES:
        raw = transcribe_audio_file(client, audio_path, language)
        return normalize_groq_response(raw, time_offset=0.0)

    print("[groq_backend] audio still exceeds 25MB at 32kbps, splitting into fixed-size chunks", file=sys.stderr)
    total_duration = _probe_duration(input_media)
    bytes_per_second = size / total_duration
    chunk_seconds = max(60.0, (MAX_UPLOAD_BYTES * 0.9) / bytes_per_second)

    all_segments = []
    pos = 0.0
    while pos < total_duration:
        c_end = min(pos + chunk_seconds, total_duration)
        all_segments.extend(transcribe_chunk(client, input_media, pos, c_end, language, tmp_dir))
        pos = c_end
    return all_segments
