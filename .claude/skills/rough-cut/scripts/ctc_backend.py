"""
CTC (wav2vec2) transcription -- a second, architecturally-different source
of truth alongside the primary Whisper transcript, specifically to catch
repeats Whisper's decoder silently fuses into one word or one clause.

Why this exists: Whisper (local or Groq) is an autoregressive decoder with
a *trained* anti-repetition bias -- verified repeatedly on real footage to
silently delete or fuse genuine repeats, with no API knob to disable it
(checked directly against Groq's SDK surface: no `repetition_penalty`,
`no_repeat_ngram_size`, `condition_on_previous_text`, or similar exposed).
Chunking only helps when there's a real acoustic pause to split on --
verified to do nothing for a genuinely gapless restart (no silence between
attempts, so no chunk boundary can land there either).

CTC has no equivalent bias: it emits one token per audio frame independently
and then collapses repeats via the standard CTC decode rule (merge adjacent
identical tokens, drop blanks) -- there is no learned suppression of
*genuinely repeated speech*, because two separate utterances of the same
word are separated by at least one blank/differing frame in between, so the
CTC collapse rule does not merge them into one. Confirmed directly on real
footage: this model recovered both a 2x fused repeat and a 3x fused repeat
that Whisper (via Groq) had collapsed to one instance each, with a clean
negative on a genuine non-repeated control span.

This is deliberately NOT a transcript source -- the output is phonetically
rough (a small multilingual wav2vec2 model, not fine-tuned per-speaker) and
should never be read for wording, only for repeat *structure*. Approximate
word-level timestamps are reconstructed from CTC's own frame-level output
(each output frame is a fixed ~20ms of audio); they are good enough to
localize a repeat to within a second or so, not to snap a cut boundary --
use audio_boundaries.py for that, same as everywhere else in this skill.

Usage as a library:
    from ctc_backend import transcribe_words
    words = transcribe_words(media_path, start=10.0, end=40.0)
    # -> [{"word": "tłumaczysz", "start": 12.34, "end": 12.71}, ...]
"""

import os
import subprocess
import sys

import numpy as np

from ffmpeg_util import find_ffmpeg

MODEL_NAME = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
SR = 16000

_model = None
_processor = None


def get_model():
    """Lazy singleton -- loading is the expensive part (~1.2GB download the
    first time, then a few seconds to initialize), do it once per process."""
    global _model, _processor
    if _model is None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        print(f"[ctc_backend] loading {MODEL_NAME}...", file=sys.stderr)
        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        _model.eval()
    return _model, _processor


def _extract_pcm(media_path, wav_path, start=None, end=None):
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", media_path, "-ac", "1", "-ar", str(SR), wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg PCM extraction failed:\n{result.stderr[-2000:]}")


def _load_wav_mono(wav_path):
    import wave
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # peak-normalize in-memory -- confirmed necessary: an ffmpeg-side dB
    # boost on already-quiet audio distorted the signal and produced
    # garbage output; normalizing the float array directly (no re-encode)
    # does not have this problem
    peak = np.abs(data).max()
    if peak > 1e-6:
        data = data / peak * 0.95
    return data


def transcribe_words(media_path, start=None, end=None, time_offset=None):
    """
    Runs CTC over [start, end] (or the whole file if omitted) and returns
    word-level {"word", "start", "end"} dicts with approximate timestamps
    in the SOURCE file's absolute time (start is added back in automatically
    unless time_offset is given explicitly, e.g. when media_path is already
    an isolated extract and times should be relative to it).
    """
    model, processor = get_model()
    import torch

    offset = start if (time_offset is None and start is not None) else (time_offset or 0.0)

    wav_path = media_path + f"._ctc_tmp_{os.getpid()}.wav"
    _extract_pcm(media_path, wav_path, start, end)
    try:
        data = _load_wav_mono(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    if len(data) < SR * 0.1:
        return []

    duration_s = len(data) / SR
    inputs = processor(data, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits[0]
    pred_ids = torch.argmax(logits, dim=-1).numpy()

    num_frames = len(pred_ids)
    frame_dur = duration_s / num_frames if num_frames else 0.0

    pad_id = processor.tokenizer.pad_token_id
    delim_id = processor.tokenizer.convert_tokens_to_ids("|")

    # standard CTC collapse: merge consecutive identical token ids, track
    # the frame span each collapsed unit occupied (for approximate timing)
    collapsed = []
    prev = None
    seg_start_frame = 0
    for i, tid in enumerate(pred_ids):
        tid = int(tid)
        if tid != prev:
            if prev is not None:
                collapsed.append((prev, seg_start_frame, i))
            seg_start_frame = i
            prev = tid
    if prev is not None:
        collapsed.append((prev, seg_start_frame, num_frames))

    words = []
    cur_chars = []
    cur_start = None
    cur_end = None
    for tid, sf, ef in collapsed:
        if tid == pad_id:
            continue
        t_start = sf * frame_dur + offset
        t_end = ef * frame_dur + offset
        if tid == delim_id:
            if cur_chars:
                words.append({"word": "".join(cur_chars), "start": cur_start, "end": cur_end})
                cur_chars = []
                cur_start = None
            continue
        ch = processor.tokenizer.convert_ids_to_tokens(tid)
        if cur_start is None:
            cur_start = t_start
        cur_chars.append(ch)
        cur_end = t_end
    if cur_chars:
        words.append({"word": "".join(cur_chars), "start": cur_start, "end": cur_end})

    return words
