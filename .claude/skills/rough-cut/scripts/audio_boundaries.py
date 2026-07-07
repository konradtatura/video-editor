"""
Measures real voice-activity boundaries directly from the source audio's
energy envelope, to correct WhisperX word timestamps that are known to
over-extend the *end* of a word when it's followed by a pause (a structural
artifact of CTC/wav2vec2 forced alignment: unclaimed silent frames get
assigned to the preceding word instead of being left as silence).

This never re-guesses "what" to cut -- that stays an editorial decision made
by reading the transcript. It only corrects "where exactly" a kept clip's
audio should start/end, snapping to the nearest measured voice boundary
instead of trusting the alignment model's word span.
"""

import hashlib
import os
import subprocess
import wave

import numpy as np

from ffmpeg_util import find_ffmpeg

HOP_MS = 10
WIN_MS = 30
FLOOR_PERCENTILE = 10
VOICE_MARGIN_DB = 8


def _cache_path(source_path, cache_dir):
    stat = os.stat(source_path)
    key = f"{source_path}|{stat.st_size}|{stat.st_mtime}|hop{HOP_MS}|win{WIN_MS}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"envelope_{h}.npz")


def _extract_pcm(source_path, wav_path, sr=16000):
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", source_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg PCM extraction failed:\n{result.stderr[-2000:]}")


def _load_wav_mono(wav_path):
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr


def _compute_envelope(data, sr, hop_ms=HOP_MS, win_ms=WIN_MS):
    hop = max(1, int(sr * hop_ms / 1000))
    win = max(1, int(sr * win_ms / 1000))
    n_frames = max(0, (len(data) - win) // hop + 1)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = data[i * hop: i * hop + win]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    times = (np.arange(n_frames) * hop + win / 2) / sr
    db = 20 * np.log10(rms + 1e-9)
    return times.astype(np.float32), db.astype(np.float32)


MIN_SUSTAINED_SILENCE_S = 0.15


class VoiceEnvelope:
    def __init__(self, times, db):
        self.times = times
        self.db = db
        floor = np.percentile(db, FLOOR_PERCENTILE)
        self.threshold = floor + VOICE_MARGIN_DB
        self.voiced = db > self.threshold

        # A raw voiced->unvoiced transition can be a brief stop-consonant or
        # breath mid-phrase, not real end-of-speech (this produced a real bug:
        # a 0.1s dip before the word "osiem" got accepted as the clip's end,
        # truncating the actual punchline). Only accept a falling/rising edge
        # as a true boundary if the silence/voice on the far side sustains for
        # at least MIN_SUSTAINED_SILENCE_S.
        hop_s = float(times[1] - times[0]) if len(times) > 1 else 0.01
        min_frames = max(1, int(round(MIN_SUSTAINED_SILENCE_S / hop_s)))
        n = len(self.voiced)

        sustained_fall = np.zeros(n, dtype=bool)
        sustained_rise = np.zeros(n, dtype=bool)
        for i in range(n - min_frames):
            if self.voiced[i] and not self.voiced[i + 1: i + 1 + min_frames].any():
                sustained_fall[i] = True
            if not self.voiced[i] and self.voiced[i + 1: i + 1 + min_frames].all():
                sustained_rise[i] = True
        self.sustained_fall = sustained_fall
        self.sustained_rise = sustained_rise

    def snap_end(self, claimed_end, lower_bound=None, lookback=2.0, forward_tolerance=0.3, pad=0.06):
        """Find the true last-voiced sample at/before claimed_end and return
        that boundary + pad. Falls back to claimed_end if no clean voice->
        silence transition is found nearby (e.g. genuinely mid-speech cut).
        lower_bound (e.g. this clip's own snapped start) prevents the lookback
        window from reaching into a neighboring clip's speech, which happens
        for very short/quiet clips."""
        lo = claimed_end - lookback
        if lower_bound is not None:
            lo = max(lo, lower_bound)
        hi = claimed_end + forward_tolerance
        i_lo = np.searchsorted(self.times, lo)
        i_hi = np.searchsorted(self.times, hi)
        window = self.sustained_fall[i_lo:i_hi]
        if window.size == 0:
            return claimed_end
        edges = np.where(window)[0]
        if edges.size == 0:
            return claimed_end
        last_edge = i_lo + edges[-1]
        true_end = float(self.times[last_edge])
        return min(true_end + pad, claimed_end + forward_tolerance)

    def snap_start(self, claimed_start, upper_bound=None, lookahead=1.0, backward_tolerance=0.3, pad=0.03):
        """Find the true first-voiced sample at/after claimed_start (minus a
        small backward tolerance) and return that boundary - pad.
        upper_bound (e.g. this clip's own claimed end) prevents the lookahead
        window from reaching into a neighboring clip's speech, which happens
        for very short/quiet clips."""
        lo = claimed_start - backward_tolerance
        hi = claimed_start + lookahead
        if upper_bound is not None:
            hi = min(hi, upper_bound)
        i_lo = np.searchsorted(self.times, lo)
        i_hi = np.searchsorted(self.times, hi)
        window = self.sustained_rise[i_lo:i_hi]
        if window.size == 0:
            return claimed_start
        edges = np.where(window)[0]
        if edges.size == 0:
            return claimed_start
        first_edge = i_lo + edges[0]
        true_start = float(self.times[first_edge])
        return max(true_start - pad, claimed_start - backward_tolerance)


def load_envelope(source_path, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = _cache_path(source_path, cache_dir)
    if os.path.exists(cache_file):
        npz = np.load(cache_file)
        return VoiceEnvelope(npz["times"], npz["db"])

    wav_path = os.path.join(cache_dir, "_boundaries_tmp.wav")
    _extract_pcm(source_path, wav_path)
    data, sr = _load_wav_mono(wav_path)
    times, db = _compute_envelope(data, sr)
    os.remove(wav_path)

    np.savez(cache_file, times=times, db=db)
    return VoiceEnvelope(times, db)
