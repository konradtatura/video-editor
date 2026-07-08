"""
Shared spectral-feature + DTW distance code for acoustic (ASR-independent)
repeat detection. Pulled out of the original find_repeats.py so both the
burst-pair comparator and any future caller can reuse the same features.

Why DTW and not fixed-time-lag matching (the original find_repeats.py
approach): a retake is almost never spoken at exactly the same pace as the
original -- the whole point of a retake is the speaker adjusting something.
Fixed-lag cosine similarity requires the two repeats to line up frame-for-
frame, so it silently misses the normal case (verified: it missed the single
most obvious repeat in this pipeline's test corpus, a stutter spoken faster
the second time). Dynamic time warping finds the cheapest alignment between
two sequences of different length/pace, so "same phrase, said 15% faster" is
still recognized as a close match.

This does not identify "what" a repeat is (that's still a transcript/editorial
question) -- it only measures "how acoustically similar are these two spans",
independent of anything Whisper decoded from them, which matters because
Whisper's repeat-suppression can make two genuinely-repeated spans read as
one clean sentence in the transcript while the audio still obviously repeats.
"""

import os
import subprocess
import wave

import numpy as np
from scipy.signal import stft

from ffmpeg_util import find_ffmpeg

SR = 16000
STFT_WIN = 400        # 25ms @ 16kHz
STFT_HOP = 160        # 10ms @ 16kHz
N_MELS = 26
FEATURE_HOP_S = 0.03  # aggregate fine STFT frames to ~30ms resolution for DTW


def _mel_filterbank(sr, n_fft, n_mels):
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    low_mel, high_mel = 0, hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb


def _extract_pcm(source_path, wav_path, sr=SR):
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", source_path, "-ac", "1", "-ar", str(sr), "-vn", wav_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg PCM extraction failed:\n{result.stderr[-2000:]}")


def _load_wav_mono(wav_path):
    with wave.open(wav_path, "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def compute_features(source_path):
    """Whole-file normalized log-mel features + their frame center times.
    Frame hop is ~FEATURE_HOP_S (30ms) -- fine enough to resolve a single
    fast syllable, coarse enough that a several-second burst is a manageable
    number of frames for O(n*m) DTW."""
    wav_path = source_path + "._dtw_tmp.wav"
    _extract_pcm(source_path, wav_path)
    try:
        data = _load_wav_mono(wav_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    freqs, times, Zxx = stft(data, fs=SR, nperseg=STFT_WIN, noverlap=STFT_WIN - STFT_HOP)
    mag = np.abs(Zxx)
    fb = _mel_filterbank(SR, STFT_WIN, N_MELS)
    mel = fb @ mag
    log_mel = np.log(mel + 1e-6)

    fine_hop_s = STFT_HOP / SR
    coarse_group = max(1, int(round(FEATURE_HOP_S / fine_hop_s)))
    n_coarse = log_mel.shape[1] // coarse_group
    if n_coarse == 0:
        return np.zeros((0, N_MELS), dtype=np.float32), np.zeros(0, dtype=np.float32)
    coarse = log_mel[:, :n_coarse * coarse_group].reshape(N_MELS, n_coarse, coarse_group).mean(axis=2)
    coarse_times = times[:n_coarse * coarse_group].reshape(n_coarse, coarse_group).mean(axis=1)

    feats = coarse.T  # (n_coarse, N_MELS)
    # per-video normalization (not per-burst) so relative loudness/timbre
    # differences between two bursts remain part of the DTW distance instead
    # of being normalized away
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats_normed = feats / (norms + 1e-8)
    return feats_normed.astype(np.float32), coarse_times.astype(np.float32)


def slice_features(feats, times, start, end):
    i_lo = np.searchsorted(times, start)
    i_hi = np.searchsorted(times, end)
    return feats[i_lo:i_hi]


def dtw_distance(a, b, band=None):
    """Normalized DTW distance between two (n, d) / (m, d) feature sequences
    of unit-norm rows, using cosine distance as the per-frame cost. Returns
    a value roughly in [0, 2]; lower = more acoustically similar regardless
    of the two sequences' relative speed. `band` (frames) restricts the warp
    path to a Sakoe-Chiba band around the diagonal, purely for speed -- not
    needed at burst-length sequences (a few dozen frames) but harmless."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 1.0
    cost = 1.0 - a @ b.T  # cosine distance matrix, (n, m)

    INF = np.inf
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        j_lo, j_hi = 1, m
        if band is not None:
            j_lo = max(1, i - band)
            j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            c = cost[i - 1, j - 1]
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    # normalize by path length (approximated as n+m, the standard DTW
    # normalization) so a long pair of bursts isn't penalized just for
    # having more frames to accumulate cost over
    return float(D[n, m] / (n + m))
