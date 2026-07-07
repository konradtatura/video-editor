"""
ASR-independent repeat detector: finds stretches of audio that closely
resemble an earlier stretch, using spectral self-similarity -- not Whisper
text. This is the safety net for the fact that no amount of decode-parameter
tuning or chunking fully stops Whisper from occasionally deleting or
mis-rendering genuinely repeated speech (see project notes): repeats found
here should be cross-checked against the transcript, and anything the
transcript missed is a direct signal of a transcription gap, not an
editorial one.

Method: log-mel-ish spectral features (via a hand-rolled triangular filter-
bank -- no librosa dependency) averaged into ~150ms frames, cosine-similarity
self-similarity matrix, then scan each usable time-lag for sustained runs of
high similarity along the diagonal (the classic "repeated chorus" detection
technique from music information retrieval, applied to speech).

Usage:
    python find_repeats.py <source_media> <output_json>
"""

import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import stft

from ffmpeg_util import find_ffmpeg

SR = 16000
STFT_WIN = 400        # 25ms @ 16kHz
STFT_HOP = 160        # 10ms @ 16kHz
N_MELS = 26
COARSE_HOP_S = 0.15   # aggregate frames to ~150ms resolution
MIN_LAG_S = 1.0        # ignore trivial near-adjacent "self" matches
MIN_RUN_S = 0.8        # minimum duration to count as a real repeat
SIMILARITY_THRESHOLD = 0.90


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


def extract_features(source_path):
    ffmpeg = find_ffmpeg()
    wav_path = source_path + "._repeats_tmp.wav"
    cmd = [ffmpeg, "-y", "-i", source_path, "-ac", "1", "-ar", str(SR), "-vn", wav_path]
    subprocess.run(cmd, capture_output=True, check=True)
    try:
        with wave.open(wav_path, "rb") as w:
            n = w.getnframes()
            raw = w.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    freqs, times, Zxx = stft(data, fs=SR, nperseg=STFT_WIN, noverlap=STFT_WIN - STFT_HOP)
    mag = np.abs(Zxx)
    fb = _mel_filterbank(SR, STFT_WIN, N_MELS)
    mel = fb @ mag
    log_mel = np.log(mel + 1e-6)

    # aggregate fine STFT frames into coarse ~150ms frames
    fine_hop_s = STFT_HOP / SR
    coarse_group = max(1, int(round(COARSE_HOP_S / fine_hop_s)))
    n_coarse = log_mel.shape[1] // coarse_group
    coarse = log_mel[:, :n_coarse * coarse_group].reshape(N_MELS, n_coarse, coarse_group).mean(axis=2)
    coarse_times = times[:n_coarse * coarse_group].reshape(n_coarse, coarse_group).mean(axis=1)

    feats = coarse.T  # (n_coarse, N_MELS)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats_normed = feats / (norms + 1e-8)
    return feats_normed, coarse_times


def find_repeats(feats, times):
    n = feats.shape[0]
    hop_s = times[1] - times[0] if n > 1 else COARSE_HOP_S
    min_lag = max(1, int(round(MIN_LAG_S / hop_s)))
    min_run = max(1, int(round(MIN_RUN_S / hop_s)))

    candidates = []
    for lag in range(min_lag, n):
        # similarity along this diagonal: feats[i] vs feats[i+lag]
        a = feats[: n - lag]
        b = feats[lag:]
        sims = np.sum(a * b, axis=1)  # cosine similarity (already normalized)
        above = sims >= SIMILARITY_THRESHOLD

        i = 0
        while i < len(above):
            if above[i]:
                j = i
                while j < len(above) and above[j]:
                    j += 1
                run_len = j - i
                if run_len >= min_run:
                    t1_start, t1_end = float(times[i]), float(times[j - 1])
                    t2_start, t2_end = float(times[i + lag]), float(times[j - 1 + lag])
                    avg_sim = float(sims[i:j].mean())
                    candidates.append({
                        "a_start": t1_start, "a_end": t1_end,
                        "b_start": t2_start, "b_end": t2_end,
                        "duration": t1_end - t1_start,
                        "similarity": avg_sim,
                    })
                i = j
            else:
                i += 1

    candidates.sort(key=lambda c: -c["duration"])
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_media")
    parser.add_argument("output_json")
    args = parser.parse_args()

    print("[find_repeats] extracting spectral features...", file=sys.stderr)
    feats, times = extract_features(args.source_media)
    print(f"[find_repeats] {feats.shape[0]} frames, scanning for repeats...", file=sys.stderr)
    candidates = find_repeats(feats, times)

    # de-duplicate heavily overlapping candidates (keep the longest per rough region)
    deduped = []
    for c in candidates:
        overlap = False
        for d in deduped:
            if abs(c["a_start"] - d["a_start"]) < 1.0 and abs(c["b_start"] - d["b_start"]) < 1.0:
                overlap = True
                break
        if not overlap:
            deduped.append(c)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False, indent=2)

    print(f"[find_repeats] found {len(deduped)} repeat candidates, wrote {args.output_json}", file=sys.stderr)
    for c in deduped[:30]:
        print(f"  {c['a_start']:7.2f}-{c['a_end']:7.2f}  <->  {c['b_start']:7.2f}-{c['b_end']:7.2f}  "
              f"(dur {c['duration']:.2f}s, sim {c['similarity']:.3f})", file=sys.stderr)


if __name__ == "__main__":
    main()
