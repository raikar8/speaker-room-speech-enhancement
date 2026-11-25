"""Room impulse response loading, reverberation with an early-reflection target, and T60 estimation."""

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def load_rir(path, sr, max_seconds=2.0):
    """Load channel 0 of an RIR, resample to `sr`, start it just before the direct path."""
    x, fs = sf.read(str(path), dtype="float32", always_2d=True)
    x = x[:, 0]
    if fs != sr:
        g = gcd(int(sr), int(fs))
        x = resample_poly(x, sr // g, fs // g).astype(np.float32)
    d = int(np.argmax(np.abs(x)))
    x = x[max(0, d - int(0.0025 * sr)):]  # keep 2.5 ms before the direct path
    return x[: int(max_seconds * sr)]


def reverberate(x, rir, sr, early_ms=50.0):
    """Return (reverberant, target). Target = direct path + early reflections (first `early_ms`).

    For the anechoic class (`rir is None`) both are the dry signal.
    """
    if rir is None:
        return x.astype(np.float32).copy(), x.astype(np.float32).copy()
    d = int(np.argmax(np.abs(rir)))
    rir = rir / (np.abs(rir[d]) + 1e-8)  # direct path peak = 1
    early = rir[: d + int(early_ms * sr / 1000) + 1]
    rev = fftconvolve(x, rir)[: len(x)].astype(np.float32)
    tgt = fftconvolve(x, early)[: len(x)].astype(np.float32)
    return rev, tgt


def estimate_t60(rir, sr):
    """T60 from Schroeder backward integration (T20 fit between -5 and -25 dB). NaN if unreliable."""
    d = int(np.argmax(np.abs(rir)))
    e = rir[d:].astype(np.float64) ** 2
    if e.sum() <= 0:
        return float("nan")
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
    below5 = np.nonzero(edc_db <= -5)[0]
    below25 = np.nonzero(edc_db <= -25)[0]
    if len(below5) == 0 or len(below25) == 0 or below25[0] - below5[0] < 10:
        return float("nan")
    i5, i25 = below5[0], below25[0]
    t = np.arange(i5, i25) / sr
    slope = np.polyfit(t, edc_db[i5:i25], 1)[0]
    return float(-60.0 / slope) if slope < 0 else float("nan")
