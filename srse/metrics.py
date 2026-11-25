"""Evaluation metrics: SI-SDR, wide-band PESQ, ESTOI, optional SRMR, speaker EER."""

import math

import numpy as np


def si_sdr(est, ref, eps=1e-8):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s = alpha * ref
    return float(10 * np.log10(np.dot(s, s) / (np.dot(s - est, s - est) + eps) + eps))


def pesq_wb(ref, est, sr=16000):
    try:
        from pesq import pesq
        return float(pesq(sr, ref, est, "wb"))
    except Exception:
        return math.nan


def estoi(ref, est, sr=16000):
    try:
        from pystoi import stoi
        return float(stoi(ref, est, sr, extended=True))
    except Exception:
        return math.nan


def srmr(x, sr=16000):
    """Non-intrusive reverberation metric; needs the optional `srmrpy` package."""
    try:
        from srmrpy import srmr as _srmr
        return float(_srmr(x, sr)[0])
    except Exception:
        return math.nan


def eer(embeddings, labels):
    """Equal error rate of cosine scoring over all pairs."""
    e = np.asarray(embeddings, dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    labels = np.asarray(labels)
    iu = np.triu_indices(len(e), 1)
    scores = (e @ e.T)[iu]
    same = labels[iu[0]] == labels[iu[1]]
    if same.all() or (~same).all():
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    s, same = scores[order], same[order]
    tp, fp = np.cumsum(same), np.cumsum(~same)
    last = np.r_[s[1:] != s[:-1], True]  # one operating point per distinct threshold (handles ties)
    tp, fp = np.r_[0, tp[last]], np.r_[0, fp[last]]
    fnr = 1 - tp / same.sum()
    fpr = fp / (~same).sum()
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)
