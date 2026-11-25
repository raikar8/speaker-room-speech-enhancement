"""Splits and on-the-fly reverberant datasets.

Every training example is a random crop of one LibriSpeech utterance placed in a room drawn with equal
probability from {anechoic, small, medium, large} (EUSIPCO 2022 recipe). Validation and test use fixed,
seeded item lists so every experiment sees exactly the same examples (needed for paired tests).
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from srse import ROOM_CLASSES
from srse.data.rir import load_rir, reverberate


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_manifests(cfg):
    mdir = Path(cfg.paths.manifest_dir)
    return {
        "train": read_jsonl(mdir / "libri_train-clean-100.jsonl"),
        "test": read_jsonl(mdir / "libri_test-clean.jsonl"),
        "rirs_sim": read_jsonl(mdir / "rirs_sim.jsonl"),
        "rirs_real": read_jsonl(mdir / "rirs_real.jsonl"),
    }


def split_speakers(utts, cfg):
    """Pick speakers and split each speaker's utterances into val (first k) and train (next n).

    Uses data.split_seed only, so the split is identical across experiments and training seeds.
    """
    sr = cfg.data.sample_rate
    min_len = cfg.data.min_utt_seconds * sr
    by_spk = {}
    for u in utts:
        if u["n_frames"] >= min_len:
            by_spk.setdefault(u["speaker"], []).append(u)
    rng = random.Random(cfg.data.split_seed)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)
    if cfg.data.max_speakers:
        speakers = speakers[: cfg.data.max_speakers]
    speakers = sorted(speakers)
    spk2idx = {s: i for i, s in enumerate(speakers)}
    train, val = [], []
    nv = cfg.data.val_utts_per_speaker
    for s in speakers:
        us = sorted(by_spk[s], key=lambda u: u["path"])
        rng.shuffle(us)
        val += us[:nv]
        rest = us[nv:]
        if cfg.data.utts_per_speaker:
            rest = rest[: cfg.data.utts_per_speaker]
        train += rest
    return train, val, spk2idx


def rir_pool(rirs, split):
    pool = {c: [] for c in ROOM_CLASSES[1:]}
    for r in rirs:
        if r["split"] == split:
            pool[r["room_class"]].append(r["path"])
    for c, paths in pool.items():
        if not paths:
            raise RuntimeError(f"No '{split}' RIRs for room class '{c}'. Re-run prepare_data.py.")
    return pool


def make_sim_items(utts, pool, n_per_class, seed, crop_len, level_db, max_len=None):
    """Fixed items: n_per_class utterances for each room class (anechoic gets no RIR)."""
    rng = np.random.default_rng(seed)
    items = []
    for c, name in enumerate(ROOM_CLASSES):
        idxs = rng.permutation(len(utts))
        if n_per_class:
            idxs = idxs[:n_per_class]
        for i in idxs:
            n = utts[i]["n_frames"]
            if crop_len:
                start, length = int(rng.integers(0, max(1, n - crop_len + 1))), crop_len
            else:
                start, length = 0, (min(n, max_len) if max_len else None)
            rir = None if c == 0 else str(rng.choice(pool[name]))
            items.append(dict(utt=int(i), room=c, rir=rir, start=start, length=length,
                              level=level_db, kind="sim", t60=float("nan")))
    return items


def make_real_items(utts, real_rirs, n_items, seed, level_db, max_len=None):
    """Fixed items using measured (real) RIRs. Room class is unknown (-1); T60 is stored for binning."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n_items):
        i = int(rng.integers(0, len(utts)))
        r = real_rirs[int(rng.integers(0, len(real_rirs)))]
        n = utts[i]["n_frames"]
        items.append(dict(utt=i, room=-1, rir=r["path"], start=0, length=(min(n, max_len) if max_len else None),
                          level=level_db, kind="real", t60=float(r.get("t60", float("nan")))))
    return items


class ReverbDataset(Dataset):
    """Random items when `items is None` (training), otherwise the given fixed items."""

    def __init__(self, cfg, utts, pool=None, spk2idx=None, items=None, shuffle_labels=False, seed=0):
        self.cfg = cfg
        self.root = Path(cfg.paths.data_root)
        self.utts = utts
        self.pool = pool
        self.spk2idx = spk2idx or {}
        self.items = items
        self.sr = cfg.data.sample_rate
        self.seg = int(cfg.data.segment_seconds * self.sr)
        self.rng = np.random.default_rng(seed)
        self.shuffle_labels = shuffle_labels
        if shuffle_labels:
            n_spk = max(len(self.spk2idx), 1)
            # fixed random speaker label per utterance; room label redrawn every time
            self.fake_spk = np.random.default_rng(cfg.data.split_seed + 99).integers(0, n_spk, size=len(utts))

    def __len__(self):
        return len(self.items) if self.items is not None else len(self.utts)

    def _random_item(self, idx):
        u = self.utts[idx]
        room = int(self.rng.integers(0, len(ROOM_CLASSES)))
        rir = None if room == 0 else str(self.rng.choice(self.pool[ROOM_CLASSES[room]]))
        start = int(self.rng.integers(0, max(1, u["n_frames"] - self.seg + 1)))
        lo, hi = self.cfg.data.train_level_db
        return dict(utt=idx, room=room, rir=rir, start=start, length=self.seg,
                    level=float(self.rng.uniform(lo, hi)), kind="sim", t60=float("nan"))

    def _read(self, u, start, length):
        n = u["n_frames"]
        pre = min(start, int(self.cfg.data.preroll_seconds * self.sr))
        stop = n if length is None else min(n, start + length)
        x, _ = sf.read(str(self.root / u["path"]), start=start - pre, stop=stop, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        return x, pre

    def __getitem__(self, idx):
        it = self._random_item(idx) if self.items is None else self.items[idx]
        u = self.utts[it["utt"]]
        x, pre = self._read(u, it["start"], it["length"])
        rir = None
        if it["rir"] is not None:
            rir = load_rir(self.root / it["rir"], self.sr, self.cfg.data.max_rir_seconds)
        rev, tgt = reverberate(x, rir, self.sr, self.cfg.data.early_ms)
        rev, tgt = rev[pre:], tgt[pre:]
        if it["length"] is not None and len(rev) < it["length"]:
            pad = it["length"] - len(rev)
            rev, tgt = np.pad(rev, (0, pad)), np.pad(tgt, (0, pad))
        gain = 10 ** (it["level"] / 20) / (np.sqrt(np.mean(rev ** 2)) + 1e-8)
        rev, tgt = rev * gain, tgt * gain
        peak = max(np.abs(rev).max(), np.abs(tgt).max(), 1e-8)
        if peak > 0.99:
            rev, tgt = rev * 0.99 / peak, tgt * 0.99 / peak

        spk = self.spk2idx.get(u["speaker"], -1)
        spk_target, room_target = spk, it["room"]
        if self.shuffle_labels:
            spk_target = int(self.fake_spk[it["utt"]])
            room_target = int(self.rng.integers(0, len(ROOM_CLASSES)))
        return {
            "input": torch.from_numpy(rev.astype(np.float32)),
            "target": torch.from_numpy(tgt.astype(np.float32)),
            "spk": spk,
            "room": it["room"],
            "spk_target": spk_target,
            "room_target": room_target,
            "index": idx,
        }


def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


def build_train_val(cfg, manifests=None):
    m = manifests or load_manifests(cfg)
    train_utts, val_utts, spk2idx = split_speakers(m["train"], cfg)
    train_ds = ReverbDataset(cfg, train_utts, rir_pool(m["rirs_sim"], "train"), spk2idx,
                             shuffle_labels=cfg.mtl.shuffle_labels, seed=cfg.seed)
    seg = int(cfg.data.segment_seconds * cfg.data.sample_rate)
    val_items = make_sim_items(val_utts, rir_pool(m["rirs_sim"], "val"), cfg.data.val_items_per_class,
                               seed=cfg.data.split_seed + 1, crop_len=seg, level_db=cfg.data.eval_level_db)
    val_ds = ReverbDataset(cfg, val_utts, spk2idx=spk2idx, items=val_items)
    return train_ds, val_ds, spk2idx
