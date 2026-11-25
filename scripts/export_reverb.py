"""Write reverberation-augmented audio to disk: reverberant input, early-reflection target, dry speech.

Uses exactly the same augmentation as training (srse/data/dataset.py), so the files are what the model sees.
Each item is written as <id>_reverb.wav, <id>_target.wav and (with --dry) <id>_dry.wav, plus
manifest.jsonl / manifest.csv with speaker, room class, RIR, T60, level and, for full utterances,
the LibriSpeech transcript (ready for ASR training / evaluation).

Splits:
  train  training speakers, training rooms            val   held-out utterances, validation rooms
  test   test-clean speakers, unseen simulated rooms  real  test-clean speakers, measured RIRs

Examples:
  # 40 examples to listen to (10 per room class)
  python scripts/export_reverb.py --split train --n 40 --out_dir /content/reverb_examples
  # exactly the items the test script scores
  python scripts/export_reverb.py --split test --match_eval --out_dir /content/reverb_test
  # every training utterance once, full length, with transcripts (for ASR; ~11 GB per stream)
  python scripts/export_reverb.py --split train --all --full --out_dir /content/reverb_asr_train
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from srse import ROOM_CLASSES  # noqa: E402
from srse.config import load_config  # noqa: E402
from srse.data.dataset import (ReverbDataset, load_manifests, make_real_items,  # noqa: E402
                               make_sim_items, rir_pool, split_speakers)


def load_transcripts(root, utts):
    """LibriSpeech transcripts: <spk>/<chapter>/<spk>-<chapter>.trans.txt, lines 'UTT_ID TEXT'."""
    text = {}
    for d in sorted({(root / u["path"]).parent for u in utts}):
        for f in d.glob("*.trans.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if " " in line:
                    k, v = line.split(" ", 1)
                    text[k] = v
    return text


def random_items(utts, pool, n, seed, crop_len, level_range, room=None, all_utts=False, max_len=None):
    """Items with a uniformly random room class (or a fixed one), like training."""
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(utts)) if all_utts else rng.permutation(len(utts))[:n]
    if not all_utts and n > len(utts):  # more items than utterances: sample with replacement
        idxs = rng.integers(0, len(utts), size=n)
    items = []
    for k, i in enumerate(idxs):
        c = ROOM_CLASSES.index(room) if room else (k % len(ROOM_CLASSES) if not all_utts
                                                   else int(rng.integers(0, len(ROOM_CLASSES))))
        nfr = utts[i]["n_frames"]
        if crop_len:
            start, length = int(rng.integers(0, max(1, nfr - crop_len + 1))), crop_len
        else:
            start, length = 0, (min(nfr, max_len) if max_len else None)
        rir = None if c == 0 else str(rng.choice(pool[ROOM_CLASSES[c]]))
        items.append(dict(utt=int(i), room=c, rir=rir, start=start, length=length,
                          level=float(rng.uniform(*level_range)), kind="sim", t60=float("nan")))
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. paths.data_root=/content/data")
    ap.add_argument("--split", choices=["train", "val", "test", "real"], default="train")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=40, help="number of items (ignored with --all / --match_eval)")
    ap.add_argument("--all", action="store_true", help="every utterance of the split once, random room each")
    ap.add_argument("--match_eval", action="store_true",
                    help="export exactly the fixed validation / test items used for scoring")
    ap.add_argument("--full", action="store_true", help="full utterances (default: training-length crops)")
    ap.add_argument("--room", choices=ROOM_CLASSES, default=None, help="force one room class")
    ap.add_argument("--dry", action="store_true", help="also write the dry (anechoic) speech")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    sr = cfg.data.sample_rate
    seg = None if a.full else int(cfg.data.segment_seconds * sr)
    max_len = int(cfg.data.max_test_seconds * sr) if a.split in ("test", "real") else None
    m = load_manifests(cfg)
    min_len = cfg.data.min_utt_seconds * sr

    if a.split in ("train", "val"):
        train_utts, val_utts, spk2idx = split_speakers(m["train"], cfg)
        utts = train_utts if a.split == "train" else val_utts
        pool = rir_pool(m["rirs_sim"], a.split)
    else:
        utts = [u for u in m["test"] if u["n_frames"] >= min_len]
        spk2idx = {}
        pool = rir_pool(m["rirs_sim"], "test") if a.split == "test" else None

    if a.match_eval:
        if a.split == "val":
            items = make_sim_items(utts, pool, cfg.data.val_items_per_class, seed=cfg.data.split_seed + 1,
                                   crop_len=int(cfg.data.segment_seconds * sr), level_db=cfg.data.eval_level_db)
        elif a.split == "test":
            items = make_sim_items(utts, pool, cfg.data.test_utts, seed=cfg.data.test_seed, crop_len=None,
                                   level_db=cfg.data.eval_level_db, max_len=max_len)
        elif a.split == "real":
            items = make_real_items(utts, m["rirs_real"], cfg.data.real_rir_test_utts, seed=cfg.data.test_seed + 1,
                                    level_db=cfg.data.eval_level_db, max_len=max_len)
        else:
            sys.exit("--match_eval applies to val, test and real (training items are random every epoch)")
    elif a.split == "real":
        rng = np.random.default_rng(a.seed)
        items = make_real_items(utts, m["rirs_real"], a.n, seed=a.seed, level_db=cfg.data.eval_level_db,
                                max_len=max_len)
        for it in items:
            if seg:
                n = utts[it["utt"]]["n_frames"]
                it["start"], it["length"] = int(rng.integers(0, max(1, n - seg + 1))), seg
    else:
        items = random_items(utts, pool, a.n, a.seed, seg, cfg.data.train_level_db, a.room, a.all, max_len)

    ds = ReverbDataset(cfg, utts, pool, spk2idx, items=items)
    root = Path(cfg.paths.data_root)
    transcripts = load_transcripts(root, [utts[it["utt"]] for it in items]) if a.full else {}
    rir_t60 = {r["path"]: r.get("t60") for r in m["rirs_real"]}

    out = Path(a.out_dir) / a.split
    wav_dir = out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[export] {len(items)} items -> {out}")
    for k, it in enumerate(items):
        u = utts[it["utt"]]
        rev, tgt, dry = ds.render(it)
        utt_id = Path(u["path"]).stem
        room = ROOM_CLASSES[it["room"]] if it["room"] >= 0 else "real"
        item_id = f"{k:06d}_{utt_id}_{room}"
        sf.write(wav_dir / f"{item_id}_reverb.wav", rev, sr, subtype="PCM_16")
        sf.write(wav_dir / f"{item_id}_target.wav", tgt, sr, subtype="PCM_16")
        if a.dry:
            sf.write(wav_dir / f"{item_id}_dry.wav", dry, sr, subtype="PCM_16")
        full = it["length"] is None or (it["start"] == 0 and it["length"] >= u["n_frames"])
        t60 = rir_t60.get(it["rir"], it.get("t60"))
        t60 = None if t60 is None or t60 != t60 else round(float(t60), 3)
        rows.append(dict(
            id=item_id, split=a.split, speaker=u["speaker"], utt_id=utt_id, source=u["path"],
            room_class=room, rir=it["rir"] or "", t60=t60,
            start_sec=round(it["start"] / sr, 3), seconds=round(len(rev) / sr, 3), level_db=round(it["level"], 2),
            reverb=f"wav/{item_id}_reverb.wav", target=f"wav/{item_id}_target.wav",
            dry=f"wav/{item_id}_dry.wav" if a.dry else "",
            text=transcripts.get(utt_id, "") if full else ""))
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {c: sum(r["room_class"] == c for r in rows) for c in ROOM_CLASSES + ["real"]}
    print(f"[done] {len(rows)} items | " + ", ".join(f"{c}: {n}" for c, n in counts.items() if n)
          + f" | manifest: {out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
