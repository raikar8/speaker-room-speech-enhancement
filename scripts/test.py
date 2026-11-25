"""Test a trained run on unseen speakers (LibriSpeech test-clean) and unseen rooms.

Two test sets, both fixed by data.test_seed (identical across experiments -> paired comparisons):
  * sim  : test-split simulated RIRs, test_utts utterances per room class (anechoic/small/medium/large)
  * real : measured RIRs from RIRS_NOISES, binned by estimated T60

Writes <run_dir>/<tag>/per_utt.csv and summary.json.

  python scripts/test.py --run_dir /content/drive/MyDrive/srse_runs/B0/seed0
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from srse import ROOM_CLASSES  # noqa: E402
from srse.data.dataset import (ReverbDataset, load_manifests, make_real_items,  # noqa: E402
                               make_sim_items, rir_pool)
from srse.metrics import eer, estoi, pesq_wb, si_sdr, srmr  # noqa: E402
from srse.utils import get_device, load_model_for_eval  # noqa: E402

T60_BINS = [(0.0, 0.3, "t60<0.3"), (0.3, 0.6, "t60_0.3-0.6"), (0.6, 99.0, "t60>0.6")]
TIME_POINTS = [0.25, 0.5, 1.0, 2.0, 3.5, 5.0]


def t60_bin(t60):
    if t60 is None or math.isnan(t60):
        return "t60_unknown"
    for lo, hi, name in T60_BINS:
        if lo <= t60 < hi:
            return name
    return "t60_unknown"


def mean(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--sim_per_class", type=int, default=None, help="default: data.test_utts")
    ap.add_argument("--real_items", type=int, default=None, help="default: data.real_rir_test_utts")
    ap.add_argument("--window", type=int, default=None, help="pooling window in frames (0 = from start)")
    ap.add_argument("--no_pesq", action="store_true")
    ap.add_argument("--srmr", action="store_true", help="also compute SRMR (needs srmrpy)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    model, cfg, state, device = load_model_for_eval(run_dir, a.ckpt, get_device(a.device))
    sr, hop = cfg.data.sample_rate, cfg.stft.hop
    window = cfg.mtl.eval_pool_frames if a.window is None else a.window
    m = load_manifests(cfg)
    min_len = cfg.data.min_utt_seconds * sr
    test_utts = [u for u in m["test"] if u["n_frames"] >= min_len]
    max_len = int(cfg.data.max_test_seconds * sr)
    n_sim = cfg.data.test_utts if a.sim_per_class is None else a.sim_per_class
    n_real = cfg.data.real_rir_test_utts if a.real_items is None else a.real_items
    items = make_sim_items(test_utts, rir_pool(m["rirs_sim"], "test"), n_sim, seed=cfg.data.test_seed,
                           crop_len=None, level_db=cfg.data.eval_level_db, max_len=max_len)
    if cfg.oracle.room:
        print("[note] oracle-room model: real-RIR set skipped (room class unknown)")
    elif n_real and m["rirs_real"]:
        items += make_real_items(test_utts, m["rirs_real"], n_real, seed=cfg.data.test_seed + 1,
                                 level_db=cfg.data.eval_level_db, max_len=max_len)
    ds = ReverbDataset(cfg, test_utts, items=items)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)
    print(f"[test] {run_dir} | {len(items)} items | window={window}")

    rows, embs, emb_spk, emb_room = [], [], [], []
    proc_time, audio_time = 0.0, 0.0
    for k, b in enumerate(dl):
        it = items[k]
        x, y = b["input"].to(device), b["target"]
        with torch.no_grad():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x, room=b["room"].clamp(min=0).to(device), window=window)
            if device.type == "cuda":
                torch.cuda.synchronize()
            proc_time += time.perf_counter() - t0
        audio_time += x.shape[-1] / sr
        xin, ref, est = x[0].cpu().numpy(), y[0].numpy(), out["est_wav"][0].float().cpu().numpy()
        room_logits = out["room_logits"][0].float().cpu()
        T = room_logits.shape[0]
        row = dict(item=k, kind=it["kind"], room=ROOM_CLASSES[it["room"]] if it["room"] >= 0 else "real",
                   t60_bin=t60_bin(it["t60"]) if it["kind"] == "real" else "",
                   speaker=test_utts[it["utt"]]["speaker"], seconds=round(x.shape[-1] / sr, 2),
                   sisdr_in=si_sdr(xin, ref), sisdr_out=si_sdr(est, ref))
        if not a.no_pesq:
            row.update(pesq_in=pesq_wb(ref, xin, sr), pesq_out=pesq_wb(ref, est, sr))
        row.update(estoi_in=estoi(ref, xin, sr), estoi_out=estoi(ref, est, sr))
        if a.srmr:
            row.update(srmr_in=srmr(xin, sr), srmr_out=srmr(est, sr))
        if it["kind"] == "sim":
            row["room_pred"] = ROOM_CLASSES[int(room_logits[-1].argmax())]
            for s in TIME_POINTS:  # online room accuracy vs seconds since start
                f = int(round(s * sr / hop))
                row[f"room_ok@{s}s"] = (int(room_logits[f].argmax()) == it["room"]) if f < T else ""
            embs.append(out["emb"][0, -1].float().cpu().numpy())
            emb_spk.append(row["speaker"])
            emb_room.append(it["room"])
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)

    out_dir = run_dir / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({kk for r in rows for kk in r}, key=lambda s: (s not in rows[0], s))
    with open(out_dir / "per_utt.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metric_names = ["sisdr", "estoi"] + ([] if a.no_pesq else ["pesq"]) + (["srmr"] if a.srmr else [])
    # sim_reverb (small+medium+large) is the main simulated result; sim_anechoic checks "do no harm"
    # (input == target there, so its SI-SDR is not meaningful as an improvement).
    groups = {"sim_reverb": [r for r in rows if r["kind"] == "sim" and r["room"] != "anechoic"],
              "real_all": [r for r in rows if r["kind"] == "real"]}
    for c in ROOM_CLASSES:
        groups[f"sim_{c}"] = [r for r in rows if r["kind"] == "sim" and r["room"] == c]
    for _, _, name in T60_BINS:
        groups[f"real_{name}"] = [r for r in rows if r["kind"] == "real" and r["t60_bin"] == name]
    summary = {"run_dir": str(run_dir), "experiment": cfg.experiment, "seed": cfg.seed, "window": window,
               "n_items": len(rows), "rtf": proc_time / max(audio_time, 1e-9), "groups": {}}
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs)}
        for mname in metric_names:
            d[f"{mname}_in"] = mean([r.get(f"{mname}_in") for r in rs])
            d[f"{mname}_out"] = mean([r.get(f"{mname}_out") for r in rs])
            d[f"{mname}_delta"] = d[f"{mname}_out"] - d[f"{mname}_in"]
        summary["groups"][g] = d
    sim = [r for r in rows if r["kind"] == "sim"]
    if sim:
        summary["room_acc"] = mean([float(r["room_pred"] == r["room"]) for r in sim])
        summary["room_acc_vs_time"] = {f"{s}s": mean([float(r[f"room_ok@{s}s"]) for r in sim
                                                      if r[f"room_ok@{s}s"] != ""]) for s in TIME_POINTS}
        summary["speaker_eer"] = eer(np.stack(embs), emb_spk)
        summary["speaker_eer_by_room"] = {
            c: eer(np.stack([e for e, r in zip(embs, emb_room) if r == ci]),
                   [s for s, r in zip(emb_spk, emb_room) if r == ci])
            for ci, c in enumerate(ROOM_CLASSES) if sum(r == ci for r in emb_room) > 2}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[saved] {out_dir}")
    for g, d in summary["groups"].items():
        print(f"  {g:22s} n={d['n']:4d} " + " ".join(f"{mn}: {d[f'{mn}_in']:.3f}->{d[f'{mn}_out']:.3f}"
                                                     for mn in metric_names))
    for k in ("room_acc", "speaker_eer", "rtf"):
        if k in summary:
            print(f"  {k} = {summary[k]:.4f}")


if __name__ == "__main__":
    main()
