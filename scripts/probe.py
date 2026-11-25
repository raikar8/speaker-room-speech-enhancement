"""Linear probes: how much speaker / room information does each tap (P1, P2, P3) of a trained model hold?

Features = mean + std over the whole utterance of the raw tap activations (no MTL heads involved).
Probe-train: utterances of the training speakers; probe-test: the held-out validation items.
Run it on B0 to see what a plain enhancement network learns on its own.

  python scripts/probe.py --run_dir /content/drive/MyDrive/srse_runs/B0/seed0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from srse import ROOM_CLASSES  # noqa: E402
from srse.data.dataset import ReverbDataset, load_manifests, make_sim_items, rir_pool, split_speakers  # noqa: E402
from srse.utils import get_device, load_model_for_eval  # noqa: E402

TAPS = ("P1", "P2", "P3")


def probe_train_items(train_utts, spk2idx, pool, per_speaker, seed, crop_len, level_db):
    rng = np.random.default_rng(seed)
    by_spk = {}
    for i, u in enumerate(train_utts):
        by_spk.setdefault(u["speaker"], []).append(i)
    items = []
    for spk in sorted(by_spk):
        idxs = rng.permutation(by_spk[spk])[:per_speaker]
        for i in idxs:
            room = int(rng.integers(0, len(ROOM_CLASSES)))
            rir = None if room == 0 else str(rng.choice(pool[ROOM_CLASSES[room]]))
            n = train_utts[i]["n_frames"]
            items.append(dict(utt=int(i), room=room, rir=rir, start=int(rng.integers(0, max(1, n - crop_len + 1))),
                              length=crop_len, level=level_db, kind="sim", t60=float("nan")))
    return items


@torch.no_grad()
def extract(model, ds, device, batch_size, num_workers):
    feats = {t: [] for t in TAPS}
    spk, room = [], []
    for b in DataLoader(ds, batch_size=batch_size, num_workers=num_workers):
        out = model(b["input"].to(device), room=b["room"].to(device), return_taps=True)
        for t in TAPS:
            z = out["taps"][t].float()
            feats[t].append(torch.cat([z.mean(1), z.std(1)], -1).cpu())
        spk.append(b["spk"])
        room.append(b["room"])
    return {t: torch.cat(v) for t, v in feats.items()}, torch.cat(spk), torch.cat(room)


def fit_probe(xtr, ytr, xte, yte, n_cls, steps, device):
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = ((xtr - mu) / sd).to(device), ((xte - mu) / sd).to(device)
    ytr, yte = ytr.to(device), yte.to(device)
    lin = torch.nn.Linear(xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((lin(xte).argmax(-1) == yte).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--per_speaker", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    model, cfg, _, device = load_model_for_eval(Path(a.run_dir), a.ckpt, get_device(a.device))
    m = load_manifests(cfg)
    train_utts, val_utts, spk2idx = split_speakers(m["train"], cfg)
    seg = int(cfg.data.segment_seconds * cfg.data.sample_rate)
    tr_items = probe_train_items(train_utts, spk2idx, rir_pool(m["rirs_sim"], "train"), a.per_speaker,
                                 cfg.data.split_seed + 7, seg, cfg.data.eval_level_db)
    te_items = make_sim_items(val_utts, rir_pool(m["rirs_sim"], "val"), cfg.data.val_items_per_class,
                              seed=cfg.data.split_seed + 1, crop_len=seg, level_db=cfg.data.eval_level_db)
    bs, nw = cfg.train.batch_size, cfg.data.num_workers
    ftr, s_tr, r_tr = extract(model, ReverbDataset(cfg, train_utts, spk2idx=spk2idx, items=tr_items), device, bs, nw)
    fte, s_te, r_te = extract(model, ReverbDataset(cfg, val_utts, spk2idx=spk2idx, items=te_items), device, bs, nw)

    res = {"chance_speaker": 1 / max(len(spk2idx), 1), "chance_room": 1 / len(ROOM_CLASSES), "taps": {}}
    for t in TAPS:
        res["taps"][t] = {
            "speaker_acc": fit_probe(ftr[t], s_tr, fte[t], s_te, len(spk2idx), a.steps, device),
            "room_acc": fit_probe(ftr[t], r_tr, fte[t], r_te, len(ROOM_CLASSES), a.steps, device),
        }
        print(f"  {t}: speaker acc {res['taps'][t]['speaker_acc']:.3f} | room acc {res['taps'][t]['room_acc']:.3f}")
    out = Path(a.run_dir) / "probe.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
