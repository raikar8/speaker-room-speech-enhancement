"""Train one experiment / seed, validating every epoch. Resumes automatically from last.pt.

  python scripts/train.py --config configs/experiments/M1.yaml --set seed=1 train.epochs=20
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from srse.config import load_config, save_config  # noqa: E402
from srse.data.dataset import build_train_val, worker_init_fn  # noqa: E402
from srse.losses import se_loss, si_sdr_torch  # noqa: E402
from srse.models.crn import SEModel  # noqa: E402
from srse.models.heads import frame_ce  # noqa: E402
from srse.utils import JsonlLogger, count_params, get_device, run_dir_for, set_seed  # noqa: E402


@torch.no_grad()
def validate(model, loader, cfg, device, amp):
    model.eval()
    tot = dict(se=0.0, sisdr_in=0.0, sisdr_out=0.0, n_rev=0, spk_ok=0, spk_n=0, room_ok=0, room_n=0, n=0)
    for b in loader:
        x, y = b["input"].to(device), b["target"].to(device)
        room = b["room"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x, room=room, window=cfg.mtl.eval_pool_frames)
        bs = x.shape[0]
        tot["se"] += se_loss(out, y, model.stft, cfg.loss).item() * bs
        rev = room > 0  # SI-SDR only on reverberant items (anechoic input == target)
        if rev.any():
            tot["sisdr_in"] += si_sdr_torch(x[rev], y[rev]).sum().item()
            tot["sisdr_out"] += si_sdr_torch(out["est_wav"].float()[rev], y[rev]).sum().item()
            tot["n_rev"] += int(rev.sum())
        spk_pred = out["spk_logits"][:, -1].argmax(-1).cpu()
        room_pred = out["room_logits"][:, -1].argmax(-1).cpu()
        known = b["spk"] >= 0
        tot["spk_ok"] += (spk_pred[known] == b["spk"][known]).sum().item()
        tot["spk_n"] += known.sum().item()
        tot["room_ok"] += (room_pred == b["room"]).sum().item()
        tot["room_n"] += bs
        tot["n"] += bs
    n, nr = max(tot["n"], 1), max(tot["n_rev"], 1)
    return {
        "val_se_loss": tot["se"] / n,
        "val_sisdr_in": tot["sisdr_in"] / nr,
        "val_sisdr_out": tot["sisdr_out"] / nr,
        "val_sisdri": (tot["sisdr_out"] - tot["sisdr_in"]) / nr,
        "val_spk_acc": tot["spk_ok"] / max(tot["spk_n"], 1),
        "val_room_acc": tot["room_ok"] / max(tot["room_n"], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. seed=1 train.epochs=10")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0, help="debug: cap steps per epoch")
    a = ap.parse_args()

    cfg = load_config(a.config, a.set)
    set_seed(cfg.seed)
    device = get_device(a.device)
    amp = bool(cfg.train.amp) and device.type == "cuda"
    run_dir = run_dir_for(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.json")
    log = JsonlLogger(run_dir / "metrics.jsonl")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, spk2idx = build_train_val(cfg)
    print(f"[data] train utts={len(train_ds.utts)} speakers={len(spk2idx)} val items={len(val_ds)}")
    nw = cfg.data.num_workers
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=nw,
                          drop_last=True, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn,
                          persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=nw)

    model = SEModel(cfg, len(spk2idx)).to(device)
    print(f"[model] {count_params(model) / 1e6:.2f} M parameters, tap={cfg.mtl.tap}, "
          f"lambda_speaker={cfg.mtl.lambda_speaker}, lambda_room={cfg.mtl.lambda_room}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                       patience=cfg.train.lr_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best, bad = 0, math.inf, 0
    last = run_dir / "last.pt"
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best, bad = st["epoch"] + 1, st["best"], st["bad_epochs"]
        print(f"[resume] from epoch {start_epoch}, best val SE loss {best:.4f}")
    if start_epoch >= cfg.train.epochs or bad >= cfg.train.early_stop_patience:
        print("[done] training already finished")
        return

    lam_s, lam_r = float(cfg.mtl.lambda_speaker), float(cfg.mtl.lambda_room)
    pool_choices = list(cfg.mtl.pool_frames)
    py_rng = random.Random(cfg.seed)
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        sums = dict(loss=0.0, se=0.0, spk=0.0, room=0.0)
        steps = 0
        for b in train_dl:
            x, y = b["input"].to(device, non_blocking=True), b["target"].to(device, non_blocking=True)
            room = b["room"].to(device)
            window = py_rng.choice(pool_choices)
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x, room=room, window=window, lam_spk=lam_s, lam_room=lam_r)
            l_se = se_loss(out, y, model.stft, cfg.loss)
            l_spk = frame_ce(out["spk_logits"], b["spk_target"].to(device), cfg.mtl.min_frames)
            l_room = frame_ce(out["room_logits"], b["room_target"].to(device), cfg.mtl.min_frames)
            # heads always learn (weight 1); lambda only scales what reaches the backbone (GradScale)
            loss = l_se + l_spk + l_room
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in (("loss", loss), ("se", l_se), ("spk", l_spk), ("room", l_room)):
                sums[k] += v.item()
            steps += 1
            if steps % cfg.train.log_every == 0:
                print(f"  ep {epoch} step {steps}/{len(train_dl)} "
                      + " ".join(f"{k}={v / steps:.4f}" for k, v in sums.items()), flush=True)
            if a.max_steps_per_epoch and steps >= a.max_steps_per_epoch:
                break

        val = validate(model, val_dl, cfg, device, amp)
        sched.step(val["val_se_loss"])
        improved = val["val_se_loss"] < best - 1e-5
        if improved:
            best, bad = val["val_se_loss"], 0
        else:
            bad += 1
        row = dict(epoch=epoch, steps=steps, lr=opt.param_groups[0]["lr"], secs=round(time.time() - t0, 1),
                   **{f"train_{k}": v / max(steps, 1) for k, v in sums.items()}, **val)
        log.log(**row)
        print(f"[epoch {epoch}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in row.items()), flush=True)
        state = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                     scaler=scaler.state_dict(), epoch=epoch, best=best, bad_epochs=bad,
                     n_speakers=len(spk2idx), spk2idx=spk2idx)
        torch.save(state, last)
        if improved:
            torch.save(state, run_dir / "best.pt")
            print(f"  new best (val SE loss {best:.4f}) -> best.pt")
        if bad >= cfg.train.early_stop_patience:
            print(f"[early stop] no improvement for {bad} epochs")
            break
    print(f"[done] best val SE loss {best:.4f}; run dir {run_dir}")


if __name__ == "__main__":
    main()
