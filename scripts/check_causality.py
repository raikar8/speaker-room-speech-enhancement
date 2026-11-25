"""Verify the model is causal (real-time capable) and measure its speed.

Changes the input from frame t0 onward and checks that every output before t0 (mask, embedding,
head logits) is unchanged. Works on a random-init model from a config, or a trained run.

  python scripts/check_causality.py --config configs/experiments/M1.yaml
  python scripts/check_causality.py --run_dir /content/drive/MyDrive/srse_runs/M1/seed0
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from srse.config import load_config  # noqa: E402
from srse.models.crn import SEModel  # noqa: E402
from srse.utils import count_params, get_device, load_model_for_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = get_device(a.device)
    if a.run_dir:
        model, cfg, _, device = load_model_for_eval(Path(a.run_dir), device=device)
    else:
        cfg = load_config(a.config, a.set)
        model = SEModel(cfg, 251).to(device).eval()
    sr, hop, n_fft = cfg.data.sample_rate, cfg.stft.hop, cfg.stft.n_fft

    torch.manual_seed(0)
    wav = torch.randn(1, 3 * sr, device=device) * 0.05
    room = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        _, feat = model.features(wav)
        T = feat.shape[2]
        t0 = T // 2
        feat2 = feat.clone()
        feat2[:, :, t0:] = torch.randn_like(feat2[:, :, t0:])
        o1 = model.core(feat, room, window=0)
        o2 = model.core(feat2, room, window=0)
    ok = True
    for key in ("mask_raw", "emb", "spk_logits", "room_logits"):
        a1, a2 = o1[key], o2[key]
        past = (a1[:, :, :t0] - a2[:, :, :t0]) if key == "mask_raw" else (a1[:, :t0] - a2[:, :t0])
        future = (a1[:, :, t0:] - a2[:, :, t0:]) if key == "mask_raw" else (a1[:, t0:] - a2[:, t0:])
        d_past, d_future = past.abs().max().item(), future.abs().max().item()
        status = "OK" if d_past < 1e-4 else "LEAK"
        ok &= d_past < 1e-4
        print(f"  {key:12s} max change before t0: {d_past:.2e} ({status}); after t0: {d_future:.2e}")
    print(f"[causality] {'PASS' if ok else 'FAIL'} - outputs before the perturbed frame are "
          f"{'unchanged' if ok else 'CHANGED'}")

    print(f"[model] {count_params(model) / 1e6:.2f} M parameters | algorithmic latency "
          f"{1000 * n_fft / sr:.0f} ms (window {1000 * n_fft / sr:.0f} ms, hop {1000 * hop / sr:.0f} ms)")
    x = torch.randn(1, 10 * sr, device=device) * 0.05
    with torch.no_grad():
        for _ in range(2):
            model(x, room=room)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            model(x, room=room)
        if device.type == "cuda":
            torch.cuda.synchronize()
    rtf = (time.perf_counter() - t) / 5 / 10.0
    print(f"[speed] real-time factor on {device.type}: {rtf:.4f} (offline pass over 10 s; < 1 = faster than real time)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
