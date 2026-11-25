"""Create a tiny fake LibriSpeech + RIRS_NOISES tree to smoke-test the whole pipeline in minutes.

  python tests/make_synthetic_data.py --data_root /tmp/srse_fake
  python scripts/prepare_data.py --data_root /tmp/srse_fake --skip_download
  python scripts/run_grid.py --experiments B0 M1 --seeds 0 \
      --set paths.data_root=/tmp/srse_fake paths.manifest_dir=/tmp/srse_fake/manifests \
            paths.out_dir=/tmp/srse_runs train.epochs=1 train.batch_size=4 data.num_workers=0 \
            data.val_items_per_class=4 data.test_utts=3 data.real_rir_test_utts=4
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def fake_speech(rng, f0, seconds):
    t = np.arange(int(seconds * SR)) / SR
    env = (np.sin(2 * np.pi * rng.uniform(2, 5) * t) > 0).astype(np.float32) * 0.9 + 0.1
    sig = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 8))
    return (0.1 * env * sig + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def fake_rir(rng, t60, channels=1):
    n = int(min(1.5, 1.2 * t60) * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal((n, channels)) * np.exp(-6.9 * t / t60)[:, None]
    h[: int(0.003 * SR)] = 0
    h[int(0.003 * SR)] = 1.0
    return h.astype(np.float32).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    rng = np.random.default_rng(0)
    for subset, n_spk, n_utt, base in (("train-clean-100", 6, 8, 100), ("test-clean", 4, 4, 900)):
        for s in range(n_spk):
            spk = str(base + s)
            d = root / "LibriSpeech" / subset / spk / "1"
            d.mkdir(parents=True, exist_ok=True)
            f0 = rng.uniform(90, 250)
            for u in range(n_utt):
                sf.write(d / f"{spk}-1-{u:04d}.flac", fake_speech(rng, f0, rng.uniform(2.5, 6.0)), SR)
    for cls, t60 in (("smallroom", 0.25), ("mediumroom", 0.5), ("largeroom", 0.9)):
        for r in range(1, 11):
            d = root / "RIRS_NOISES" / "simulated_rirs" / cls / f"Room{r:03d}"
            d.mkdir(parents=True, exist_ok=True)
            for k in range(1, 4):
                sf.write(d / f"Room{r:03d}-{k:05d}.wav", fake_rir(rng, t60 * rng.uniform(0.8, 1.2)), SR)
    real = root / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    real.mkdir(parents=True, exist_ok=True)
    for k, t60 in enumerate((0.2, 0.45, 0.8, 1.1)):
        sf.write(real / f"air_type1_fake_{k}.wav", fake_rir(rng, t60, channels=2), SR)
    sf.write(real / "air_type1_fake_noise_0.wav", (0.01 * rng.standard_normal((SR, 2))).astype(np.float32), SR)
    print(f"Synthetic corpus written to {root}")


if __name__ == "__main__":
    main()
