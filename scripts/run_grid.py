"""Train + test a list of experiments x seeds, skipping runs that are already tested.

Safe to re-run after a Colab disconnect: finished runs are skipped and unfinished ones resume.

  python scripts/run_grid.py --experiments B0 S2 R2 M1 --seeds 0 1 2
  python scripts/run_grid.py --experiments B0 --seeds 0 --set train.epochs=2 data.utts_per_speaker=5
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srse.config import load_config  # noqa: E402
from srse.utils import run_dir_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every run")
    ap.add_argument("--test_args", default="", help='extra args for test.py, e.g. "--no_pesq"')
    ap.add_argument("--probe", action="store_true", help="also run linear probes after testing")
    a = ap.parse_args()

    for exp in a.experiments:
        cfg_path = ROOT / "configs" / "experiments" / f"{exp}.yaml"
        for seed in a.seeds:
            overrides = [f"seed={seed}"] + a.set
            run_dir = run_dir_for(load_config(cfg_path, overrides))
            if (run_dir / "test" / "summary.json").exists():
                print(f"[skip] {exp} seed {seed}: already tested")
                continue
            print(f"\n===== {exp} | seed {seed} =====", flush=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(cfg_path),
                            "--set", *overrides], check=True)
            subprocess.run([sys.executable, str(ROOT / "scripts" / "test.py"), "--run_dir", str(run_dir),
                            *a.test_args.split()], check=True)
            if a.probe:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "probe.py"), "--run_dir", str(run_dir)],
                               check=True)


if __name__ == "__main__":
    main()
