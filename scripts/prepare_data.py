"""Download LibriSpeech (train-clean-100, test-clean) + RIRS_NOISES and write manifests.

Example (Colab):
  python scripts/prepare_data.py --data_root /content/data \
      --archive_dir /content/drive/MyDrive/srse_archives
Keeping archives on Drive means later sessions only re-extract (no re-download).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from srse.data.prepare import prepare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--manifest_dir", default=None, help="default: <data_root>/manifests")
    ap.add_argument("--archive_dir", default=None, help="default: <data_root>/archives")
    ap.add_argument("--skip_download", action="store_true", help="corpora already extracted under data_root")
    ap.add_argument("--delete_archives", action="store_true")
    ap.add_argument("--split_seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.data_root)
    prepare(root, a.manifest_dir or root / "manifests", a.archive_dir or root / "archives",
            skip_download=a.skip_download, split_seed=a.split_seed, delete_archives=a.delete_archives)


if __name__ == "__main__":
    main()
