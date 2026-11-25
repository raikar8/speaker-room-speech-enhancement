"""Download LibriSpeech + RIRS_NOISES (OpenSLR 12 / 28) and write JSONL manifests.

Layout after extraction (relative to data_root):
  LibriSpeech/train-clean-100/<spk>/<chapter>/*.flac
  LibriSpeech/test-clean/...
  RIRS_NOISES/simulated_rirs/{smallroom,mediumroom,largeroom}/RoomXXX/*.wav
  RIRS_NOISES/real_rirs_isotropic_noises/*.wav
"""

import json
import random
import sys
import tarfile
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from srse.data.rir import estimate_t60, load_rir

ARCHIVES = {
    "train-clean-100": ("train-clean-100.tar.gz", "12", "LibriSpeech/train-clean-100"),
    "test-clean": ("test-clean.tar.gz", "12", "LibriSpeech/test-clean"),
    "rirs_noises": ("rirs_noises.zip", "28", "RIRS_NOISES"),
}
MIRRORS = ["https://www.openslr.org/resources", "https://us.openslr.org/resources",
           "https://openslr.elda.org/resources"]
SIM_CLASSES = {"small": "smallroom", "medium": "mediumroom", "large": "largeroom"}


def _progress(block, block_size, total):
    if total > 0 and block % 200 == 0:
        done = min(block * block_size, total)
        sys.stdout.write(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB")
        sys.stdout.flush()


def download(name, archive_dir):
    fname, res, _ = ARCHIVES[name]
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / fname
    if dest.exists():
        print(f"[skip] {dest} already downloaded")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for mirror in MIRRORS:
        url = f"{mirror}/{res}/{fname}"
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, tmp, _progress)
            print()
            tmp.rename(dest)
            return dest
        except Exception as e:  # try the next mirror
            print(f"\n  failed: {e}")
    raise RuntimeError(f"Could not download {fname} from any mirror")


def extract(name, archive, data_root):
    _, _, marker = ARCHIVES[name]
    if (data_root / marker).exists():
        print(f"[skip] {marker} already extracted")
        return
    print(f"Extracting {archive.name} ...")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(data_root)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(data_root, filter="data")
            except TypeError:  # Python < 3.12
                t.extractall(data_root)


def libri_manifest(data_root, subset):
    base = data_root / "LibriSpeech" / subset
    files = sorted(base.rglob("*.flac"))
    if not files:
        raise RuntimeError(f"No .flac files under {base}")

    def row(p):
        info = sf.info(str(p))
        return dict(path=p.relative_to(data_root).as_posix(), speaker=p.parent.parent.name,
                    chapter=p.parent.name, n_frames=int(info.frames), sr=int(info.samplerate))

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(row, files))


def sim_rir_manifest(data_root, split_seed=0, fractions=(0.8, 0.1, 0.1)):
    """Simulated RIRs labelled small/medium/large; split train/val/test by ROOM so test rooms are unseen."""
    rows = []
    base = data_root / "RIRS_NOISES" / "simulated_rirs"
    for cls, folder in SIM_CLASSES.items():
        rooms = sorted(d for d in (base / folder).iterdir() if d.is_dir())
        if not rooms:
            raise RuntimeError(f"No rooms under {base / folder}")
        random.Random(split_seed).shuffle(rooms)
        n_tr = int(round(fractions[0] * len(rooms)))
        n_va = max(1, int(round(fractions[1] * len(rooms))))
        for i, room in enumerate(rooms):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for wav in sorted(room.glob("*.wav")):
                rows.append(dict(path=wav.relative_to(data_root).as_posix(), room_class=cls,
                                 room_id=f"{folder}/{room.name}", split=split))
    return rows


def real_rir_manifest(data_root, sr=16000):
    """Measured RIRs from RIRS_NOISES (test only). Room size unknown; T60 estimated for binning."""
    real_dir = data_root / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    paths = []
    list_file = real_dir / "rir_list"
    if list_file.exists():
        for line in list_file.read_text().splitlines():
            for tok in line.split():
                if tok.endswith(".wav"):
                    p = data_root / tok if tok.startswith("RIRS_NOISES") else real_dir / Path(tok).name
                    if p.exists():
                        paths.append(p)
    if not paths:
        paths = [p for p in sorted(real_dir.glob("*.wav")) if "noise" not in p.name.lower()]
    paths = sorted(set(paths))
    rows = []
    for p in paths:
        t60 = estimate_t60(load_rir(p, sr, max_seconds=4.0), sr)
        rows.append(dict(path=p.relative_to(data_root).as_posix(), source=p.name.split("_")[0],
                         t60=t60, split="test"))
    return rows


def write_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows):,} rows -> {path}")


def prepare(data_root, manifest_dir, archive_dir, skip_download=False, split_seed=0, delete_archives=False):
    data_root, manifest_dir, archive_dir = Path(data_root), Path(manifest_dir), Path(archive_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    if not skip_download:
        for name in ARCHIVES:
            archive = download(name, archive_dir)
            extract(name, archive, data_root)
            if delete_archives:
                archive.unlink()
    for subset in ("train-clean-100", "test-clean"):
        write_jsonl(libri_manifest(data_root, subset), manifest_dir / f"libri_{subset}.jsonl")
    write_jsonl(sim_rir_manifest(data_root, split_seed), manifest_dir / "rirs_sim.jsonl")
    write_jsonl(real_rir_manifest(data_root), manifest_dir / "rirs_real.jsonl")
