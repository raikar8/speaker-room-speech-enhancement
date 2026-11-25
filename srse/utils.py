import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from srse.config import load_saved_config
from srse.models.crn import SEModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir_for(cfg):
    return Path(cfg.paths.out_dir) / cfg.experiment / f"seed{cfg.seed}"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **row):
        row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def load_model_for_eval(run_dir, ckpt="best.pt", device=None):
    run_dir = Path(run_dir)
    cfg = load_saved_config(run_dir / "config.json")
    state = torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)
    model = SEModel(cfg, state["n_speakers"])
    model.load_state_dict(state["model"])
    device = device or get_device()
    model.to(device).eval()
    return model, cfg, state, device
