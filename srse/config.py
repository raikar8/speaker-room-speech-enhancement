"""YAML configs with `base:` inheritance and `key.sub=value` command-line overrides."""

import copy
import json
from pathlib import Path

import yaml


class DotDict(dict):
    """dict with attribute access (cfg.train.lr)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def to_dot(obj):
    if isinstance(obj, dict):
        return DotDict({k: to_dot(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_dot(v) for v in obj]
    return obj


def to_plain(obj):
    return json.loads(json.dumps(obj))


def deep_update(base, update):
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path):
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(_load_yaml(path.parent / base), cfg)
    return cfg


def apply_overrides(cfg, overrides):
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got: {ov}")
        key, raw = ov.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def load_config(path, overrides=None):
    cfg = apply_overrides(_load_yaml(path), overrides)
    return to_dot(cfg)


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_plain(cfg), f, indent=2)


def load_saved_config(path):
    with open(path, encoding="utf-8") as f:
        return to_dot(json.load(f))
