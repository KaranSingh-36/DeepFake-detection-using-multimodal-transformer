"""YAML configuration with dotted overrides.

    cfg = load_config("configs/default.yaml", overrides=["train.lr=1e-4"])
    cfg.train.lr            # attribute access
    cfg["train"]["lr"]      # dict access
"""

import copy
import json
from pathlib import Path

import yaml

from fusiontrace import PROJECT_ROOT

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return value

    def __setattr__(self, name, value):
        self[name] = value

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_plain(self):
        return json.loads(json.dumps(self))


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_value(text):
    """'1e-4' -> float, 'true' -> True, '[6,12]' -> list, 'abc' -> 'abc'."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_override(config, dotted_key, value):
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def load_config(path=None, overrides=None):
    with open(DEFAULT_CONFIG, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path, "r", encoding="utf-8") as handle:
            config = deep_merge(config, yaml.safe_load(handle) or {})

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")
        key, text = item.split("=", 1)
        apply_override(config, key.strip(), parse_value(text.strip()))

    return Cfg.wrap(config)


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.to_plain() if isinstance(config, Cfg) else config,
            handle,
            sort_keys=False,
        )
