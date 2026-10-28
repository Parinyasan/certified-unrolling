from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict, n: int = 10) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:n]
