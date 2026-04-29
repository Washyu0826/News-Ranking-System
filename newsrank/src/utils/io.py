"""I/O helpers for NewsRank."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load YAML config and raise clear error if missing."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found at {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
