"""
io.py
-----
I/O helpers: directory creation, JSON/CSV saving, checkpoint I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories (including parents) if they do not exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Serialise a dictionary to a JSON file."""
    path = Path(path)
    ensure_dirs(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return as dict."""
    with open(Path(path), "r") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame to CSV."""
    path = Path(path)
    ensure_dirs(path.parent)
    df.to_csv(path, index=False)


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """
    Save a PyTorch model checkpoint dictionary.

    The state dict is expected to contain at least ``model_state_dict``
    and optionally ``optimizer_state_dict``, ``step``, ``episode``.
    """
    path = Path(path)
    ensure_dirs(path.parent)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint from disk."""
    return torch.load(Path(path), map_location=device)
