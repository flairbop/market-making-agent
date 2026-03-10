"""
config.py
---------
Loads and validates YAML configuration files into a structured Python object.

Design philosophy:
  - Use a simple namespace-based config object (no Pydantic overhead).
  - Provide sensible defaults so partial configs remain valid.
  - Keep config loading deterministic and loggable.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Default configuration values
# These act as a fallback when a YAML key is absent.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "experiment_name": "default_experiment",
    "output_dir": "outputs",
    "checkpoint_dir": "outputs/checkpoints",
    "log_dir": "outputs/logs",
    "figure_dir": "outputs/figures",
    "report_dir": "outputs/reports",
    "env": {
        "episode_length": 500,
        "dt": 1.0,
        "initial_price": 100.0,
        "price_volatility": 0.02,
        "price_drift": 0.0,
        "tick_size": 0.01,
        "max_inventory": 10,
        "inventory_penalty": 0.01,
        "transaction_cost": 0.001,
        "adverse_selection_factor": 0.3,
        "quote_offsets": [0.01, 0.02, 0.04, 0.06, 0.10],
    },
    "fill_model": {
        "base_fill_rate": 0.5,
        "decay_rate": 10.0,
    },
    "agent": {
        "hidden_layers": [128, 128],
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 50000,
        "replay_buffer_capacity": 100000,
        "batch_size": 256,
        "target_update_freq": 500,
        "double_dqn": True,
        "dueling": True,
        "gradient_clip": 1.0,
    },
    "training": {
        "total_steps": 200000,
        "warmup_steps": 2000,
        "eval_every_steps": 10000,
        "save_every_steps": 25000,
        "eval_episodes": 30,
        "log_every_steps": 1000,
    },
    "evaluation": {
        "n_episodes": 100,
        "deterministic": True,
        "checkpoint_path": None,
    },
}


class Config:
    """
    Lightweight recursive config object.

    Wraps a nested dictionary so that keys are accessible as attributes::

        cfg = load_config("configs/default.yaml")
        print(cfg.env.episode_length)
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain dictionary."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merges ``override`` into a copy of ``base``.
    Leaf values in ``override`` win over ``base``.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(path: str | Path) -> Config:
    """
    Load a YAML config file and merge it with defaults.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Fully merged configuration as an attribute-accessible object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(_DEFAULTS, user_cfg)
    return Config(merged)
