"""
train_default.py
----------------
Script to run a full training experiment from the command line.

Usage:
    python scripts/train_default.py
    python scripts/train_default.py --config configs/train_fast.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from market_making_agent.config import load_config
from market_making_agent.training.run_experiment import main as run_main

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

if __name__ == "__main__":
    run_main()
