"""
compare_baselines.py
--------------------
Evaluate all baseline strategies and print a comparison table.

Usage:
    python scripts/compare_baselines.py
    python scripts/compare_baselines.py --config configs/eval.yaml --n-episodes 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from market_making_agent.config import load_config
from market_making_agent.agents.baselines import get_all_baselines
from market_making_agent.training.evaluate import evaluate_baseline
from market_making_agent.utils.metrics import compute_episode_metrics, summarise_metrics
from market_making_agent.utils.plotting import plot_baseline_comparison
from market_making_agent.utils.io import save_json

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--n-episodes", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_episodes = args.n_episodes or int(cfg.evaluation.n_episodes)

    all_summaries: dict[str, dict] = {}
    baselines = get_all_baselines()

    for agent in baselines:
        logger.info(f"Evaluating baseline: {agent.name}")
        episodes = evaluate_baseline(cfg, agent, n_episodes=n_episodes)
        df = compute_episode_metrics(episodes)
        summary = summarise_metrics(df)
        all_summaries[agent.name] = summary
        logger.info(
            f"  {agent.name}: mean_pnl={summary['total_pnl_mean']:.4f} ± "
            f"{summary['total_pnl_std']:.4f}, sharpe={summary['sharpe_ratio']:.4f}"
        )

    plot_baseline_comparison(
        all_summaries,
        metric="total_pnl_mean",
        title="Baseline Strategy Comparison — Mean PnL",
        save_path=Path(cfg.figure_dir) / "baseline_comparison.png",
    )

    save_json(all_summaries, Path(cfg.log_dir) / "baseline_results.json")
    logger.info("Baseline comparison complete.")
    logger.info(f"Figure saved to {cfg.figure_dir}/baseline_comparison.png")


if __name__ == "__main__":
    main()
