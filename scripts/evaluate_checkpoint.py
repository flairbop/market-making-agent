"""
evaluate_checkpoint.py
----------------------
Load a saved DQN checkpoint and run a full evaluation.

Usage:
    python scripts/evaluate_checkpoint.py
    python scripts/evaluate_checkpoint.py --config configs/eval.yaml
    python scripts/evaluate_checkpoint.py --checkpoint outputs/checkpoints/dqn_best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from market_making_agent.config import load_config
from market_making_agent.env.market_env import MarketMakingEnv
from market_making_agent.agents.dqn_agent import DQNAgent
from market_making_agent.training.evaluate import evaluate_agent
from market_making_agent.utils.metrics import compute_episode_metrics, summarise_metrics
from market_making_agent.utils.plotting import (
    plot_pnl_curve, plot_inventory_path, plot_return_distribution
)
from market_making_agent.utils.io import load_checkpoint, save_json
from market_making_agent.analysis.performance_report import generate_report

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DQN checkpoint.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path from config.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ckpt_path = args.checkpoint or getattr(cfg.evaluation, "checkpoint_path", None)
    if ckpt_path is None:
        ckpt_path = str(Path(cfg.checkpoint_dir) / "dqn_best.pt")

    logger.info(f"Loading checkpoint: {ckpt_path}")
    env = MarketMakingEnv(cfg)
    agent = DQNAgent(cfg, state_dim=env.state_dim, n_actions=env.n_actions, device=args.device)
    ckpt = load_checkpoint(ckpt_path, device=args.device)
    agent.load_checkpoint(ckpt)
    logger.info(f"Checkpoint loaded (step={ckpt.get('steps_done', '?')})")

    n_episodes = int(cfg.evaluation.n_episodes)
    logger.info(f"Running {n_episodes} evaluation episodes...")
    results = evaluate_agent(cfg, agent, n_episodes=n_episodes, device=args.device)

    # Plots
    plot_pnl_curve(
        [e["pnl_path"] for e in results],
        title="DQN — Evaluation Cumulative PnL",
        save_path=Path(cfg.figure_dir) / "pnl_curve_eval.png",
    )
    plot_inventory_path(
        [e["inventory_path"] for e in results],
        max_inventory=cfg.env.max_inventory,
        save_path=Path(cfg.figure_dir) / "inventory_path_eval.png",
    )
    plot_return_distribution(
        [e["total_pnl"] for e in results],
        save_path=Path(cfg.figure_dir) / "return_distribution_eval.png",
    )

    # Metrics
    df = compute_episode_metrics(results)
    summary = summarise_metrics(df)
    save_json(summary, Path(cfg.log_dir) / "eval_summary.json")

    report = generate_report(
        experiment_name=cfg.experiment_name + "_eval",
        dqn_summary=summary,
        baseline_summaries={},
        save_path=Path(cfg.report_dir) / "eval_report.txt",
        n_eval_episodes=n_episodes,
    )
    print(report)


if __name__ == "__main__":
    main()
