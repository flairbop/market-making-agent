"""
run_experiment.py
-----------------
Top-level entry point for training and evaluation experiments.

Usage:
    python -m market_making_agent.training.run_experiment --config configs/default.yaml

Or via the installed script:
    mma-train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from market_making_agent.config import load_config
from market_making_agent.training.train import train
from market_making_agent.training.evaluate import evaluate_agent, evaluate_baseline
from market_making_agent.agents.baselines import get_all_baselines
from market_making_agent.utils.metrics import compute_episode_metrics, summarise_metrics
from market_making_agent.utils.plotting import (
    plot_pnl_curve,
    plot_inventory_path,
    plot_reward_curve,
    plot_baseline_comparison,
    plot_return_distribution,
)
from market_making_agent.utils.io import save_json
from market_making_agent.analysis.performance_report import generate_report

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL market-making experiment.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="PyTorch device (cpu or cuda)."
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Skip training; load checkpoint and evaluate."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    logger.info(f"Experiment: {cfg.experiment_name} | device={device}")

    # ---- Training ----
    if not args.eval_only:
        logger.info("=== Phase 1: Training ===")
        agent, train_log = train(cfg, device=device)

        # Plot training reward curve
        if train_log:
            ep_rewards = [e["episode_reward"] for e in train_log]
            plot_reward_curve(
                ep_rewards,
                title=f"Training Reward — {cfg.experiment_name}",
                save_path=Path(cfg.figure_dir) / "reward_curve.png",
            )
    else:
        # Load existing checkpoint
        from market_making_agent.agents.dqn_agent import DQNAgent
        from market_making_agent.env.market_env import MarketMakingEnv
        from market_making_agent.utils.io import load_checkpoint

        env = MarketMakingEnv(cfg)
        agent = DQNAgent(cfg, state_dim=env.state_dim, n_actions=env.n_actions, device=device)
        ckpt_path = getattr(cfg.evaluation, "checkpoint_path", None) or str(
            Path(cfg.checkpoint_dir) / "dqn_best.pt"
        )
        ckpt = load_checkpoint(ckpt_path, device=device)
        agent.load_checkpoint(ckpt)
        logger.info(f"Loaded checkpoint from {ckpt_path}")
        train_log = []

    # ---- Evaluation: DQN ----
    logger.info("=== Phase 2: Evaluating DQN Agent ===")
    n_eval = int(cfg.evaluation.n_episodes)
    dqn_episodes = evaluate_agent(cfg, agent, n_episodes=n_eval, device=device)

    dqn_pnl_paths = [e["pnl_path"] for e in dqn_episodes]
    dqn_inv_paths = [e["inventory_path"] for e in dqn_episodes]

    plot_pnl_curve(
        dqn_pnl_paths,
        title="DQN — Cumulative PnL (Evaluation)",
        save_path=Path(cfg.figure_dir) / "pnl_curve.png",
    )
    plot_inventory_path(
        dqn_inv_paths,
        max_inventory=cfg.env.max_inventory,
        title="DQN — Inventory Path (Evaluation)",
        save_path=Path(cfg.figure_dir) / "inventory_path.png",
    )
    plot_return_distribution(
        [e["total_pnl"] for e in dqn_episodes],
        label="DQN",
        save_path=Path(cfg.figure_dir) / "return_distribution.png",
    )

    dqn_metrics_df = compute_episode_metrics(dqn_episodes)
    dqn_summary = summarise_metrics(dqn_metrics_df)

    # ---- Evaluation: Baselines ----
    logger.info("=== Phase 3: Evaluating Baselines ===")
    baselines = get_all_baselines()
    all_results: dict[str, dict] = {"DQN": dqn_summary}

    for baseline in baselines:
        bl_episodes = evaluate_baseline(cfg, baseline, n_episodes=n_eval)
        bl_metrics = compute_episode_metrics(bl_episodes)
        bl_summary = summarise_metrics(bl_metrics)
        all_results[baseline.name] = bl_summary
        logger.info(f"{baseline.name}: mean_pnl={bl_summary['total_pnl_mean']:.4f}")

    # Baseline comparison plot
    plot_baseline_comparison(
        all_results,
        metric="total_pnl_mean",
        title="Baseline Comparison — Mean Episode PnL",
        save_path=Path(cfg.figure_dir) / "baseline_comparison.png",
    )

    # Save results JSON
    save_json(all_results, Path(cfg.log_dir) / f"{cfg.experiment_name}_results.json")

    # ---- Report ----
    logger.info("=== Phase 4: Generating Report ===")
    generate_report(
        experiment_name=cfg.experiment_name,
        dqn_summary=dqn_summary,
        baseline_summaries={k: v for k, v in all_results.items() if k != "DQN"},
        save_path=Path(cfg.report_dir) / f"{cfg.experiment_name}_report.txt",
    )

    logger.info("Experiment complete. Outputs saved to ./outputs/")


if __name__ == "__main__":
    main()
