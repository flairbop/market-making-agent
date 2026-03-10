"""
train.py
--------
Core training loop for the DQN market-making agent.

The training loop follows the standard DQN recipe:
  1. Collect experience by running the agent in the environment.
  2. Once the replay buffer is warm, sample minibatches and train.
  3. Periodically sync the target network.
  4. Periodically evaluate and save checkpoints.
  5. Log metrics to CSV for analysis.

Training continues for a fixed number of environment steps (total_steps),
not episodes, so that comparisons across runs are step-consistent.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from market_making_agent.agents.dqn_agent import DQNAgent
from market_making_agent.config import Config
from market_making_agent.env.market_env import MarketMakingEnv
from market_making_agent.training.evaluate import evaluate_agent
from market_making_agent.utils.io import ensure_dirs, save_checkpoint, save_dataframe
from market_making_agent.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def train(cfg: Config, device: str = "cpu") -> tuple[DQNAgent, list[dict]]:
    """
    Execute the full DQN training loop.

    Parameters
    ----------
    cfg:
        Full configuration object.
    device:
        PyTorch compute device.

    Returns
    -------
    agent:
        The trained DQN agent.
    training_log:
        List of per-step/episode log dicts for post-hoc analysis.
    """
    set_global_seed(cfg.seed)
    ensure_dirs(cfg.checkpoint_dir, cfg.log_dir, cfg.figure_dir, cfg.report_dir)

    rng = np.random.default_rng(cfg.seed)
    env = MarketMakingEnv(cfg, rng=rng)
    agent = DQNAgent(cfg, state_dim=env.state_dim, n_actions=env.n_actions, device=device)

    tcfg = cfg.training
    total_steps = int(tcfg.total_steps)
    warmup_steps = int(tcfg.warmup_steps)
    target_update_freq = int(cfg.agent.target_update_freq)
    eval_every = int(tcfg.eval_every_steps)
    save_every = int(tcfg.save_every_steps)
    log_every = int(tcfg.log_every_steps)

    training_log: list[dict] = []
    eval_log: list[dict] = []

    # Episode-level accumulators
    episode_rewards: list[float] = []
    episode_pnls: list[float] = []

    obs, _ = env.reset()
    ep_reward = 0.0
    ep_steps = 0
    start_time = time.time()
    best_eval_pnl = -np.inf

    logger.info(f"Starting training: {total_steps} steps, seed={cfg.seed}")

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for global_step in range(1, total_steps + 1):
            # --- Action selection ---
            if global_step <= warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)

            # --- Environment step ---
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            ep_steps += 1

            # --- Gradient update ---
            if global_step > warmup_steps:
                loss = agent.train_step()

                # Target network sync
                if global_step % target_update_freq == 0:
                    agent.sync_target_network()

            # --- Episode end ---
            if done:
                ep_info = info.get("episode", env.get_episode_summary())
                episode_rewards.append(ep_reward)
                episode_pnls.append(ep_info["total_pnl"])

                if global_step % log_every < ep_steps:
                    entry = {
                        "global_step": global_step,
                        "episode": len(episode_rewards),
                        "episode_reward": ep_reward,
                        "episode_pnl": ep_info["total_pnl"],
                        "fill_count": ep_info["fill_count"],
                        "epsilon": agent.epsilon,
                        "loss": agent.last_loss,
                        "buffer_size": agent.buffer_size,
                    }
                    training_log.append(entry)

                obs, _ = env.reset()
                ep_reward = 0.0
                ep_steps = 0

            # --- Periodic evaluation ---
            if global_step % eval_every == 0:
                eval_results = evaluate_agent(
                    cfg=cfg,
                    agent=agent,
                    n_episodes=int(tcfg.eval_episodes),
                    device=device,
                )
                mean_pnl = float(np.mean([e["total_pnl"] for e in eval_results]))
                eval_log.append({"global_step": global_step, "eval_mean_pnl": mean_pnl})
                logger.info(
                    f"Step {global_step:>7d} | eval_mean_pnl={mean_pnl:.4f} | ε={agent.epsilon:.3f}"
                )

                # Save best checkpoint
                if mean_pnl > best_eval_pnl:
                    best_eval_pnl = mean_pnl
                    ckpt_path = Path(cfg.checkpoint_dir) / "dqn_best.pt"
                    save_checkpoint(
                        agent.get_checkpoint(extra={"step": global_step, "eval_mean_pnl": mean_pnl}),
                        ckpt_path,
                    )

            # --- Periodic checkpoint save ---
            if global_step % save_every == 0:
                ckpt_path = Path(cfg.checkpoint_dir) / f"dqn_step_{global_step}.pt"
                save_checkpoint(agent.get_checkpoint(extra={"step": global_step}), ckpt_path)

            pbar.update(1)
            if global_step % log_every == 0:
                elapsed = time.time() - start_time
                pbar.set_postfix(
                    {
                        "eps": f"{agent.epsilon:.3f}",
                        "loss": f"{agent.last_loss:.4f}",
                        "buf": agent.buffer_size,
                        "elapsed": f"{elapsed:.0f}s",
                    }
                )

    # --- Save training log ---
    if training_log:
        df_log = pd.DataFrame(training_log)
        save_dataframe(df_log, Path(cfg.log_dir) / f"{cfg.experiment_name}_train_log.csv")

    if eval_log:
        df_eval = pd.DataFrame(eval_log)
        save_dataframe(df_eval, Path(cfg.log_dir) / f"{cfg.experiment_name}_eval_log.csv")

    logger.info(f"Training complete. Best eval PnL: {best_eval_pnl:.4f}")
    return agent, training_log
