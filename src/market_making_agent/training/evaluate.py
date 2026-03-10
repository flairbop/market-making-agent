"""
evaluate.py
-----------
Evaluation routines for the DQN agent and baseline strategies.

Runs a fixed number of episodes with deterministic policy (no exploration)
and collects episode-level metrics for reporting and plotting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from market_making_agent.agents.dqn_agent import DQNAgent
from market_making_agent.config import Config
from market_making_agent.env.market_env import MarketMakingEnv

logger = logging.getLogger(__name__)


def evaluate_agent(
    cfg: Config,
    agent: DQNAgent,
    n_episodes: int = 100,
    device: str = "cpu",
    seed_offset: int = 10000,
) -> list[dict[str, Any]]:
    """
    Run the DQN agent for ``n_episodes`` with deterministic policy.

    Parameters
    ----------
    cfg:
        Configuration object.
    agent:
        Trained DQN agent.
    n_episodes:
        Number of evaluation episodes.
    device:
        Torch device.
    seed_offset:
        Added to episode index for a reproducible but independent seed.

    Returns
    -------
    list of episode result dicts.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    env = MarketMakingEnv(cfg, rng=rng)
    results: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        summary = env.get_episode_summary()
        summary["total_reward"] = ep_reward
        results.append(summary)

    return results


def evaluate_baseline(
    cfg: Config,
    agent: Any,  # BaselineAgent duck type
    n_episodes: int = 100,
    seed_offset: int = 20000,
) -> list[dict[str, Any]]:
    """
    Run a baseline agent for ``n_episodes``.

    Parameters
    ----------
    agent:
        Baseline agent with ``select_action(state, n_actions) -> int``.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    env = MarketMakingEnv(cfg, rng=rng)
    results: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, env.n_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        summary = env.get_episode_summary()
        summary["total_reward"] = ep_reward
        results.append(summary)

    return results
