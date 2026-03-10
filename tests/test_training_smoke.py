"""
test_training_smoke.py
----------------------
Smoke test: verify that a very short training run completes without errors.

This does not test correctness of the learned policy — only that the
entire training pipeline can execute end-to-end.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.config import Config, _DEFAULTS, _deep_merge
from market_making_agent.training.train import train
from market_making_agent.training.evaluate import evaluate_agent, evaluate_baseline
from market_making_agent.agents.baselines import FixedSpreadAgent, RandomAgent


@pytest.fixture
def smoke_cfg() -> Config:
    """Minimal config for a very fast smoke test."""
    overrides = {
        "seed": 0,
        "output_dir": "/tmp/mma_smoke",
        "checkpoint_dir": "/tmp/mma_smoke/checkpoints",
        "log_dir": "/tmp/mma_smoke/logs",
        "figure_dir": "/tmp/mma_smoke/figures",
        "report_dir": "/tmp/mma_smoke/reports",
        "env": {
            "episode_length": 20,
            "max_inventory": 5,
        },
        "agent": {
            "hidden_layers": [32, 32],
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay_steps": 100,
            "replay_buffer_capacity": 500,
            "batch_size": 16,
            "target_update_freq": 50,
        },
        "training": {
            "total_steps": 200,
            "warmup_steps": 30,
            "eval_every_steps": 100,
            "save_every_steps": 200,
            "eval_episodes": 3,
            "log_every_steps": 50,
        },
        "evaluation": {
            "n_episodes": 5,
            "deterministic": True,
        },
    }
    return Config(_deep_merge(_DEFAULTS, overrides))


def test_training_smoke(smoke_cfg):
    """Full training loop should complete without raising any exceptions."""
    agent, log = train(smoke_cfg, device="cpu")
    assert agent is not None
    # Replay buffer should have been populated
    assert agent.buffer_size > 0


def test_evaluation_runs_after_training(smoke_cfg):
    """Evaluation should return results after training."""
    agent, _ = train(smoke_cfg, device="cpu")
    results = evaluate_agent(smoke_cfg, agent, n_episodes=3)
    assert len(results) == 3
    for ep in results:
        assert "total_pnl" in ep
        assert "inventory_path" in ep
        assert "pnl_path" in ep


def test_baseline_evaluation_smoke(smoke_cfg):
    """Baseline evaluation should complete without errors."""
    agent = FixedSpreadAgent(action_index=1)
    results = evaluate_baseline(smoke_cfg, agent, n_episodes=3)
    assert len(results) == 3


def test_replay_buffer_fills(smoke_cfg):
    """After warmup_steps, replay buffer should be ready for sampling."""
    agent, _ = train(smoke_cfg, device="cpu")
    warmup = smoke_cfg.training.warmup_steps
    # Buffer should have at least warmup_steps transitions
    assert agent.buffer_size >= min(warmup, smoke_cfg.training.total_steps)


def test_dqn_agent_selects_valid_actions(smoke_cfg):
    """DQN agent should select valid action indices after training."""
    import numpy as np
    from market_making_agent.env.market_env import MarketMakingEnv
    agent, _ = train(smoke_cfg, device="cpu")
    env = MarketMakingEnv(smoke_cfg)
    obs, _ = env.reset()
    for _ in range(10):
        action = agent.select_action(obs, deterministic=True)
        assert 0 <= action < env.n_actions, f"Invalid action {action}"
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
