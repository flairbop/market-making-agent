"""
test_env.py
-----------
Tests for MarketMakingEnv: reset, step, inventory bounds, episode end.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.config import load_config, Config, _DEFAULTS
from market_making_agent.env.market_env import MarketMakingEnv


@pytest.fixture
def fast_cfg() -> Config:
    """Config with very short episodes for fast tests."""
    from market_making_agent.config import _DEFAULTS, Config, _deep_merge
    overrides = {
        "env": {
            "episode_length": 50,
            "max_inventory": 5,
        }
    }
    return Config(_deep_merge(_DEFAULTS, overrides))


def test_env_reset_returns_correct_shape(fast_cfg):
    env = MarketMakingEnv(fast_cfg)
    obs, info = env.reset()
    assert obs.shape == (8,), f"Expected shape (8,), got {obs.shape}"
    assert obs.dtype == np.float32


def test_env_state_dim_matches(fast_cfg):
    env = MarketMakingEnv(fast_cfg)
    assert env.state_dim == 8


def test_env_step_returns_correct_types(fast_cfg):
    env = MarketMakingEnv(fast_cfg)
    obs, _ = env.reset()
    action = 0  # Tightest spread
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_episode_terminates(fast_cfg):
    """Episode should end after episode_length steps."""
    env = MarketMakingEnv(fast_cfg)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
        assert steps <= fast_cfg.env.episode_length + 1, "Episode did not terminate"
    assert steps == fast_cfg.env.episode_length


def test_inventory_hard_limit(fast_cfg):
    """Inventory should never exceed ±max_inventory."""
    env = MarketMakingEnv(fast_cfg, rng=np.random.default_rng(0))
    env.reset()
    max_inv = fast_cfg.env.max_inventory
    for _ in range(fast_cfg.env.episode_length):
        action = 0  # Tightest offset = most fills
        _, _, terminated, truncated, info = env.step(action)
        assert abs(info["inventory"]) <= max_inv, (
            f"Inventory {info['inventory']} exceeded max {max_inv}"
        )
        if terminated or truncated:
            break


def test_env_action_space_size(fast_cfg):
    env = MarketMakingEnv(fast_cfg)
    assert env.n_actions == len(fast_cfg.env.quote_offsets)


def test_env_reset_clears_state(fast_cfg):
    """After reset, inventory and cash should be zero."""
    env = MarketMakingEnv(fast_cfg)
    env.reset()
    for _ in range(10):
        env.step(0)
    obs, _ = env.reset()
    # Inventory feature (index 0) should be 0 after reset
    assert obs[0] == pytest.approx(0.0, abs=1e-5)
