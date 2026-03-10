"""
test_baselines.py
-----------------
Tests for baseline agents: action validity, interface compliance.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.agents.baselines import (
    FixedSpreadAgent,
    RandomAgent,
    InventorySkewAgent,
    get_all_baselines,
)
from market_making_agent.env.state_builder import STATE_DIM


# A dummy state vector (inventory=0, so state[0]=0)
DUMMY_STATE = np.zeros(STATE_DIM, dtype=np.float32)
N_ACTIONS = 5


def test_fixed_spread_always_returns_same_action():
    agent = FixedSpreadAgent(action_index=2)
    actions = {agent.select_action(DUMMY_STATE, N_ACTIONS) for _ in range(10)}
    assert actions == {2}


def test_fixed_spread_clips_to_valid_range():
    agent = FixedSpreadAgent(action_index=100)
    action = agent.select_action(DUMMY_STATE, N_ACTIONS)
    assert 0 <= action < N_ACTIONS


def test_random_agent_returns_valid_actions():
    agent = RandomAgent(rng=np.random.default_rng(0))
    for _ in range(100):
        action = agent.select_action(DUMMY_STATE, N_ACTIONS)
        assert 0 <= action < N_ACTIONS


def test_inventory_skew_positive_inventory_wider():
    """With positive inventory (state[0] > 0), agent should choose a wider quote."""
    agent = InventorySkewAgent(skew_sensitivity=2.0, base_action=2)
    # Positive inventory → state[0] positive
    state_flat = DUMMY_STATE.copy()
    state_long = DUMMY_STATE.copy()
    state_long[0] = 0.8  # 80% of max inventory
    action_flat = agent.select_action(state_flat, N_ACTIONS)
    action_long = agent.select_action(state_long, N_ACTIONS)
    # Long inventory → higher action (wider spread)
    assert action_long >= action_flat


def test_inventory_skew_negative_inventory_tighter():
    """With negative inventory, agent should choose a tighter bid."""
    agent = InventorySkewAgent(skew_sensitivity=2.0, base_action=2)
    state_short = DUMMY_STATE.copy()
    state_short[0] = -0.8
    state_flat = DUMMY_STATE.copy()
    action_short = agent.select_action(state_short, N_ACTIONS)
    action_flat = agent.select_action(state_flat, N_ACTIONS)
    assert action_short <= action_flat


def test_get_all_baselines_returns_three():
    agents = get_all_baselines()
    assert len(agents) == 3
    names = {a.name for a in agents}
    assert "FixedSpread" in names
    assert "Random" in names
    assert "InventorySkew" in names


def test_all_baselines_have_reset():
    """All baseline agents must implement reset()."""
    for agent in get_all_baselines():
        assert hasattr(agent, "reset"), f"{agent.name} missing reset()"
        agent.reset()  # Should not raise
