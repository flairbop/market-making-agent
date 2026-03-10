"""
test_reward.py
--------------
Tests for the reward function in env/reward.py.
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.env.reward import compute_step_reward, StepRewardComponents


def _basic_reward(**kwargs) -> StepRewardComponents:
    defaults = dict(
        cash_change=0.0,
        mtm_change=0.0,
        inventory=0,
        inventory_penalty=0.01,
        bid_filled=False,
        ask_filled=False,
        fill_price_bid=100.0,
        fill_price_ask=100.02,
        transaction_cost_rate=0.001,
    )
    defaults.update(kwargs)
    return compute_step_reward(**defaults)


def test_zero_reward_with_no_fills_and_zero_inventory():
    """No fills, no inventory → all components are zero."""
    r = _basic_reward()
    assert r.realized_pnl == pytest.approx(0.0)
    assert r.inventory_penalty == pytest.approx(0.0)
    assert r.transaction_cost == pytest.approx(0.0)
    assert r.step_reward == pytest.approx(0.0)


def test_inventory_penalty_increases_with_inventory():
    """Quadratic penalty: larger |inventory| → larger penalty."""
    r_small = _basic_reward(inventory=2)
    r_large = _basic_reward(inventory=5)
    assert r_large.inventory_penalty > r_small.inventory_penalty


def test_transaction_cost_applied_on_fill():
    """Transaction costs are only applied when fills occur."""
    r_no_fill = _basic_reward(bid_filled=False, ask_filled=False)
    r_fill = _basic_reward(bid_filled=True, ask_filled=False, fill_price_bid=100.0)
    assert r_no_fill.transaction_cost == pytest.approx(0.0)
    assert r_fill.transaction_cost > 0.0


def test_positive_mtm_increases_reward():
    """Mark-to-market gain should positively affect realized_pnl."""
    r = _basic_reward(mtm_change=0.5)
    assert r.realized_pnl == pytest.approx(0.5)


def test_step_reward_decreases_with_penalty():
    """Higher inventory penalty coefficient → lower step reward."""
    r_low = _basic_reward(inventory=3, inventory_penalty=0.01)
    r_high = _basic_reward(inventory=3, inventory_penalty=0.10)
    assert r_high.step_reward < r_low.step_reward


def test_round_trip_positive_reward():
    """A round-trip fill (both bid and ask) should generally be positive."""
    # Bid fill: buy at 99.99, sell at 100.01 → cash gain of 0.02 per unit
    r = _basic_reward(
        cash_change=+0.02,  # net cash from round trip
        mtm_change=0.0,
        inventory=0,  # back to neutral
        bid_filled=True,
        ask_filled=True,
        fill_price_bid=99.99,
        fill_price_ask=100.01,
        transaction_cost_rate=0.00001,  # very low fee (~0.002 total), well below 0.02 spread
        inventory_penalty=0.01,
    )
    # With zero inventory penalty (inv=0) and small fees, reward should be positive
    assert r.step_reward > 0, f"Expected positive round-trip reward, got {r.step_reward}"
