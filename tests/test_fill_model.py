"""
test_fill_model.py
------------------
Tests for ExponentialFillModel: fill probabilities, boundary conditions.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.env.fill_model import ExponentialFillModel


def test_fill_prob_at_zero_offset_equals_base():
    model = ExponentialFillModel(base_fill_rate=0.5, decay_rate=10.0)
    p = model.fill_probability(offset=0.0)
    assert p == pytest.approx(0.5, abs=1e-6)


def test_fill_prob_decreases_with_offset():
    model = ExponentialFillModel(base_fill_rate=0.5, decay_rate=10.0)
    p0 = model.fill_probability(0.0)
    p1 = model.fill_probability(0.1)
    p2 = model.fill_probability(0.5)
    assert p0 > p1 > p2


def test_fill_prob_stays_in_unit_interval():
    model = ExponentialFillModel(base_fill_rate=0.8, decay_rate=5.0)
    for offset in [0.0, 0.01, 0.1, 1.0, 10.0]:
        p = model.fill_probability(offset)
        assert 0.0 <= p <= 1.0, f"fill_prob={p} out of [0,1] for offset={offset}"


def test_sample_fill_returns_bool():
    model = ExponentialFillModel()
    result = model.sample_fill(offset=0.02)
    assert isinstance(result, bool)


def test_sample_fills_returns_two_bools():
    model = ExponentialFillModel()
    bid_filled, ask_filled = model.sample_fills(0.01, 0.03)
    assert isinstance(bid_filled, bool)
    assert isinstance(ask_filled, bool)


def test_fill_frequency_decreasing_empirically():
    """Statistical: tight quotes should fill more often than wide ones."""
    rng = np.random.default_rng(42)
    model = ExponentialFillModel(base_fill_rate=0.6, decay_rate=10.0, rng=rng)
    n_trials = 5000

    tight_fills = sum(model.sample_fill(0.01) for _ in range(n_trials))
    wide_fills = sum(model.sample_fill(0.20) for _ in range(n_trials))
    assert tight_fills > wide_fills, (
        f"Expected tight fills ({tight_fills}) > wide fills ({wide_fills})"
    )


def test_invalid_base_fill_rate():
    with pytest.raises(ValueError):
        ExponentialFillModel(base_fill_rate=0.0)
    with pytest.raises(ValueError):
        ExponentialFillModel(base_fill_rate=1.5)
