"""
metrics.py
----------
Financial performance metrics computed over a series of episode results.

All metrics are computed from raw episode-level data (PnL, inventory
trajectory, fill counts, etc.) without any obscuring abstractions.

Assumptions / limitations:
  - "Sharpe ratio" here is computed per-episode PnL mean / std, treating
    each episode as a return observation. This is a stylised metric only;
    it does not annualise or account for the exact time horizon.
  - Calmar ratio uses drawdown from the cumulative episode PnL series.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_sharpe_ratio(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """
    Compute a Sharpe-like ratio from a sequence of episode PnL values.

    Parameters
    ----------
    returns:
        Per-episode PnL or return values.
    risk_free:
        Assumed risk-free baseline (default 0).

    Returns
    -------
    float
        Sharpe ratio, or 0.0 if std is zero.
    """
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free
    std = excess.std()
    return float(excess.mean() / std) if std > 1e-9 else 0.0


def compute_max_drawdown(cumulative_pnl: Sequence[float]) -> float:
    """
    Compute the maximum drawdown of a cumulative PnL series.

    Maximum drawdown = max(peak - trough) / |peak|.

    Parameters
    ----------
    cumulative_pnl:
        Sequence of cumulative PnL values over time.

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.15 = 15%).
    """
    arr = np.array(cumulative_pnl, dtype=float)
    if len(arr) == 0:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = peak - arr
    max_dd = drawdown.max()
    peak_val = np.abs(peak[np.argmax(drawdown)]) if np.abs(peak[np.argmax(drawdown)]) > 1e-9 else 1.0
    return float(max_dd / peak_val)


def compute_episode_metrics(episodes: list[dict]) -> pd.DataFrame:
    """
    Aggregate per-episode statistics from a list of episode result dicts.

    Each episode dict should contain keys:
        - ``total_pnl``: scalar total PnL for the episode
        - ``total_reward``: scalar total reward
        - ``fill_count``: int total fills (bids + asks)
        - ``avg_spread_captured``: mean spread per filled pair
        - ``inventory_path``: list of inventory values over time
        - ``pnl_path``: list of step-level PnL increments

    Returns
    -------
    pd.DataFrame
        One row per episode with computed metrics.
    """
    rows = []
    for ep in episodes:
        inv_path = np.array(ep.get("inventory_path", [0.0]))
        pnl_path = np.cumsum(ep.get("pnl_path", [0.0]))

        max_abs_inv = float(np.abs(inv_path).max()) if len(inv_path) else 0.0
        max_dd = compute_max_drawdown(pnl_path)

        rows.append(
            {
                "total_pnl": ep.get("total_pnl", 0.0),
                "total_reward": ep.get("total_reward", 0.0),
                "fill_count": ep.get("fill_count", 0),
                "avg_spread_captured": ep.get("avg_spread_captured", 0.0),
                "max_abs_inventory": max_abs_inv,
                "max_drawdown": max_dd,
            }
        )
    return pd.DataFrame(rows)


def summarise_metrics(df: pd.DataFrame) -> dict[str, float]:
    """
    Summarise a metrics DataFrame into mean ± std for each column.

    Returns a flat dict like ``{"total_pnl_mean": ..., "total_pnl_std": ...}``.
    """
    summary: dict[str, float] = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std())
    # Add Sharpe computed over episode PnLs
    summary["sharpe_ratio"] = compute_sharpe_ratio(df["total_pnl"].tolist())
    return summary
