"""
plotting.py
-----------
Matplotlib-based plotting utilities. All functions save figures to disk
and optionally display them.

Design:
  - Pure matplotlib (no seaborn dependency).
  - All plots include clean styling, axis labels, and titles.
  - Functions return the figure so callers can further customise if needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style — clean, professional, suitable for reports
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    }
)


def _save(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")


def plot_pnl_curve(
    pnl_paths: list[list[float]],
    title: str = "Cumulative PnL",
    labels: Optional[list[str]] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot cumulative PnL over steps for one or more evaluation episodes.

    If multiple paths are supplied, plot each as a thin line plus a bold mean.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    if labels is None:
        # Group by agent if list-of-lists is given
        all_arrays = [np.cumsum(p) for p in pnl_paths]
        for i, arr in enumerate(all_arrays):
            ax.plot(arr, alpha=0.3, linewidth=0.8, color=colors[0])
        mean_arr = np.mean(all_arrays, axis=0)
        ax.plot(mean_arr, linewidth=2, color=colors[0], label="Mean cumulative PnL")
    else:
        for i, (path, label) in enumerate(zip(pnl_paths, labels)):
            ax.plot(np.cumsum(path), linewidth=2, color=colors[i % 10], label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative PnL")
    ax.set_title(title)
    ax.legend(framealpha=0.7)
    _save(fig, save_path)
    return fig


def plot_inventory_path(
    inventory_paths: list[list[float]],
    max_inventory: float = 10.0,
    title: str = "Inventory over Time",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot inventory (unit position) over time for multiple episodes.
    Highlights the max inventory bound.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for path in inventory_paths:
        ax.plot(path, alpha=0.3, linewidth=0.6, color="steelblue")

    mean_inv = np.mean(inventory_paths, axis=0)  # type: ignore[arg-type]
    ax.plot(mean_inv, linewidth=2, color="steelblue", label="Mean inventory")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(max_inventory, color="red", linewidth=1, linestyle="--", label=f"+{max_inventory:.0f} bound")
    ax.axhline(-max_inventory, color="red", linewidth=1, linestyle="--", label=f"−{max_inventory:.0f} bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Inventory (units)")
    ax.set_title(title)
    ax.legend(framealpha=0.7)
    _save(fig, save_path)
    return fig


def plot_reward_curve(
    episode_rewards: list[float],
    window: int = 50,
    title: str = "Training Reward",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot per-episode training rewards with a rolling mean.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    rewards = np.array(episode_rewards, dtype=float)
    ax.plot(rewards, alpha=0.3, linewidth=0.8, color="darkorange", label="Episode reward")

    if len(rewards) >= window:
        rolling = pd.Series(rewards).rolling(window=window).mean().values
        ax.plot(rolling, linewidth=2, color="darkorange", label=f"{window}-ep rolling mean")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend(framealpha=0.7)
    _save(fig, save_path)
    return fig


def plot_baseline_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "total_pnl_mean",
    title: str = "Baseline Comparison — Mean PnL per Episode",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing agents on a chosen metric (mean ± std).

    Parameters
    ----------
    results:
        Dict of {agent_name: metrics_summary_dict}.
    metric:
        The key from each metrics summary to plot (mean).
    """
    names = list(results.keys())
    means = [results[n].get(metric, 0.0) for n in names]
    stds = [results[n].get(metric.replace("_mean", "_std"), 0.0) for n in names]

    colors = ["#4CAF50" if "dqn" in n.lower() else "#2196F3" for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + abs(bar.get_height()) * 0.02,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=20, ha="right")
    _save(fig, save_path)
    return fig


def plot_return_distribution(
    pnl_values: Sequence[float],
    label: str = "DQN",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Histogram of episode PnL values with mean and std annotations.
    """
    arr = np.array(pnl_values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={arr.mean():.2f}")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Episode PnL")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Episode Returns — {label}")
    ax.legend(framealpha=0.7)
    _save(fig, save_path)
    return fig
