"""
performance_report.py
---------------------
Generates a human-readable performance report comparing the DQN agent
against baseline strategies.

Report includes:
  - Summary metrics for each agent (mean ± std PnL, Sharpe, fills, drawdown)
  - Relative performance vs baselines
  - Honest caveats about the simulation setup

Output is a plain-text file suitable for sharing with non-technical reviewers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


REPORT_TEMPLATE = """\
================================================================================
MARKET-MAKING AGENT — PERFORMANCE REPORT
Experiment: {experiment_name}
================================================================================

NOTE: All results are from a stylised simulation environment and do NOT
represent real trading performance. The simulator uses simplified fill
models and does not capture real market microstructure effects.

--------------------------------------------------------------------------------
DQN AGENT RESULTS ({n_eval_episodes} evaluation episodes)
--------------------------------------------------------------------------------
  Mean Episode PnL      : {dqn_pnl_mean:>10.4f}
  Std  Episode PnL      : {dqn_pnl_std:>10.4f}
  Sharpe Ratio          : {dqn_sharpe:>10.4f}  (episode PnL / std, not annualised)
  Mean Total Reward     : {dqn_reward_mean:>10.4f}
  Mean Fill Count       : {dqn_fills_mean:>10.2f}
  Mean Avg Spread Cap.  : {dqn_spread_mean:>10.4f}
  Mean Max |Inventory|  : {dqn_inv_mean:>10.2f}
  Mean Max Drawdown     : {dqn_dd_mean:>10.4f}

--------------------------------------------------------------------------------
BASELINE COMPARISON
--------------------------------------------------------------------------------
{baseline_table}
--------------------------------------------------------------------------------
RELATIVE PERFORMANCE (DQN vs Baselines)
--------------------------------------------------------------------------------
{relative_table}
--------------------------------------------------------------------------------
INTERPRETATION
--------------------------------------------------------------------------------
  - A positive mean PnL indicates the agent successfully captured spread
    on average across episodes.
  - Sharpe > 0 suggests consistent PnL relative to volatility.
  - High fill count = aggressive quoting (more risk, more reward potential).
  - Max inventory shows how much directional risk the agent took on.

CAVEATS:
  - The fill model is exponential and not calibrated to real data.
  - Adverse selection is a simplified fraction of realised vol.
  - No maker/taker rebate structure is modelled.
  - Results do not generalise to real exchange conditions.

================================================================================
"""


def _format_baseline_table(baseline_summaries: dict[str, dict]) -> str:
    """Format a table of baseline metrics."""
    header = f"  {'Agent':<20} {'PnL Mean':>12} {'PnL Std':>10} {'Sharpe':>8} {'Fills':>8}"
    sep = "  " + "-" * 62
    rows = [header, sep]
    for name, s in baseline_summaries.items():
        row = (
            f"  {name:<20}"
            f" {s.get('total_pnl_mean', 0):>12.4f}"
            f" {s.get('total_pnl_std', 0):>10.4f}"
            f" {s.get('sharpe_ratio', 0):>8.4f}"
            f" {s.get('fill_count_mean', 0):>8.1f}"
        )
        rows.append(row)
    return "\n".join(rows)


def _format_relative_table(dqn_pnl_mean: float, baseline_summaries: dict[str, dict]) -> str:
    """Format DQN improvement over each baseline."""
    header = f"  {'Baseline':<20} {'BL PnL Mean':>12} {'DQN − BL':>12} {'Improvement':>12}"
    sep = "  " + "-" * 60
    rows = [header, sep]
    for name, s in baseline_summaries.items():
        bl_mean = s.get("total_pnl_mean", 0.0)
        diff = dqn_pnl_mean - bl_mean
        pct = (diff / abs(bl_mean) * 100) if abs(bl_mean) > 1e-9 else float("nan")
        row = (
            f"  {name:<20}"
            f" {bl_mean:>12.4f}"
            f" {diff:>12.4f}"
            f" {pct:>11.1f}%"
        )
        rows.append(row)
    return "\n".join(rows)


def generate_report(
    experiment_name: str,
    dqn_summary: dict[str, float],
    baseline_summaries: dict[str, dict[str, float]],
    save_path: Optional[str | Path] = None,
    n_eval_episodes: int = 100,
) -> str:
    """
    Generate a plain-text performance report.

    Parameters
    ----------
    experiment_name:
        Name of the experiment run.
    dqn_summary:
        Metrics summary dict for the DQN agent.
    baseline_summaries:
        Dict of {baseline_name: metrics_summary}.
    save_path:
        If provided, save report to this path.
    n_eval_episodes:
        Number of eval episodes used (for display only).

    Returns
    -------
    str
        The formatted report text.
    """
    baseline_table = _format_baseline_table(baseline_summaries)
    relative_table = _format_relative_table(
        dqn_pnl_mean=dqn_summary.get("total_pnl_mean", 0.0),
        baseline_summaries=baseline_summaries,
    )

    report = REPORT_TEMPLATE.format(
        experiment_name=experiment_name,
        n_eval_episodes=n_eval_episodes,
        dqn_pnl_mean=dqn_summary.get("total_pnl_mean", 0.0),
        dqn_pnl_std=dqn_summary.get("total_pnl_std", 0.0),
        dqn_sharpe=dqn_summary.get("sharpe_ratio", 0.0),
        dqn_reward_mean=dqn_summary.get("total_reward_mean", 0.0),
        dqn_fills_mean=dqn_summary.get("fill_count_mean", 0.0),
        dqn_spread_mean=dqn_summary.get("avg_spread_captured_mean", 0.0),
        dqn_inv_mean=dqn_summary.get("max_abs_inventory_mean", 0.0),
        dqn_dd_mean=dqn_summary.get("max_drawdown_mean", 0.0),
        baseline_table=baseline_table,
        relative_table=relative_table,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report)

    return report
