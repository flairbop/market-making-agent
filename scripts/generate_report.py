"""
generate_report.py
------------------
Load saved evaluation results from JSON and generate a performance report.

Useful when training and evaluation have already been run and you want to
regenerate or customise the report without re-running evaluation.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --results-json outputs/logs/dqn_market_maker_default_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from market_making_agent.analysis.performance_report import generate_report

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-json",
        default="outputs/logs/dqn_market_maker_default_results.json",
        help="Path to the results JSON saved by run_experiment.py",
    )
    parser.add_argument("--output", default="outputs/reports/generated_report.txt")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        logger.error("Run train_default.py first to generate results.")
        sys.exit(1)

    with open(results_path) as f:
        all_results: dict = json.load(f)

    dqn_summary = all_results.pop("DQN", {})
    baseline_summaries = all_results

    report = generate_report(
        experiment_name=results_path.stem,
        dqn_summary=dqn_summary,
        baseline_summaries=baseline_summaries,
        save_path=Path(args.output),
    )
    print(report)
    logger.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
