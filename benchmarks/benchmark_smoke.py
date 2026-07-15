# SPDX-License-Identifier: BSD-3-Clause
"""Smoke example for running a tiny frsutils benchmark suite.

This script writes JSON and CSV benchmark artifacts and is not part of the
stable Python API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

# Allow direct execution from a source checkout without editable installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.benchmark_fuzzy_rough_execution import (  # noqa: E402
    run_benchmark_suite,
    write_csv_report,
    write_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the smoke example.

    Returns
    -------
    parser : argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run a tiny frsutils benchmark smoke example.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_smoke_output",
        help="Directory where benchmark JSON/CSV outputs will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tiny benchmark and write JSON/CSV outputs.

    Parameters
    ----------
    argv : Sequence[str] or None, default=None
        Optional command-line arguments for tests.

    Returns
    -------
    exit_code : int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = run_benchmark_suite(
        models=["itfrs", "vqrs"],
        sample_sizes=[16],
        n_features=3,
        block_sizes=[8],
        scenario_names=["dense_numpy", "blockwise_numpy"],
        repeats=1,
        random_state=11,
    )

    json_path = output_dir / "benchmark_smoke.json"
    csv_path = output_dir / "benchmark_smoke.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)

    successful_rows = [row for row in report["results"] if row["status"] == "success"]
    print("frsutils benchmark smoke complete")
    print(f"rows={len(report['results'])}")
    print(f"successful_rows={len(successful_rows)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
