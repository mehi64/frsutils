"""
@file phase7_benchmark_smoke.py
@brief Release-ready smoke example for the Phase 6 benchmark suite.

This example runs a tiny CPU-only benchmark through the public benchmark module.
It is intentionally small enough for documentation and CI smoke tests. Real
paper numbers should be generated later on a stable benchmark machine.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# main                                 Run tiny dense/blockwise benchmark rows
# write_json_report                    Persist a machine-readable JSON report
# write_csv_report                     Persist a flat CSV report

# ✅ Design Patterns & Clean Code Notes
# - Facade Pattern: benchmark module itself depends only on FRsutils.api
# - Reproducibility: fixed random seed and tiny sample size
# - Artifact Readiness: writes JSON and CSV files for inspection
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# python examples/phase7_benchmark_smoke.py --output-dir .phase7_benchmark_smoke
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
    """
    @brief Build the command-line parser for the smoke example.

    @return: Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Run a tiny FRsutils benchmark smoke example.")
    parser.add_argument(
        "--output-dir",
        default="phase7_benchmark_smoke_output",
        help="Directory where benchmark JSON/CSV outputs will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    @brief Run the tiny benchmark and write JSON/CSV outputs.

    @param argv: Optional command-line arguments for tests.
    @return: Process exit code.
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

    json_path = output_dir / "phase7_benchmark_smoke.json"
    csv_path = output_dir / "phase7_benchmark_smoke.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)

    successful_rows = [row for row in report["results"] if row["status"] == "success"]
    print("FRsutils Phase 7 benchmark smoke complete")
    print(f"rows={len(report['results'])}")
    print(f"successful_rows={len(successful_rows)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
