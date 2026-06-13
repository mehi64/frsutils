# SPDX-License-Identifier: BSD-3-Clause
"""Phase 6 contract tests for the benchmark suite."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.benchmark_fuzzy_rough_execution import run_benchmark_suite, write_csv_report, write_json_report


def test_phase6_benchmark_suite_writes_machine_readable_artifacts(tmp_path):
    """@brief Tiny CPU-only benchmark matrix should produce JSON/CSV rows."""
    report = run_benchmark_suite(
        models=["itfrs"],
        sample_sizes=[12],
        n_features=3,
        block_sizes=[4],
        scenario_names=["dense_numpy", "blockwise_numpy"],
        repeats=1,
        random_state=7,
    )

    assert report["metadata"]["benchmark_phase"] == "phase_6"
    assert len(report["results"]) == 2
    assert all(row["status"] == "success" for row in report["results"])

    blockwise_rows = [row for row in report["results"] if row["scenario"] == "blockwise_numpy"]
    assert len(blockwise_rows) == 1
    assert blockwise_rows[0]["used_blockwise"] is True
    assert blockwise_rows[0]["max_abs_error_lower"] <= 1e-12
    assert blockwise_rows[0]["max_abs_error_upper"] <= 1e-12
    assert blockwise_rows[0]["max_abs_error_positive_region"] <= 1e-12

    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"
    write_json_report(report, json_path)
    write_csv_report(report, csv_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["benchmark_phase"] == "phase_6"

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 2


def test_phase6_benchmark_cli_smoke(tmp_path):
    """@brief CLI should run a tiny benchmark and write JSON/CSV outputs."""
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "benchmarks" / "benchmark_fuzzy_rough_execution.py"
    json_path = tmp_path / "cli_benchmark.json"
    csv_path = tmp_path / "cli_benchmark.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--models",
            "itfrs",
            "--sample-sizes",
            "10",
            "--n-features",
            "2",
            "--block-sizes",
            "5",
            "--scenarios",
            "dense_numpy,blockwise_numpy",
            "--repeats",
            "1",
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Phase 6 benchmark complete" in completed.stdout
    assert json_path.exists()
    assert csv_path.exists()
