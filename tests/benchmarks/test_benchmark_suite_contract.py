# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the benchmark suite."""

import csv
import json

import numpy as np

from benchmarks.benchmark_fuzzy_rough_execution import (
    load_npy_dataset,
    main,
    run_benchmark_suite,
    write_csv_report,
    write_json_report,
)


def test_benchmark_suite_writes_machine_readable_artifacts(tmp_path):
    """Tiny CPU-only benchmark matrix should produce JSON/CSV rows."""
    report = run_benchmark_suite(
        models=["itfrs"],
        sample_sizes=[12],
        n_features=3,
        block_sizes=[4],
        scenario_names=["dense_numpy", "blockwise_numpy"],
        repeats=1,
        random_state=7,
    )

    assert report["metadata"]["benchmark_suite"] == "public_api_execution"
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
    assert loaded["metadata"]["benchmark_suite"] == "public_api_execution"

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 2


def test_benchmark_cli_smoke(tmp_path):
    """CLI should run a tiny benchmark and write JSON/CSV outputs."""
    json_path = tmp_path / "cli_benchmark.json"
    csv_path = tmp_path / "cli_benchmark.csv"

    exit_code = main(
        [
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
    )

    assert exit_code == 0
    assert json_path.exists()
    assert csv_path.exists()


def test_benchmark_suite_can_skip_dense_reference_for_blockwise_runs():
    """Large-run mode should avoid dense reference computation."""
    report = run_benchmark_suite(
        models=["itfrs"],
        sample_sizes=[12],
        n_features=3,
        block_sizes=[4],
        scenario_names=["blockwise_numpy"],
        repeats=1,
        random_state=7,
        skip_dense_reference=True,
    )

    assert report["metadata"]["dense_reference_enabled"] is False
    assert len(report["results"]) == 1
    row = report["results"][0]
    assert row["status"] == "success"
    assert row["used_blockwise"] is True
    assert row["max_abs_error_lower"] is None
    assert row["max_abs_error_upper"] is None
    assert row["max_abs_error_positive_region"] is None


def test_benchmark_suite_accepts_npy_dataset_with_full_default_size(tmp_path):
    """Loaded datasets should be benchmarkable without synthetic generation."""
    x_path = tmp_path / "X.npy"
    y_path = tmp_path / "y.npy"
    np.save(x_path, np.arange(24, dtype=float).reshape(8, 3) / 24.0)
    np.save(y_path, np.asarray([0, 1, 0, 1, 0, 1, 0, 1]))

    dataset = load_npy_dataset(x_path=x_path, y_path=y_path, name="tiny_npy")
    report = run_benchmark_suite(
        models=["itfrs"],
        sample_sizes=[8],
        n_features=99,
        block_sizes=[4],
        scenario_names=["blockwise_numpy"],
        repeats=1,
        random_state=7,
        dataset=dataset,
        skip_dense_reference=True,
    )

    assert report["metadata"]["dataset_name"] == "tiny_npy"
    assert report["metadata"]["dataset_source"].startswith("npy:")
    assert report["metadata"]["n_features"] == 3
    assert report["results"][0]["n_samples"] == 8


def test_benchmark_cli_can_load_csv_and_skip_dense_reference(tmp_path):
    """CLI should support large-dataset CSV mode without dense reference."""
    csv_input = tmp_path / "dataset.csv"
    json_path = tmp_path / "csv_benchmark.json"

    csv_input.write_text(
        "f1,f2,target\n"
        "0.0,0.1,a\n"
        "0.2,0.3,b\n"
        "0.4,0.5,a\n"
        "0.6,0.7,b\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--models",
            "itfrs",
            "--input-csv",
            str(csv_input),
            "--target-column",
            "target",
            "--block-sizes",
            "2",
            "--scenarios",
            "blockwise_numpy",
            "--repeats",
            "1",
            "--skip-dense-reference",
            "--output-json",
            str(json_path),
        ],
    )

    assert exit_code == 0
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["dataset_name"] == "dataset"
    assert loaded["metadata"]["sample_sizes"] == [4]
    assert loaded["metadata"]["dense_reference_enabled"] is False
    assert loaded["results"][0]["max_abs_error_lower"] is None
