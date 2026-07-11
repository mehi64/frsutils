# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the reproducible fuzzy-rough reference study."""

import csv
import json
from pathlib import Path

from studies.fuzzy_rough_reference_study.run_study import (
    load_study_config,
    load_study_dataset,
    run_reference_analysis,
)


CONFIG_PATH = Path("studies/fuzzy_rough_reference_study/study_config.json")


def test_canonical_study_config_covers_three_models_and_real_tasks():
    """Canonical configuration should cover all public model families."""
    config = load_study_config(CONFIG_PATH)

    assert [item["name"] for item in config["models"]] == ["itfrs", "vqrs", "owafrs"]
    assert len(config["datasets"]) == 3
    assert all(item["loader"] in {"breast_cancer", "wine", "digits"} for item in config["datasets"])


def test_dataset_preparation_is_binary_scaled_and_traceable():
    """Configured dataset conversion should preserve labels and normalize features."""
    config = load_study_config(CONFIG_PATH)
    dataset = load_study_dataset(config["datasets"][1])

    assert dataset.name == "wine_class_0_vs_rest"
    assert dataset.X.ndim == 2
    assert dataset.y.ndim == 1
    assert dataset.original_y.shape == dataset.y.shape
    assert set(dataset.y.tolist()) == {0, 1}
    assert dataset.X.min() >= 0.0
    assert dataset.X.max() <= 1.0


def test_reference_analysis_writes_equivalent_dense_and_blockwise_outputs(tmp_path):
    """A small canonical case should write passing research-artifact tables."""
    config = load_study_config(CONFIG_PATH)
    smoke_config = dict(config)
    smoke_config["datasets"] = [config["datasets"][1]]
    smoke_config["models"] = [config["models"][0]]
    smoke_config["repeats"] = 1
    smoke_config["block_size"] = 32

    tables = run_reference_analysis(smoke_config, output_dir=tmp_path)

    assert len(tables["equivalence"]) == 1
    assert tables["equivalence"][0]["passed"] is True
    assert tables["equivalence"][0]["max_abs_error_lower"] <= 1e-12
    assert tables["equivalence"][0]["max_abs_error_upper"] <= 1e-12

    expected_files = {
        "approximation_summary.csv",
        "dense_blockwise_equivalence.csv",
        "runtime_results.csv",
        "sample_scores.csv",
        "highest_boundary_gap_samples.csv",
    }
    assert expected_files.issubset({path.name for path in tmp_path.iterdir()})

    with (tmp_path / "dense_blockwise_equivalence.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert rows[0]["passed"] == "True"


def test_study_config_is_valid_json_and_has_fixed_tolerance():
    """The committed study configuration should be machine-readable and fixed."""
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw["schema_version"] == 1
    assert raw["equivalence_atol"] == 1e-12
    assert raw["random_seed"] == 42
