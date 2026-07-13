# SPDX-License-Identifier: BSD-3-Clause
"""Integrity and schema contract tests for JSON scientific reference data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests import reference_data_loader as loader


REFERENCE_DATA_DIR = Path(loader.__file__).with_name("reference_data")
MANIFEST = loader._load_manifest()
MANIFEST_FILES = MANIFEST["files"]


def _iter_arrays(value: Any):
    """Yield every NumPy array contained in a decoded reference payload."""
    if isinstance(value, np.ndarray):
        yield value
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_arrays(item)


def _assert_unique_case_names(cases: list[dict[str, Any]], *, context: str) -> None:
    """Assert that a case collection has non-empty unique names."""
    names = [case.get("name") for case in cases]
    assert all(isinstance(name, str) and name for name in names), context
    assert len(names) == len(set(names)), context


def _assert_component_spec(spec: dict[str, Any]) -> None:
    """Validate a registered scientific-component specification."""
    assert set(spec) == {"name", "params"}
    assert isinstance(spec["name"], str) and spec["name"]
    assert isinstance(spec["params"], dict)


def _assert_plain_label_spec(spec: dict[str, Any], *, n_samples: int) -> None:
    """Validate labels stored as plain JSON values with explicit metadata."""
    assert set(spec) == {"values", "dtype", "shape"}
    assert spec["dtype"] == "object"
    assert spec["shape"] == [n_samples]
    assert isinstance(spec["values"], list)
    assert len(spec["values"]) == n_samples


def _assert_label_vector(labels: Any, *, n_samples: int) -> None:
    """Validate either an encoded numeric label vector or plain JSON labels."""
    if isinstance(labels, np.ndarray):
        assert labels.shape == (n_samples,)
        return
    _assert_plain_label_spec(labels, n_samples=n_samples)


def _assert_similarity_matrix(matrix: np.ndarray) -> None:
    """Validate the common contract of a dense reference similarity matrix."""
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(np.isfinite(matrix))
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.diag(matrix),
        np.ones(matrix.shape[0], dtype=float),
        rtol=0.0,
        atol=1e-15,
    )


def _assert_expected_vectors(
    expected: dict[str, np.ndarray],
    *,
    n_samples: int,
    include_interim: bool = False,
) -> None:
    """Validate model approximation vectors and their shapes."""
    required = {"lower", "upper", "boundary", "positive_region"}
    if include_interim:
        required.add("interim")
    assert set(expected) == required
    assert all(value.shape == (n_samples,) for value in expected.values())
    assert all(np.all(np.isfinite(value)) for value in expected.values())
    np.testing.assert_allclose(
        expected["boundary"],
        expected["upper"] - expected["lower"],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        expected["positive_region"],
        expected["lower"],
        rtol=0.0,
        atol=1e-15,
    )


def _validate_approximation_baselines(decoded: Any) -> None:
    """Validate public dense approximation baseline cases."""
    _assert_unique_case_names(decoded, context="dense approximation baseline cases")
    for case in decoded:
        assert set(case) == {
            "name",
            "X",
            "labels",
            "similarity",
            "expected_by_model",
        }
        n_samples = case["X"].shape[0]
        assert case["X"].ndim == 2
        assert case["labels"].shape == (n_samples,)
        _assert_component_spec(case["similarity"])
        assert set(case["expected_by_model"]) == {"itfrs", "vqrs", "owafrs"}
        for expected in case["expected_by_model"].values():
            _assert_expected_vectors(expected, n_samples=n_samples)


def _validate_itfrs_reference_data(decoded: Any) -> None:
    """Validate grouped legacy and dense ITFRS reference cases."""
    assert set(decoded) == {"legacy_testsets", "dense_baselines"}
    _assert_unique_case_names(decoded["legacy_testsets"], context="legacy ITFRS cases")
    _assert_unique_case_names(decoded["dense_baselines"], context="dense ITFRS cases")
    for case in decoded["dense_baselines"]:
        assert set(case) == {
            "name",
            "similarity_matrix",
            "labels",
            "components",
            "expected",
        }
        n_samples = case["similarity_matrix"].shape[0]
        _assert_similarity_matrix(case["similarity_matrix"])
        _assert_label_vector(case["labels"], n_samples=n_samples)
        assert set(case["components"]) == {"upper_tnorm", "lower_implicator"}
        _assert_component_spec(case["components"]["upper_tnorm"])
        _assert_component_spec(case["components"]["lower_implicator"])
        _assert_expected_vectors(case["expected"], n_samples=n_samples)


def _validate_vqrs_reference_data(decoded: Any) -> None:
    """Validate grouped legacy and dense VQRS reference cases."""
    assert set(decoded) == {"legacy_testsets", "dense_fixtures"}
    _assert_unique_case_names(decoded["legacy_testsets"], context="legacy VQRS cases")
    _assert_unique_case_names(decoded["dense_fixtures"], context="dense VQRS fixtures")
    baseline_names: list[str] = []
    for fixture in decoded["dense_fixtures"]:
        assert set(fixture) == {
            "name",
            "similarity_matrix",
            "labels",
            "baselines",
        }
        n_samples = fixture["similarity_matrix"].shape[0]
        _assert_similarity_matrix(fixture["similarity_matrix"])
        _assert_plain_label_spec(fixture["labels"], n_samples=n_samples)
        _assert_unique_case_names(
            fixture["baselines"],
            context=f"VQRS baselines in {fixture['name']}",
        )
        for baseline in fixture["baselines"]:
            baseline_names.append(baseline["name"])
            assert set(baseline) == {"name", "components", "expected"}
            assert set(baseline["components"]) == {
                "lower_quantifier",
                "upper_quantifier",
            }
            _assert_component_spec(baseline["components"]["lower_quantifier"])
            _assert_component_spec(baseline["components"]["upper_quantifier"])
            _assert_expected_vectors(
                baseline["expected"],
                n_samples=n_samples,
                include_interim=True,
            )
    assert len(baseline_names) == len(set(baseline_names))


def _validate_owafrs_reference_data(decoded: Any) -> None:
    """Validate grouped legacy and dense OWAFRS reference cases."""
    assert set(decoded) == {"legacy_testsets", "dense_baselines"}
    _assert_unique_case_names(decoded["legacy_testsets"], context="legacy OWAFRS cases")
    _assert_unique_case_names(decoded["dense_baselines"], context="dense OWAFRS cases")
    for case in decoded["dense_baselines"]:
        assert set(case) == {
            "name",
            "similarity_matrix",
            "labels",
            "components",
            "expected",
        }
        n_samples = case["similarity_matrix"].shape[0]
        _assert_similarity_matrix(case["similarity_matrix"])
        _assert_plain_label_spec(case["labels"], n_samples=n_samples)
        assert set(case["components"]) == {
            "upper_tnorm",
            "lower_implicator",
            "upper_owa",
            "lower_owa",
        }
        _assert_component_spec(case["components"]["upper_tnorm"])
        _assert_component_spec(case["components"]["lower_implicator"])
        for key, order in (("upper_owa", "descending"), ("lower_owa", "ascending")):
            spec = case["components"][key]
            assert set(spec) == {"name", "params", "order"}
            assert spec["order"] == order
            _assert_component_spec({"name": spec["name"], "params": spec["params"]})
        _assert_expected_vectors(case["expected"], n_samples=n_samples)


def _validate_changed_dataset_schema(dataset_kind: str, decoded: Any) -> None:
    """Validate schemas introduced or extended by the inline-data migration."""
    if dataset_kind == "implicator_scalar":
        assert set(decoded) == {
            "scalar_call_testsets",
            "boundary_cases",
            "branch_edge_cases",
        }
        _assert_unique_case_names(
            decoded["scalar_call_testsets"],
            context="implicator scalar-call cases",
        )
        for case in decoded["boundary_cases"]:
            assert set(case) == {"implicator", "a", "b", "expected"}
            assert isinstance(case["implicator"], str)
        for case in decoded["branch_edge_cases"]:
            assert set(case) == {"implicator", "a", "b", "expected"}
            assert case["a"].shape == case["b"].shape == case["expected"].shape
        return

    if dataset_kind == "similarities":
        assert set(decoded) == {"matrix_testsets", "dense_baselines"}
        _assert_unique_case_names(
            decoded["matrix_testsets"],
            context="legacy similarity matrix cases",
        )
        _assert_unique_case_names(
            decoded["dense_baselines"],
            context="dense similarity baseline cases",
        )
        for case in decoded["dense_baselines"]:
            assert set(case) == {"name", "X", "similarity", "expected"}
            assert case["X"].ndim == 2
            assert case["expected"].shape == (
                case["X"].shape[0],
                case["X"].shape[0],
            )
            _assert_component_spec(case["similarity"])
        return

    if dataset_kind == "fuzzy_quantifiers":
        _assert_unique_case_names(decoded, context="fuzzy-quantifier cases")
        for case in decoded:
            assert set(case) == {"name", "quantifier", "x", "expected"}
            assert case["x"].shape == case["expected"].shape
            _assert_component_spec(case["quantifier"])
            params = case["quantifier"]["params"]
            assert set(params) == {"alpha", "beta"}
            assert 0.0 <= params["alpha"] < params["beta"] <= 1.0
        return

    if dataset_kind == "approximation_baselines":
        _validate_approximation_baselines(decoded)
        return

    if dataset_kind == "itfrs":
        _validate_itfrs_reference_data(decoded)
        return

    if dataset_kind == "vqrs":
        _validate_vqrs_reference_data(decoded)
        return

    if dataset_kind == "owafrs":
        _validate_owafrs_reference_data(decoded)


def test_manifest_lists_every_reference_data_json_file():
    """Ensure the manifest has no missing or orphaned reference-data files."""
    json_files = {
        path.name
        for path in REFERENCE_DATA_DIR.glob("*.json")
        if path.name != "manifest.json"
    }

    assert set(MANIFEST_FILES) == json_files


@pytest.mark.parametrize("filename", sorted(MANIFEST_FILES))
def test_reference_data_file_matches_manifest_and_loads_read_only_arrays(filename):
    """Verify each file's integrity, schema, payload, and array immutability."""
    manifest_entry = MANIFEST_FILES[filename]
    path = REFERENCE_DATA_DIR / filename
    raw = path.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == manifest_entry["sha256"]

    decoded = loader.load_reference_data(
        filename,
        expected_dataset_kind=manifest_entry["dataset_kind"],
    )
    arrays = list(_iter_arrays(decoded))

    assert decoded is not None
    assert arrays, f"No NumPy arrays were decoded from {filename!r}."
    assert all(not array.flags.writeable for array in arrays)
    _validate_changed_dataset_schema(manifest_entry["dataset_kind"], decoded)


def test_json_parser_rejects_duplicate_keys():
    """Ensure ambiguous JSON objects with duplicate keys are rejected."""
    raw = b'{"schema_version": 1, "schema_version": 2}'

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        loader._parse_json_bytes(raw, source=Path("duplicate.json"))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_parser_rejects_non_standard_numbers(constant):
    """Ensure non-standard JSON numeric constants are rejected."""
    raw = f'{{"value": {constant}}}'.encode("utf-8")

    with pytest.raises(ValueError, match="Non-standard JSON numeric value"):
        loader._parse_json_bytes(raw, source=Path("non_standard_number.json"))


def test_array_decoder_rejects_unsupported_dtype():
    """Ensure reference arrays cannot silently change to an unsupported dtype."""
    encoded = {
        "__ndarray__": [1.0, 2.0],
        "dtype": "float32",
        "shape": [2],
    }

    with pytest.raises(ValueError, match="Unsupported reference-array dtype"):
        loader._decode_reference_value(encoded)


def test_array_decoder_rejects_declared_shape_mismatch():
    """Ensure declared array shapes must match their encoded values exactly."""
    encoded = {
        "__ndarray__": [1.0, 2.0],
        "dtype": "float64",
        "shape": [1, 2],
    }

    with pytest.raises(ValueError, match="Reference-array shape mismatch"):
        loader._decode_reference_value(encoded)
