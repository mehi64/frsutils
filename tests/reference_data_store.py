# SPDX-License-Identifier: BSD-3-Clause
"""Compatibility accessors for JSON-backed scientific reference test data.

This module preserves the existing test-data API while delegating loading,
integrity checks, schema validation, and NumPy reconstruction to the shared
reference-data loader.
"""

from __future__ import annotations

from typing import Any

from tests.reference_data_loader import load_reference_data


def get_tnorm_call_testsets() -> list[dict[str, Any]]:
    """Return reference cases for direct T-norm calls."""
    return load_reference_data(
        "tnorm_call.json",
        expected_dataset_kind="tnorm_call",
    )


def get_tnorm_reduce_testsets() -> list[dict[str, Any]]:
    """Return legacy reference cases for T-norm matrix combinations."""
    return load_reference_data(
        "tnorm_reduce.json",
        expected_dataset_kind="tnorm_reduce",
    )


def get_tnorm_regression_testsets() -> dict[str, Any]:
    """Return reference values used by focused T-norm regression tests."""
    return load_reference_data(
        "tnorm_regression.json",
        expected_dataset_kind="tnorm_regression",
    )


def _load_implicator_reference_data() -> dict[str, Any]:
    """Return the grouped implicator reference-data payload."""
    return load_reference_data(
        "implicator_scalar.json",
        expected_dataset_kind="implicator_scalar",
    )


def get_implicator_scalar_testsets() -> list[dict[str, Any]]:
    """Return reference cases for scalar implicator computations."""
    return _load_implicator_reference_data()["scalar_call_testsets"]


def get_implicator_boundary_cases() -> list[dict[str, Any]]:
    """Return scalar boundary cases for implicator contracts."""
    return _load_implicator_reference_data()["boundary_cases"]


def get_implicator_branch_edge_cases() -> list[dict[str, Any]]:
    """Return branch-sensitive vector cases for implicator contracts."""
    return _load_implicator_reference_data()["branch_edge_cases"]


def owa_weights_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for OWA weight generation."""
    return load_reference_data(
        "owa_weights.json",
        expected_dataset_kind="owa_weights",
    )


def _load_similarity_reference_data() -> dict[str, Any]:
    """Return the grouped similarity reference-data payload."""
    return load_reference_data(
        "similarities.json",
        expected_dataset_kind="similarities",
    )


def get_similarity_testing_testsets() -> list[dict[str, Any]]:
    """Return legacy reference cases for similarity matrix construction."""
    return _load_similarity_reference_data()["matrix_testsets"]


def get_dense_similarity_baseline_testsets() -> list[dict[str, Any]]:
    """Return focused dense similarity baseline cases."""
    return _load_similarity_reference_data()["dense_baselines"]


def get_fuzzy_quantifier_testsets() -> list[dict[str, Any]]:
    """Return formula-based fuzzy-quantifier reference cases."""
    return load_reference_data(
        "fuzzy_quantifiers.json",
        expected_dataset_kind="fuzzy_quantifiers",
    )


def get_dense_approximation_baseline_testsets() -> list[dict[str, Any]]:
    """Return public dense approximation baseline cases."""
    return load_reference_data(
        "approximation_baselines.json",
        expected_dataset_kind="approximation_baselines",
    )


def _load_itfrs_reference_data() -> dict[str, Any]:
    """Return the grouped ITFRS reference-data payload."""
    return load_reference_data(
        "itfrs.json",
        expected_dataset_kind="itfrs",
    )


def get_ITFRS_testing_testsets() -> list[dict[str, Any]]:
    """Return legacy reference cases for ITFRS approximation computations."""
    return _load_itfrs_reference_data()["legacy_testsets"]


def get_itfrs_dense_baseline_testsets() -> list[dict[str, Any]]:
    """Return focused dense ITFRS baseline cases."""
    return _load_itfrs_reference_data()["dense_baselines"]


def _load_vqrs_reference_data() -> dict[str, Any]:
    """Return the grouped VQRS reference-data payload."""
    return load_reference_data(
        "vqrs.json",
        expected_dataset_kind="vqrs",
    )


def get_VQRS_testing_testsets() -> list[dict[str, Any]]:
    """Return legacy reference cases for VQRS approximation computations."""
    return _load_vqrs_reference_data()["legacy_testsets"]


def get_vqrs_dense_baseline_testsets() -> list[dict[str, Any]]:
    """Return focused dense VQRS baseline cases."""
    payload = _load_vqrs_reference_data()
    expanded_cases: list[dict[str, Any]] = []

    for fixture in payload["dense_fixtures"]:
        for baseline in fixture["baselines"]:
            expanded_cases.append(
                {
                    "name": baseline["name"],
                    "similarity_matrix": fixture["similarity_matrix"],
                    "labels": fixture["labels"],
                    "components": baseline["components"],
                    "expected": baseline["expected"],
                }
            )

    return expanded_cases


def _load_owafrs_reference_data() -> dict[str, Any]:
    """Return the grouped OWAFRS reference-data payload."""
    return load_reference_data(
        "owafrs.json",
        expected_dataset_kind="owafrs",
    )


def get_OWAFRS_testing_testsets() -> list[dict[str, Any]]:
    """Return legacy reference cases for OWAFRS approximation computations."""
    return _load_owafrs_reference_data()["legacy_testsets"]


def get_owafrs_dense_baseline_testsets() -> list[dict[str, Any]]:
    """Return focused dense OWAFRS baseline cases."""
    return _load_owafrs_reference_data()["dense_baselines"]
