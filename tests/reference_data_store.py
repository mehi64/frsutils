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


def get_implicator_scalar_testsets() -> list[dict[str, Any]]:
    """Return reference cases for scalar implicator computations."""
    return load_reference_data(
        "implicator_scalar.json",
        expected_dataset_kind="implicator_scalar",
    )


def owa_weights_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for OWA weight generation."""
    return load_reference_data(
        "owa_weights.json",
        expected_dataset_kind="owa_weights",
    )


def get_similarity_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for similarity matrix construction."""
    return load_reference_data(
        "similarities.json",
        expected_dataset_kind="similarities",
    )


def get_ITFRS_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for ITFRS approximation computations."""
    return load_reference_data(
        "itfrs.json",
        expected_dataset_kind="itfrs",
    )


def get_VQRS_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for VQRS approximation computations."""
    return load_reference_data(
        "vqrs.json",
        expected_dataset_kind="vqrs",
    )


def get_OWAFRS_testing_testsets() -> list[dict[str, Any]]:
    """Return reference cases for OWAFRS approximation computations."""
    return load_reference_data(
        "owafrs.json",
        expected_dataset_kind="owafrs",
    )
