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
