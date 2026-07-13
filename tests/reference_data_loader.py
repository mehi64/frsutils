# SPDX-License-Identifier: BSD-3-Clause
"""Load and validate JSON reference data used by the FRsutils test suite.

The loader verifies file integrity, rejects ambiguous JSON objects, restores
NumPy arrays with their declared dtype and shape, and marks loaded arrays as
read-only.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


_REFERENCE_DATA_DIR = Path(__file__).with_name("reference_data")
_MANIFEST_PATH = _REFERENCE_DATA_DIR / "manifest.json"
_SUPPORTED_SCHEMA_VERSION = 1
_SUPPORTED_DTYPES = {"float64", "int64"}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key detected: {key!r}")
        result[key] = value
    return result


def _reject_non_standard_number(value: str) -> None:
    """Reject non-standard JSON numbers such as NaN and Infinity."""
    raise ValueError(f"Non-standard JSON numeric value detected: {value}")


def _parse_json_bytes(raw: bytes, *, source: Path) -> Any:
    """Parse JSON bytes using strict duplicate-key and number handling."""
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_standard_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid reference-data JSON in {source}: {exc}") from exc


@lru_cache(maxsize=1)
def _load_manifest() -> dict[str, Any]:
    """Load and validate the reference-data integrity manifest."""
    if not _MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Reference-data manifest was not found: {_MANIFEST_PATH}"
        )

    manifest = _parse_json_bytes(_MANIFEST_PATH.read_bytes(), source=_MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise ValueError("Reference-data manifest must contain a JSON object.")
    if manifest.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported reference-data manifest schema version: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError("Reference-data manifest must use SHA-256 hashes.")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Reference-data manifest must define a 'files' object.")

    return manifest


def _read_verified_reference_file(
    filename: str,
    *,
    expected_dataset_kind: str,
) -> bytes:
    """Read one reference file after validating its manifest entry and hash."""
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError(f"Invalid reference-data filename: {filename!r}")

    manifest_entry = _load_manifest()["files"].get(filename)
    if not isinstance(manifest_entry, dict):
        raise ValueError(f"Reference-data file is missing from manifest: {filename}")
    if manifest_entry.get("dataset_kind") != expected_dataset_kind:
        raise ValueError(
            f"Manifest dataset kind mismatch for {filename!r}: expected "
            f"{expected_dataset_kind!r}, found "
            f"{manifest_entry.get('dataset_kind')!r}."
        )

    path = _REFERENCE_DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Reference-data file was not found: {path}")

    raw = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    expected_sha256 = manifest_entry.get("sha256")
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Reference-data integrity check failed for {filename!r}: "
            f"expected SHA-256 {expected_sha256!r}, found {actual_sha256!r}."
        )

    return raw


def _validate_array_scalars(values: Any, *, dtype_name: str) -> None:
    """Validate scalar types before converting encoded values to an array."""
    if isinstance(values, list):
        for item in values:
            _validate_array_scalars(item, dtype_name=dtype_name)
        return

    if dtype_name == "int64":
        if isinstance(values, bool) or not isinstance(values, int):
            raise ValueError(
                "An int64 reference array contains a non-integer JSON value: "
                f"{values!r}."
            )
        return

    if isinstance(values, bool) or not isinstance(values, (int, float)):
        raise ValueError(
            "A float64 reference array contains a non-numeric JSON value: "
            f"{values!r}."
        )
    if not np.isfinite(values):
        raise ValueError("Reference arrays must contain only finite values.")


def _decode_reference_value(value: Any) -> Any:
    """Recursively restore encoded NumPy arrays as read-only arrays."""
    if isinstance(value, dict) and "__ndarray__" in value:
        required_keys = {"__ndarray__", "dtype", "shape"}
        if set(value) != required_keys:
            raise ValueError(
                "Encoded reference arrays must contain exactly the keys "
                f"{sorted(required_keys)!r}."
            )

        dtype_name = value["dtype"]
        if dtype_name not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported reference-array dtype: {dtype_name!r}. "
                f"Supported dtypes are {sorted(_SUPPORTED_DTYPES)!r}."
            )

        shape = value["shape"]
        if (
            not isinstance(shape, list)
            or any(
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
                for dimension in shape
            )
        ):
            raise ValueError(f"Invalid reference-array shape declaration: {shape!r}")

        encoded_values = value["__ndarray__"]
        _validate_array_scalars(encoded_values, dtype_name=dtype_name)
        array = np.asarray(encoded_values, dtype=np.dtype(dtype_name))
        expected_shape = tuple(shape)
        if array.shape != expected_shape:
            raise ValueError(
                "Reference-array shape mismatch: declared "
                f"{expected_shape}, loaded {array.shape}."
            )
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError("Reference arrays must contain only finite values.")

        array.setflags(write=False)
        return array

    if isinstance(value, dict):
        return {key: _decode_reference_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_reference_value(item) for item in value]
    return value


def load_reference_data(
    filename: str,
    *,
    expected_dataset_kind: str,
) -> Any:
    """Load one verified reference-data payload.

    Parameters
    ----------
    filename : str
        JSON filename located in ``tests/reference_data``.
    expected_dataset_kind : str
        Dataset-kind identifier required in both the manifest and JSON payload.

    Returns
    -------
    Any
        Decoded value stored under the payload's ``data`` field. Encoded arrays
        are returned as read-only NumPy arrays.

    Raises
    ------
    FileNotFoundError
        If the manifest or requested reference-data file is missing.
    ValueError
        If integrity, schema, dtype, shape, or JSON validation fails.
    """
    raw = _read_verified_reference_file(
        filename,
        expected_dataset_kind=expected_dataset_kind,
    )
    path = _REFERENCE_DATA_DIR / filename
    payload = _parse_json_bytes(raw, source=path)

    if not isinstance(payload, dict):
        raise ValueError(f"Reference-data payload in {filename!r} must be an object.")
    if payload.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version in {filename!r}: "
            f"{payload.get('schema_version')!r}."
        )
    if payload.get("dataset_kind") != expected_dataset_kind:
        raise ValueError(
            f"Dataset kind mismatch in {filename!r}: expected "
            f"{expected_dataset_kind!r}, found {payload.get('dataset_kind')!r}."
        )
    if "data" not in payload:
        raise ValueError(f"Reference-data payload in {filename!r} has no 'data' field.")

    return _decode_reference_value(payload["data"])
