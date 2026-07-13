# SPDX-License-Identifier: BSD-3-Clause
"""Build the phase-one inventory for migrating inline test oracles to JSON.

The generated files document and snapshot inline scientific reference values
without changing any test or making the snapshot part of the canonical
reference-data manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_JSON_OUTPUT = (
    _REPOSITORY_ROOT
    / "docs"
    / "developer"
    / "reference_data_migration_inventory.json"
)
_MARKDOWN_OUTPUT = (
    _REPOSITORY_ROOT
    / "docs"
    / "developer"
    / "reference_data_migration_inventory.md"
)
_SOURCE_PATHS = (
    "tests/api/test_dense_approximation_baseline_contract.py",
    "tests/api/test_dense_similarity_baseline_contract.py",
    "tests/models_tests/test_itfrs_fast.py",
    "tests/models_tests/test_vqrs_fast.py",
    "tests/models_tests/test_owafrs_fast.py",
    "tests/core_tests/test_implicators.py",
    "tests/core_tests/test_fuzzy_quantifiers.py",
    "tests/core_tests/test_approximation_engines.py",
)


def _load_module(relative_path: str, module_name: str) -> ModuleType:
    """Load a repository Python file as an isolated module."""
    path = _REPOSITORY_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode_array(value: Any) -> dict[str, Any]:
    """Encode an array with exact dtype, shape, and JSON-compatible values."""
    array = np.asarray(value)
    return {
        "__ndarray__": array.tolist(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _decode_inventory_array(value: dict[str, Any]) -> np.ndarray:
    """Decode an inventory array for exact round-trip validation."""
    array = np.asarray(value["__ndarray__"], dtype=np.dtype(value["dtype"]))
    if list(array.shape) != value["shape"]:
        raise ValueError(
            "Inventory array shape mismatch: "
            f"declared {value['shape']!r}, decoded {list(array.shape)!r}."
        )
    return array


def _source_location(function: Any) -> str:
    """Return a compact source line range for a Python function."""
    source_lines, start_line = inspect.getsourcelines(function)
    end_line = start_line + len(source_lines) - 1
    return f"{function.__code__.co_filename}:{start_line}-{end_line}"


def _extract_local_array(function: Any, variable_name: str) -> np.ndarray:
    """Extract a local ``np.array`` assignment from a trusted test function."""
    import ast

    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        expression = ast.Expression(body=node.value)
        compiled = compile(expression, filename="<reference-inventory>", mode="eval")
        value = eval(compiled, {"np": np}, {})  # noqa: S307 - trusted repository source
        return np.asarray(value)
    raise ValueError(
        f"Could not find local assignment {variable_name!r} in {function.__name__}."
    )


def _linear_quantifier(values: np.ndarray, *, alpha: float, beta: float) -> np.ndarray:
    """Compute the linear quantifier independently from production classes."""
    values = np.asarray(values, dtype=float)
    return np.where(
        values <= alpha,
        0.0,
        np.where(values >= beta, 1.0, (values - alpha) / (beta - alpha)),
    )


def _quadratic_quantifier(
    values: np.ndarray,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Compute the quadratic quantifier independently from production classes."""
    values = np.asarray(values, dtype=float)
    midpoint = (alpha + beta) / 2.0
    denominator = (beta - alpha) ** 2.0
    lower_curve = 2.0 * ((values - alpha) ** 2.0) / denominator
    upper_curve = 1.0 - 2.0 * ((values - beta) ** 2.0) / denominator
    return np.where(
        values <= alpha,
        0.0,
        np.where(
            values <= midpoint,
            lower_curve,
            np.where(values <= beta, upper_curve, 1.0),
        ),
    )


def _independent_vqrs_values(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    lower_kind: str,
    lower_alpha: float,
    lower_beta: float,
    upper_kind: str,
    upper_alpha: float,
    upper_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute VQRS reference outputs using explicit formulas only."""
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    labels = np.asarray(labels)
    label_mask = (labels[:, None] == labels[None, :]).astype(float)
    tnorm_values = np.minimum(similarity_matrix, label_mask)
    np.fill_diagonal(tnorm_values, 0.0)
    numerator = np.sum(tnorm_values, axis=1)
    denominator = np.sum(similarity_matrix, axis=1) - 1.0
    interim = numerator / denominator

    quantifiers = {
        "linear": _linear_quantifier,
        "quadratic": _quadratic_quantifier,
    }
    lower = quantifiers[lower_kind](
        interim,
        alpha=lower_alpha,
        beta=lower_beta,
    )
    upper = quantifiers[upper_kind](
        interim,
        alpha=upper_alpha,
        beta=upper_beta,
    )
    return lower, upper, upper - lower, lower.copy(), interim


def _independent_owafrs_values(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the selected OWAFRS case using explicit linear OWA formulas."""
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    labels = np.asarray(labels)
    label_mask = (labels[:, None] == labels[None, :]).astype(float)
    lower_evidence = np.minimum(1.0, 1.0 - similarity_matrix + label_mask)
    upper_evidence = np.minimum(similarity_matrix, label_mask)
    np.fill_diagonal(lower_evidence, 0.0)
    np.fill_diagonal(upper_evidence, 0.0)

    sorted_lower = np.sort(lower_evidence, axis=1)[:, ::-1][:, :-1]
    sorted_upper = np.sort(upper_evidence, axis=1)[:, ::-1][:, :-1]
    compared_count = similarity_matrix.shape[0] - 1
    raw_weights = np.arange(1, compared_count + 1, dtype=np.longdouble)
    normalized = raw_weights / raw_weights.sum()
    lower_weights = np.sort(normalized)
    upper_weights = np.sort(normalized)[::-1]

    lower = np.asarray(sorted_lower @ lower_weights, dtype=float)
    upper = np.asarray(sorted_upper @ upper_weights, dtype=float)
    return lower, upper, upper - lower, lower.copy()


def _independent_itfrs_values(
    similarity_matrix: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute minimum/Lukasiewicz ITFRS values from defining formulas."""
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    labels = np.asarray(labels)
    label_mask = (labels[:, None] == labels[None, :]).astype(float)
    implication = np.minimum(1.0, 1.0 - similarity_matrix + label_mask)
    upper_evidence = np.minimum(similarity_matrix, label_mask)
    np.fill_diagonal(implication, 1.0)
    np.fill_diagonal(upper_evidence, 0.0)
    lower = np.min(implication, axis=1)
    upper = np.max(upper_evidence, axis=1)
    return lower, upper, upper - lower, lower.copy()


def _assert_exact(left: Any, right: Any, *, context: str) -> None:
    """Assert exact equality including dtype and shape for two arrays."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype:
        raise AssertionError(
            f"{context}: dtype mismatch {left_array.dtype} != {right_array.dtype}."
        )
    if left_array.shape != right_array.shape:
        raise AssertionError(
            f"{context}: shape mismatch {left_array.shape} != {right_array.shape}."
        )
    if not np.array_equal(left_array, right_array):
        raise AssertionError(f"{context}: values are not exactly equal.")




def _assert_formula_match(left: Any, right: Any, *, context: str) -> None:
    """Assert a strict numerical match for independently evaluated formulas."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype:
        raise AssertionError(
            f"{context}: dtype mismatch {left_array.dtype} != {right_array.dtype}."
        )
    if left_array.shape != right_array.shape:
        raise AssertionError(
            f"{context}: shape mismatch {left_array.shape} != {right_array.shape}."
        )
    np.testing.assert_allclose(
        left_array,
        right_array,
        rtol=0.0,
        atol=1e-15,
        err_msg=context,
    )


def _case(
    *,
    case_id: str,
    target_json: str,
    sources: list[str],
    provenance: str,
    source_tolerance: str,
    data: dict[str, Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build one normalized migration-inventory case record."""
    return {
        "id": case_id,
        "target_json": target_json,
        "source_locations": sources,
        "provenance": provenance,
        "source_assertion_tolerance": source_tolerance,
        "data": data,
        "notes": notes or [],
    }


def _build_inventory() -> dict[str, Any]:
    """Build the complete phase-one migration inventory."""
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))

    modules = {
        path: _load_module(path, f"_frsutils_reference_inventory_{index}")
        for index, path in enumerate(_SOURCE_PATHS)
    }
    dense_approx = modules[_SOURCE_PATHS[0]]
    dense_similarity = modules[_SOURCE_PATHS[1]]
    itfrs_fast = modules[_SOURCE_PATHS[2]]
    vqrs_fast = modules[_SOURCE_PATHS[3]]
    owafrs_fast = modules[_SOURCE_PATHS[4]]
    implicators = modules[_SOURCE_PATHS[5]]
    quantifiers = modules[_SOURCE_PATHS[6]]
    engines = modules[_SOURCE_PATHS[7]]

    vqrs_model = vqrs_fast._build_reference_model()
    vqrs_helper_values = vqrs_fast._manual_vqrs_values(
        vqrs_fast.VQRS_SIMILARITY_MATRIX,
        vqrs_fast.VQRS_LABELS,
        lb_fuzzy_quantifier=vqrs_model.lb_fuzzy_quantifier,
        ub_fuzzy_quantifier=vqrs_model.ub_fuzzy_quantifier,
    )
    vqrs_independent_values = _independent_vqrs_values(
        vqrs_fast.VQRS_SIMILARITY_MATRIX,
        vqrs_fast.VQRS_LABELS,
        lower_kind="linear",
        lower_alpha=0.1,
        lower_beta=0.6,
        upper_kind="quadratic",
        upper_alpha=0.0,
        upper_beta=0.8,
    )
    for index, name in enumerate(
        ("lower", "upper", "boundary", "positive_region", "interim")
    ):
        _assert_exact(
            vqrs_helper_values[index],
            vqrs_independent_values[index],
            context=f"VQRS fast {name}",
        )

    owafrs_labels = np.array(["a", "a", "b", "b"], dtype=object)
    owafrs_matrix = owafrs_fast._small_similarity_matrix()
    owafrs_helper_values = owafrs_fast._manual_owafrs_values(
        owafrs_matrix,
        owafrs_labels,
    )
    owafrs_independent_values = _independent_owafrs_values(
        owafrs_matrix,
        owafrs_labels,
    )
    for index, name in enumerate(
        ("lower", "upper", "boundary", "positive_region")
    ):
        _assert_exact(
            owafrs_helper_values[index],
            owafrs_independent_values[index],
            context=f"OWAFRS fast {name}",
        )

    engine_itfrs_helper = engines._manual_itfrs_expected(
        engines.ITFRS_SIMILARITY_MATRIX,
        engines.ITFRS_LABELS,
    )
    engine_itfrs_independent = _independent_itfrs_values(
        engines.ITFRS_SIMILARITY_MATRIX,
        engines.ITFRS_LABELS,
    )
    for index, name in enumerate(
        ("lower", "upper", "boundary", "positive_region")
    ):
        _assert_exact(
            engine_itfrs_helper[index],
            engine_itfrs_independent[index],
            context=f"blockwise ITFRS {name}",
        )

    engine_vqrs_helper = engines._manual_vqrs_expected(
        engines.VQRS_SIMILARITY_MATRIX,
        engines.VQRS_LABELS,
    )
    engine_vqrs_independent = _independent_vqrs_values(
        engines.VQRS_SIMILARITY_MATRIX,
        engines.VQRS_LABELS,
        lower_kind="linear",
        lower_alpha=0.1,
        lower_beta=0.6,
        upper_kind="linear",
        upper_alpha=0.1,
        upper_beta=0.6,
    )
    for index, name in enumerate(
        ("lower", "upper", "boundary", "positive_region", "interim")
    ):
        _assert_exact(
            engine_vqrs_helper[index],
            engine_vqrs_independent[index],
            context=f"blockwise VQRS {name}",
        )

    _assert_exact(
        dense_approx.X_BASELINE,
        dense_similarity.X_ONE_FEATURE,
        context="shared one-feature dense input",
    )
    _assert_exact(
        vqrs_fast.VQRS_SIMILARITY_MATRIX,
        engines.VQRS_SIMILARITY_MATRIX,
        context="shared VQRS similarity matrix",
    )
    _assert_exact(
        vqrs_fast.VQRS_LABELS,
        engines.VQRS_LABELS,
        context="shared VQRS labels",
    )

    linear_1d_x = _extract_local_array(
        quantifiers.test_linear_quantifier_matches_piecewise_formula_at_internal_points,
        "x",
    )
    linear_1d_expected = _extract_local_array(
        quantifiers.test_linear_quantifier_matches_piecewise_formula_at_internal_points,
        "expected",
    )
    quadratic_1d_x = _extract_local_array(
        quantifiers.test_quadratic_quantifier_matches_piecewise_formula_at_internal_points,
        "x",
    )
    quadratic_1d_expected = _extract_local_array(
        quantifiers.test_quadratic_quantifier_matches_piecewise_formula_at_internal_points,
        "expected",
    )
    linear_2d_x = _extract_local_array(
        quantifiers.test_linear_quantifier_matches_formula_on_two_dimensional_input,
        "x",
    )
    linear_2d_expected = _extract_local_array(
        quantifiers.test_linear_quantifier_matches_formula_on_two_dimensional_input,
        "expected",
    )
    quadratic_2d_x = _extract_local_array(
        quantifiers.test_quadratic_quantifier_matches_formula_on_two_dimensional_input,
        "x",
    )
    quadratic_2d_expected = _extract_local_array(
        quantifiers.test_quadratic_quantifier_matches_formula_on_two_dimensional_input,
        "expected",
    )
    _assert_formula_match(
        linear_1d_expected,
        _linear_quantifier(linear_1d_x, alpha=0.2, beta=0.8),
        context="linear quantifier one-dimensional formula",
    )
    _assert_formula_match(
        quadratic_1d_expected,
        _quadratic_quantifier(quadratic_1d_x, alpha=0.2, beta=0.8),
        context="quadratic quantifier one-dimensional formula",
    )
    _assert_formula_match(
        linear_2d_expected,
        _linear_quantifier(linear_2d_x, alpha=0.2, beta=0.8),
        context="linear quantifier two-dimensional formula",
    )
    _assert_formula_match(
        quadratic_2d_expected,
        _quadratic_quantifier(quadratic_2d_x, alpha=0.2, beta=0.8),
        context="quadratic quantifier two-dimensional formula",
    )

    cases: list[dict[str, Any]] = []
    cases.append(
        _case(
            case_id="dense_approximation_four_sample_linear_similarity",
            target_json="approximation_baselines.json",
            sources=[f"{_SOURCE_PATHS[0]}:17-40"],
            provenance="literal_expected_values",
            source_tolerance="atol=1e-12",
            data={
                "X": _encode_array(dense_approx.X_BASELINE),
                "labels": _encode_array(dense_approx.Y_BASELINE),
                "similarity": {"name": "linear", "params": {}},
                "expected_by_model": {
                    model_name: {
                        output_name: _encode_array(output)
                        for output_name, output in outputs.items()
                    }
                    for model_name, outputs in dense_approx.EXPECTED_BY_MODEL.items()
                },
            },
            notes=[
                (
                    "The OWAFRS boundary values are negative in the current locked "
                    "baseline; migration must preserve them verbatim."
                ),
                (
                    "X is identical to the one-feature input in the dense similarity "
                    "baseline and should be stored only once if the final schema "
                    "supports shared fixtures."
                ),
            ],
        )
    )

    similarity_cases = (
        (
            "dense_linear_one_feature",
            dense_similarity.X_ONE_FEATURE,
            "linear",
            {"tnorm": "minimum"},
            dense_similarity.EXPECTED_LINEAR_ONE_FEATURE,
            "tests/api/test_dense_similarity_baseline_contract.py:9-30",
        ),
        (
            "dense_linear_two_features_minimum",
            dense_similarity.X_TWO_FEATURES,
            "linear",
            {"tnorm": "minimum"},
            dense_similarity.EXPECTED_LINEAR_TWO_FEATURES_MINIMUM,
            "tests/api/test_dense_similarity_baseline_contract.py:10-41",
        ),
        (
            "dense_gaussian_one_feature_sigma_0_5",
            dense_similarity.X_ONE_FEATURE,
            "gaussian",
            {"sigma": 0.5, "tnorm": "minimum"},
            dense_similarity.EXPECTED_GAUSSIAN_ONE_FEATURE_SIGMA_05,
            "tests/api/test_dense_similarity_baseline_contract.py:9-52",
        ),
    )
    for case_id, values, similarity_name, params, expected, source in similarity_cases:
        cases.append(
            _case(
                case_id=case_id,
                target_json="similarities.json",
                sources=[source],
                provenance="literal_expected_values",
                source_tolerance="atol=1e-12",
                data={
                    "X": _encode_array(values),
                    "similarity": {"name": similarity_name, "params": params},
                    "expected": _encode_array(expected),
                },
            )
        )

    cases.append(
        _case(
            case_id="itfrs_dense_fast_minimum_lukasiewicz",
            target_json="itfrs.json",
            sources=[f"{_SOURCE_PATHS[2]}:14-45"],
            provenance="literal_hand_computed_expected_values",
            source_tolerance="atol=1e-12",
            data={
                "similarity_matrix": _encode_array(itfrs_fast.SIMILARITY_MATRIX),
                "labels": _encode_array(itfrs_fast.LABELS),
                "components": {
                    "upper_tnorm": "minimum",
                    "lower_implicator": "lukasiewicz",
                },
                "expected": {
                    "lower": _encode_array(itfrs_fast.EXPECTED_LOWER),
                    "upper": _encode_array(itfrs_fast.EXPECTED_UPPER),
                    "boundary": _encode_array(
                        itfrs_fast.EXPECTED_UPPER - itfrs_fast.EXPECTED_LOWER
                    ),
                    "positive_region": _encode_array(itfrs_fast.EXPECTED_LOWER),
                },
            },
        )
    )

    cases.append(
        _case(
            case_id="vqrs_dense_fast_asymmetric_quantifiers",
            target_json="vqrs.json",
            sources=[f"{_SOURCE_PATHS[3]}:16-91"],
            provenance="helper_output_exactly_verified_against_independent_formula",
            source_tolerance="atol=1e-12",
            data={
                "similarity_matrix": _encode_array(vqrs_fast.VQRS_SIMILARITY_MATRIX),
                "labels": _encode_array(vqrs_fast.VQRS_LABELS),
                "components": {
                    "lower_quantifier": {
                        "name": "linear",
                        "params": {"alpha": 0.1, "beta": 0.6},
                    },
                    "upper_quantifier": {
                        "name": "quadratic",
                        "params": {"alpha": 0.0, "beta": 0.8},
                    },
                },
                "expected": {
                    "lower": _encode_array(vqrs_independent_values[0]),
                    "upper": _encode_array(vqrs_independent_values[1]),
                    "boundary": _encode_array(vqrs_independent_values[2]),
                    "positive_region": _encode_array(vqrs_independent_values[3]),
                    "interim": _encode_array(vqrs_independent_values[4]),
                },
            },
            notes=[
                "The same matrix and labels also appear in test_approximation_engines.py.",
                (
                    "The current loader supports only float64 and int64 encoded "
                    "arrays; final migration should store string labels as a JSON "
                    "list plus explicit label metadata unless loader support is "
                    "deliberately expanded."
                ),
            ],
        )
    )

    cases.append(
        _case(
            case_id="owafrs_dense_fast_linear_owa",
            target_json="owafrs.json",
            sources=[f"{_SOURCE_PATHS[4]}:14-68"],
            provenance="helper_output_exactly_verified_against_independent_formula",
            source_tolerance="np.testing.assert_allclose defaults: rtol=1e-7, atol=0",
            data={
                "similarity_matrix": _encode_array(owafrs_matrix),
                "labels": _encode_array(owafrs_labels),
                "components": {
                    "upper_tnorm": "minimum",
                    "lower_implicator": "lukasiewicz",
                    "upper_owa": "linear_descending",
                    "lower_owa": "linear_ascending",
                },
                "expected": {
                    "lower": _encode_array(owafrs_independent_values[0]),
                    "upper": _encode_array(owafrs_independent_values[1]),
                    "boundary": _encode_array(owafrs_independent_values[2]),
                    "positive_region": _encode_array(owafrs_independent_values[3]),
                },
            },
            notes=[
                (
                    "The current expected helper calls the production "
                    "LinearOWAWeights class; phase one independently reproduced "
                    "its 1..n normalized linear formula and obtained exact equality."
                ),
                "Final migration should keep assertion tolerances in Python rather than moving them into JSON.",
            ],
        )
    )

    cases.append(
        _case(
            case_id="implicator_boundary_scalar_cases",
            target_json="implicator_scalar.json",
            sources=[f"{_SOURCE_PATHS[5]}:174-218"],
            provenance="literal_formula_and_boundary_expected_values",
            source_tolerance="atol=1e-12",
            data={
                "cases": [
                    {
                        "implicator": name,
                        "a": a,
                        "b": b,
                        "expected": expected,
                    }
                    for name, a, b, expected in implicators.BOUNDARY_CASES
                ]
            },
        )
    )
    cases.append(
        _case(
            case_id="implicator_vector_branch_edge_cases",
            target_json="implicator_scalar.json",
            sources=[f"{_SOURCE_PATHS[5]}:224-267"],
            provenance="literal_branch_expected_values",
            source_tolerance="atol=1e-12",
            data={
                "cases": [
                    {
                        "implicator": name,
                        "a": _encode_array(a),
                        "b": _encode_array(b),
                        "expected": _encode_array(expected),
                    }
                    for name, a, b, expected in implicators.BRANCH_EDGE_CASES
                ]
            },
        )
    )

    quantifier_specs = (
        (
            "fuzzy_quantifier_linear_piecewise_1d",
            quantifiers.test_linear_quantifier_matches_piecewise_formula_at_internal_points,
            linear_1d_x,
            linear_1d_expected,
            "linear",
        ),
        (
            "fuzzy_quantifier_quadratic_piecewise_1d",
            quantifiers.test_quadratic_quantifier_matches_piecewise_formula_at_internal_points,
            quadratic_1d_x,
            quadratic_1d_expected,
            "quadratic",
        ),
        (
            "fuzzy_quantifier_linear_piecewise_2d",
            quantifiers.test_linear_quantifier_matches_formula_on_two_dimensional_input,
            linear_2d_x,
            linear_2d_expected,
            "linear",
        ),
        (
            "fuzzy_quantifier_quadratic_piecewise_2d",
            quantifiers.test_quadratic_quantifier_matches_formula_on_two_dimensional_input,
            quadratic_2d_x,
            quadratic_2d_expected,
            "quadratic",
        ),
    )
    for case_id, function, values, expected, quantifier_name in quantifier_specs:
        cases.append(
            _case(
                case_id=case_id,
                target_json="fuzzy_quantifiers.json",
                sources=[_source_location(function).replace(str(_REPOSITORY_ROOT) + "/", "")],
                provenance="literal_expected_values_verified_against_independent_formula",
                source_tolerance="atol=1e-12",
                data={
                    "quantifier": {
                        "name": quantifier_name,
                        "params": {"alpha": 0.2, "beta": 0.8},
                    },
                    "x": _encode_array(values),
                    "expected": _encode_array(expected),
                },
            )
        )

    cases.append(
        _case(
            case_id="itfrs_blockwise_default_components",
            target_json="itfrs.json",
            sources=[f"{_SOURCE_PATHS[7]}:75-98", f"{_SOURCE_PATHS[7]}:795-825"],
            provenance="helper_output_exactly_verified_against_independent_formula",
            source_tolerance="atol=1e-12",
            data={
                "similarity_matrix": _encode_array(engines.ITFRS_SIMILARITY_MATRIX),
                "labels": _encode_array(engines.ITFRS_LABELS),
                "components": {
                    "upper_tnorm": "minimum",
                    "lower_implicator": "lukasiewicz",
                },
                "expected": {
                    "lower": _encode_array(engine_itfrs_independent[0]),
                    "upper": _encode_array(engine_itfrs_independent[1]),
                    "boundary": _encode_array(engine_itfrs_independent[2]),
                    "positive_region": _encode_array(engine_itfrs_independent[3]),
                },
            },
        )
    )
    cases.append(
        _case(
            case_id="vqrs_blockwise_default_quantifiers",
            target_json="vqrs.json",
            sources=[f"{_SOURCE_PATHS[7]}:101-136", f"{_SOURCE_PATHS[7]}:962-993"],
            provenance="helper_output_exactly_verified_against_independent_formula",
            source_tolerance="atol=1e-12",
            data={
                "shared_fixture_id": "vqrs_four_sample_similarity_and_labels",
                "similarity_matrix": _encode_array(engines.VQRS_SIMILARITY_MATRIX),
                "labels": _encode_array(engines.VQRS_LABELS),
                "components": {
                    "lower_quantifier": {
                        "name": "linear",
                        "params": {"alpha": 0.1, "beta": 0.6},
                    },
                    "upper_quantifier": {
                        "name": "linear",
                        "params": {"alpha": 0.1, "beta": 0.6},
                    },
                },
                "expected": {
                    "lower": _encode_array(engine_vqrs_independent[0]),
                    "upper": _encode_array(engine_vqrs_independent[1]),
                    "boundary": _encode_array(engine_vqrs_independent[2]),
                    "positive_region": _encode_array(engine_vqrs_independent[3]),
                    "interim": _encode_array(engine_vqrs_independent[4]),
                },
            },
            notes=[
                (
                    "Reuse the same final JSON fixture as the dense VQRS fast case; "
                    "only the quantifier configuration and expected outputs differ."
                ),
            ],
        )
    )

    return {
        "schema_version": 1,
        "inventory_kind": "inline_reference_data_migration_phase_1",
        "status": "candidate_snapshot_not_canonical_reference_data",
        "generated_on": "2026-07-13",
        "scope": {
            "tests_modified": False,
            "reference_manifest_modified": False,
            "source_files": [
                {
                    "path": relative_path,
                    "sha256": _sha256(_REPOSITORY_ROOT / relative_path),
                }
                for relative_path in _SOURCE_PATHS
            ],
        },
        "policies": {
            "tolerances": "Assertion tolerances remain in Python tests and must not become mutable JSON data.",
            "production_independence": (
                "Helper-derived candidates were independently recomputed from "
                "defining formulas and required exact equality before inclusion."
            ),
            "migration_safety": (
                "Final migration must preserve values, dtype, and shape exactly "
                "and must not change test count."
            ),
            "label_encoding": (
                "Because the canonical loader currently supports only float64 and "
                "int64 encoded arrays, string/object labels should initially be "
                "stored as JSON lists with explicit label metadata rather than as "
                "encoded ndarrays."
            ),
        },
        "duplicate_groups": [
            {
                "id": "dense_one_feature_input",
                "members": [
                    "tests/api/test_dense_approximation_baseline_contract.py:X_BASELINE",
                    "tests/api/test_dense_similarity_baseline_contract.py:X_ONE_FEATURE",
                ],
                "verification": "exact values, dtype, and shape",
                "decision": (
                    "Keep one canonical fixture if the final schemas support shared "
                    "fixture references; otherwise duplicate only when domain files "
                    "must remain self-contained."
                ),
            },
            {
                "id": "vqrs_four_sample_similarity_and_labels",
                "members": [
                    "tests/models_tests/test_vqrs_fast.py:VQRS_SIMILARITY_MATRIX/VQRS_LABELS",
                    "tests/core_tests/test_approximation_engines.py:VQRS_SIMILARITY_MATRIX/VQRS_LABELS",
                ],
                "verification": "exact values, dtype, and shape",
                "decision": (
                    "Store the fixture once in vqrs.json with multiple component "
                    "configurations and expected-output records."
                ),
            },
            {
                "id": "fuzzy_quantifier_coarse_cases",
                "members": [
                    "test_quantifier_known_outputs",
                    "piecewise formula tests",
                    "boundary and midpoint property tests",
                ],
                "verification": "the coarse values are subsets of the seven-point piecewise cases",
                "decision": (
                    "Use the full one-dimensional linear and quadratic piecewise "
                    "cases as canonical data; keep property-oriented test inputs "
                    "local where they test API shape or boundaries rather than a "
                    "separate oracle."
                ),
            },
        ],
        "cases": cases,
        "summary": {
            "source_file_count": len(_SOURCE_PATHS),
            "candidate_case_count": len(cases),
            "new_target_json_files": [
                "approximation_baselines.json",
                "fuzzy_quantifiers.json",
            ],
            "existing_target_json_files_to_extend": [
                "implicator_scalar.json",
                "itfrs.json",
                "owafrs.json",
                "similarities.json",
                "vqrs.json",
            ],
        },
    }


def _validate_inventory(inventory: dict[str, Any]) -> None:
    """Validate inventory array round trips and case identifier uniqueness."""
    identifiers = [case["id"] for case in inventory["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Migration inventory contains duplicate case identifiers.")

    def visit(value: Any) -> None:
        """Recursively validate encoded inventory arrays."""
        if isinstance(value, dict):
            if set(value) == {"__ndarray__", "dtype", "shape"}:
                decoded = _decode_inventory_array(value)
                reencoded = _encode_array(decoded)
                if reencoded != value:
                    raise ValueError("Inventory array did not survive exact round trip.")
                return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(inventory)


def _render_markdown(inventory: dict[str, Any]) -> str:
    """Render a human-readable summary of the phase-one inventory."""
    source_rows = "\n".join(
        f"| `{entry['path']}` | `{entry['sha256']}` |"
        for entry in inventory["scope"]["source_files"]
    )
    target_rows = "\n".join(
        f"| `{case['id']}` | `{case['target_json']}` | {case['provenance']} | {case['source_assertion_tolerance']} |"
        for case in inventory["cases"]
    )
    duplicate_sections = "\n\n".join(
        (
            f"### `{group['id']}`\n\n"
            + "\n".join(f"- `{member}`" for member in group["members"])
            + f"\n\n**Decision:** {group['decision']}"
        )
        for group in inventory["duplicate_groups"]
    )
    return f"""# Inline Reference Data Migration Inventory

## Status

This is the phase-one inventory for migrating scattered scientific test oracles to
JSON. No test, loader, manifest entry, or canonical reference-data file is changed
by this phase. The machine-readable snapshot is stored in
`reference_data_migration_inventory.json` beside this document.

The inventory was generated from {inventory['summary']['source_file_count']} source
files and contains {inventory['summary']['candidate_case_count']} candidate reference
cases. Every helper-derived output included in the snapshot was independently
recomputed from an explicit mathematical formula. Captured arrays and duplicate
fixtures were checked exactly for value, dtype, and shape; formula checks used a
strict absolute tolerance of `1e-15` where decimal floating-point evaluation can
differ by a final machine-representation bit.

## Source files locked for this inventory

| Source file | SHA-256 |
|---|---|
{source_rows}

If any source digest changes, regenerate and review this inventory before starting
the next migration phase.

## Canonical target map

| Candidate case | Planned JSON | Provenance | Current test tolerance |
|---|---|---|---|
{target_rows}

Planned new canonical files:

- `tests/reference_data/approximation_baselines.json`
- `tests/reference_data/fuzzy_quantifiers.json`

Planned existing files to extend:

- `tests/reference_data/implicator_scalar.json`
- `tests/reference_data/itfrs.json`
- `tests/reference_data/owafrs.json`
- `tests/reference_data/similarities.json`
- `tests/reference_data/vqrs.json`

## Duplicate and consolidation decisions

{duplicate_sections}

## Important migration constraints

1. Assertion tolerances remain in Python tests. They must not be moved to JSON,
   because changing both an expected value and its tolerance could silently weaken
   a scientific regression test.
2. `tests/reference_data_loader.py` currently supports encoded `float64` and
   `int64` arrays only. The selected VQRS, OWAFRS, and blockwise fixtures contain
   string or object labels. The low-risk next step is to store those labels as plain
   JSON lists with explicit label metadata, unless loader support is deliberately
   expanded and contract-tested.
3. The dense approximation baseline currently includes negative OWAFRS boundary
   values. Phase two and phase three must copy those values verbatim rather than
   normalizing, clipping, or recomputing them from current production code.
4. The VQRS and OWAFRS fast tests currently build expected results through helpers
   that call production quantifier or OWA components. This inventory independently
   reproduced those formulas and verified exact equality before freezing the
   candidate values.
5. The main blockwise ITFRS and VQRS outputs were also independently reproduced
   from their defining formulas. Configuration-routing cases that use alternate
   components remain local contract tests unless a separate scientific oracle is
   justified.
6. Inputs for shape, validation, invalid values, empty datasets, singleton datasets,
   backend routing, aliases, and serialization remain in Python. They are software
   fixtures, not stable scientific reference data.
7. Later phases must preserve the number of tests. Only the location from which
   expected values are loaded should change.

## Phase-one completion criteria

- All eight agreed source files are represented.
- Candidate values, dtypes, and shapes are machine-readable.
- Duplicate fixtures have explicit consolidation decisions.
- Helper-derived outputs have an independent formula check.
- Source assertion tolerances are documented but excluded from candidate data.
- No existing project file was changed by phase one.
"""


def _write_outputs(inventory: dict[str, Any]) -> None:
    """Write deterministic JSON and Markdown inventory files."""
    _JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _JSON_OUTPUT.write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _MARKDOWN_OUTPUT.write_text(_render_markdown(inventory), encoding="utf-8")


def main() -> None:
    """Build, validate, and write the phase-one migration inventory."""
    inventory = _build_inventory()
    _validate_inventory(inventory)
    _write_outputs(inventory)
    print(f"Wrote {_JSON_OUTPUT.relative_to(_REPOSITORY_ROOT)}")
    print(f"Wrote {_MARKDOWN_OUTPUT.relative_to(_REPOSITORY_ROOT)}")
    print(
        "Captured "
        f"{inventory['summary']['candidate_case_count']} candidate cases from "
        f"{inventory['summary']['source_file_count']} source files."
    )


if __name__ == "__main__":
    main()
