"""
@file approximation_engines.py
@brief Exact blockwise fuzzy-rough approximation engines.

This module contains execution helpers that consume SimilarityEngine blocks
without requiring callers to materialize a full pairwise similarity matrix.
Phase 2 introduced an exact ITFRS blockwise accumulator. Phase 4 extends the
same exact blockwise execution contract to VQRS. Phase 5 adds exact OWAFRS
row-buffer execution for the sort/OWA case while keeping dense model behavior
available through the existing model classes and public APIs.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# ITFRSBlockwiseApproximation          Value object for exact ITFRS blockwise outputs
# VQRSBlockwiseApproximation           Value object for exact VQRS blockwise outputs
# OWAFRSBlockwiseApproximation         Value object for exact OWAFRS blockwise outputs
# compute_itfrs_blockwise              Exact ITFRS lower/upper/boundary/positive-region computation
# compute_vqrs_blockwise               Exact VQRS lower/upper/boundary/positive-region computation
# compute_owafrs_blockwise             Exact OWAFRS row-buffer lower/upper computation
# build_itfrs_components_from_config   Resolve ITFRS T-norm/implicator components
# build_vqrs_components_from_config    Resolve VQRS fuzzy-quantifier components
# build_owafrs_components_from_config  Resolve OWAFRS T-norm/implicator/OWA components

# ✅ Design Patterns & Clean Code Notes
# - Strategy Pattern: separates blockwise model execution from dense model classes
# - Adapter Pattern: accepts flat or nested FRsutils configs
# - Streaming Accumulator: keeps only row-level accumulators, not an n x n matrix
# - Row-Buffer Execution: OWAFRS stores one row block at a time for exact sorting
# - Conservative Extension: blockwise support is added without changing dense behavior
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.core.similarity_engine import build_similarity_engine
# from FRsutils.core.approximation_engines import compute_itfrs_blockwise, compute_vqrs_blockwise
#
# engine = build_similarity_engine(X, engine="blockwise", similarity="linear")
# itfrs_result = compute_itfrs_blockwise(engine, y, config={"type": "itfrs"})
# vqrs_result = compute_vqrs_blockwise(engine, y, config={"type": "vqrs"})
# owafrs_result = compute_owafrs_blockwise(engine, y, config={"type": "owafrs"})
# positive_region = owafrs_result.positive_region
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.implicators import Implicator
from FRsutils.core.owa_weights import OWAWeights
from FRsutils.core.similarity_engine import BaseSimilarityEngine
from FRsutils.core.tnorms import TNorm
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


@dataclass(frozen=True)
class ITFRSBlockwiseApproximation:
    """
    @brief Immutable exact ITFRS blockwise approximation outputs.

    @param lower: Lower approximation values.
    @param upper: Upper approximation values.
    @param boundary: Boundary values computed as upper - lower.
    @param positive_region: Positive-region values, identical to lower for ITFRS.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray


@dataclass(frozen=True)
class VQRSBlockwiseApproximation:
    """
    @brief Immutable exact VQRS blockwise approximation outputs.

    @param lower: Lower approximation values after the lower fuzzy quantifier.
    @param upper: Upper approximation values after the upper fuzzy quantifier.
    @param boundary: Boundary values computed as upper - lower.
    @param positive_region: Positive-region values, identical to lower by the base model contract.
    @param interim: Raw VQRS ratio values before fuzzy quantifier application.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    interim: np.ndarray


@dataclass(frozen=True)
class OWAFRSBlockwiseApproximation:
    """
    @brief Immutable exact OWAFRS blockwise approximation outputs.

    @param lower: Lower approximation values after OWA aggregation.
    @param upper: Upper approximation values after OWA aggregation.
    @param boundary: Boundary values computed as upper - lower.
    @param positive_region: Positive-region values, identical to lower by the base model contract.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """
    @brief Return True when config already looks like FRsutils nested config.

    @param config: Candidate config mapping.
    @return: True when fuzzy-rough nested sections are present.
    """
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _as_labels(labels: Any, *, expected_length: int) -> np.ndarray:
    """
    @brief Convert and validate label input for blockwise approximations.

    @param labels: Candidate label vector.
    @param expected_length: Required number of labels.
    @return: One-dimensional NumPy label array.
    @raises ValueError: If labels are not one-dimensional or length-matched.
    """
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array-like vector.")
    if len(labels_array) != expected_length:
        raise ValueError("Length of labels must match the number of samples in the similarity engine.")
    return labels_array


def _as_nested_config(config: Optional[Mapping[str, Any]], *, default_model_type: str = "itfrs") -> Mapping[str, Any]:
    """
    @brief Normalize a flat or nested config into nested FRsutils form.

    @param config: Optional flat or nested config mapping.
    @param default_model_type: Model alias used when config is omitted.
    @return: Nested FRsutils config mapping.
    @raises TypeError: If config is not mapping-like.
    """
    if config is None:
        if default_model_type == "vqrs":
            config = {
                "type": "vqrs",
                "lb_fuzzy_quantifier_name": "linear",
                "lb_fuzzy_quantifier_alpha": 0.1,
                "lb_fuzzy_quantifier_beta": 0.6,
                "ub_fuzzy_quantifier_name": "linear",
                "ub_fuzzy_quantifier_alpha": 0.1,
                "ub_fuzzy_quantifier_beta": 0.6,
            }
        elif default_model_type == "owafrs":
            config = {
                "type": "owafrs",
                "ub_tnorm_name": "minimum",
                "lb_implicator_name": "lukasiewicz",
                "ub_owa_method_name": "linear",
                "lb_owa_method_name": "linear",
            }
        else:
            config = {"type": "itfrs", "ub_tnorm_name": "minimum", "lb_implicator_name": "lukasiewicz"}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")
    return config if _is_nested_frs_config(config) else normalize_flat_config_to_nested(dict(config))


def build_itfrs_components_from_config(config: Optional[Mapping[str, Any]]) -> Tuple[TNorm, Implicator]:
    """
    @brief Build ITFRS upper T-norm and lower implicator from flat or nested config.

    @param config: Optional flat or nested FRsutils config mapping.
    @return: Tuple `(ub_tnorm, lb_implicator)`.
    @raises ValueError: If required component specs cannot be resolved.
    """
    nested = _as_nested_config(config)
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    ub_tnorm = TNorm.create_from_spec(fr_cfg.get("ub_tnorm"))
    lb_implicator = Implicator.create_from_spec(fr_cfg.get("lb_implicator"))

    if ub_tnorm is None:
        ub_tnorm = TNorm.create("minimum")
    if lb_implicator is None:
        lb_implicator = Implicator.create("lukasiewicz")

    return ub_tnorm, lb_implicator


def build_vqrs_components_from_config(config: Optional[Mapping[str, Any]]) -> Tuple[FuzzyQuantifier, FuzzyQuantifier, TNorm]:
    """
    @brief Build VQRS fuzzy quantifiers and fixed minimum T-norm from config.

    Dense VQRS currently uses a minimum T-norm internally. The blockwise path
    mirrors that behavior exactly and resolves only the lower/upper fuzzy
    quantifiers from the public flat or nested FRsutils config.

    @param config: Optional flat or nested FRsutils config mapping.
    @return: Tuple `(lb_fuzzy_quantifier, ub_fuzzy_quantifier, tnorm)`.
    @raises TypeError: If config sections cannot be resolved.
    """
    nested = _as_nested_config(config, default_model_type="vqrs")
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    lb_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("lb_fuzzy_quantifier"))
    ub_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("ub_fuzzy_quantifier"))

    if lb_fuzzy_quantifier is None:
        lb_fuzzy_quantifier = FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)
    if ub_fuzzy_quantifier is None:
        ub_fuzzy_quantifier = FuzzyQuantifier.create("linear", alpha=0.1, beta=0.6)

    tnorm = TNorm.create("minimum")
    return lb_fuzzy_quantifier, ub_fuzzy_quantifier, tnorm


def build_owafrs_components_from_config(
    config: Optional[Mapping[str, Any]],
) -> Tuple[TNorm, Implicator, OWAWeights, OWAWeights]:
    """
    @brief Build OWAFRS T-norm, implicator, and OWA strategies from config.

    @param config: Optional flat or nested FRsutils config mapping.
    @return: Tuple `(ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method)`.
    @raises TypeError: If config sections cannot be resolved.
    """
    nested = _as_nested_config(config, default_model_type="owafrs")
    fr_cfg = nested.get("fr_model", {}) if isinstance(nested, Mapping) else {}
    if not isinstance(fr_cfg, Mapping):
        raise TypeError("nested config section 'fr_model' must be a mapping.")

    ub_tnorm = TNorm.create_from_spec(fr_cfg.get("ub_tnorm"))
    lb_implicator = Implicator.create_from_spec(fr_cfg.get("lb_implicator"))
    ub_owa_method = OWAWeights.create_from_spec(fr_cfg.get("ub_owa_method"))
    lb_owa_method = OWAWeights.create_from_spec(fr_cfg.get("lb_owa_method"))

    if ub_tnorm is None:
        ub_tnorm = TNorm.create("minimum")
    if lb_implicator is None:
        lb_implicator = Implicator.create("lukasiewicz")
    if ub_owa_method is None:
        ub_owa_method = OWAWeights.create("linear")
    if lb_owa_method is None:
        lb_owa_method = OWAWeights.create("linear")

    return ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method


def _diagonal_positions_for_block(row_slice: slice, col_slice: slice) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Return local diagonal positions for overlapping row/column slices.

    @param row_slice: Global row slice for a similarity block.
    @param col_slice: Global column slice for a similarity block.
    @return: Tuple of local row indices and local column indices where global sample ids match.
    """
    row_indices = np.arange(row_slice.start or 0, row_slice.stop or 0)
    col_indices = np.arange(col_slice.start or 0, col_slice.stop or 0)
    common, row_local, col_local = np.intersect1d(row_indices, col_indices, return_indices=True)
    if common.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return row_local.astype(int), col_local.astype(int)


def _finalize_owafrs_row_buffer(
    *,
    row_slice: slice,
    lower_buffer: np.ndarray,
    upper_buffer: np.ndarray,
    lower_weights: np.ndarray,
    upper_weights: np.ndarray,
    lower_out: np.ndarray,
    upper_out: np.ndarray,
) -> None:
    """
    @brief Sort one OWAFRS row buffer and write final lower/upper values.

    Dense OWAFRS sets the diagonal to zero, sorts every row descending, removes
    one zero-valued self-comparison column, and applies model-specific OWA
    weights. This helper mirrors that exact behavior for one row block.

    @param row_slice: Global output row slice represented by the buffers.
    @param lower_buffer: Lower implication values for all columns in the row block.
    @param upper_buffer: Upper T-norm values for all columns in the row block.
    @param lower_weights: Dense-compatible lower OWA weights.
    @param upper_weights: Dense-compatible upper OWA weights.
    @param lower_out: Full lower output array to mutate.
    @param upper_out: Full upper output array to mutate.
    """
    sorted_lower = np.sort(lower_buffer, axis=1)[:, ::-1][:, :-1]
    sorted_upper = np.sort(upper_buffer, axis=1)[:, ::-1][:, :-1]
    lower_out[row_slice] = np.matmul(sorted_lower, lower_weights)
    upper_out[row_slice] = np.matmul(sorted_upper, upper_weights)


def compute_itfrs_blockwise(
    similarity_engine: BaseSimilarityEngine,
    labels: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> ITFRSBlockwiseApproximation:
    """
    @brief Compute exact ITFRS approximations from similarity blocks.

    This function is mathematically equivalent to the dense ITFRS implementation
    but keeps only row-level lower/upper accumulators in memory. It is therefore
    the first exact blockwise approximation path introduced in Phase 2.

    @param similarity_engine: Dense or blockwise SimilarityEngine instance.
    @param labels: Label vector aligned with the engine samples.
    @param config: Flat or nested model configuration used to build ITFRS components.
    @return: ITFRSBlockwiseApproximation value object.
    @raises ValueError: If labels are not aligned with the engine samples.
    """
    if not isinstance(similarity_engine, BaseSimilarityEngine):
        raise TypeError("similarity_engine must be a BaseSimilarityEngine instance.")

    labels_array = _as_labels(labels, expected_length=similarity_engine.n_samples)
    ub_tnorm, lb_implicator = build_itfrs_components_from_config(config)

    n_samples = similarity_engine.n_samples
    lower_acc = np.ones(n_samples, dtype=np.float64)
    upper_acc = np.zeros(n_samples, dtype=np.float64)

    for block in similarity_engine.iter_blocks():
        row_labels = labels_array[block.row_slice]
        col_labels = labels_array[block.col_slice]
        label_mask = (row_labels[:, None] == col_labels[None, :]).astype(float)
        values = np.asarray(block.values, dtype=np.float64)

        implication_vals = lb_implicator(values, label_mask)
        tnorm_vals = ub_tnorm(values, label_mask)

        diagonal_rows, diagonal_cols = _diagonal_positions_for_block(block.row_slice, block.col_slice)
        if diagonal_rows.size:
            implication_vals[diagonal_rows, diagonal_cols] = 1.0
            tnorm_vals[diagonal_rows, diagonal_cols] = 0.0

        if implication_vals.shape[1] > 0:
            lower_acc[block.row_slice] = np.minimum(lower_acc[block.row_slice], np.min(implication_vals, axis=1))
        if tnorm_vals.shape[1] > 0:
            upper_acc[block.row_slice] = np.maximum(upper_acc[block.row_slice], np.max(tnorm_vals, axis=1))

    boundary = upper_acc - lower_acc
    positive_region = lower_acc.copy()
    return ITFRSBlockwiseApproximation(
        lower=lower_acc,
        upper=upper_acc,
        boundary=boundary,
        positive_region=positive_region,
    )


def compute_vqrs_blockwise(
    similarity_engine: BaseSimilarityEngine,
    labels: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> VQRSBlockwiseApproximation:
    """
    @brief Compute exact VQRS approximations from similarity blocks.

    This function is mathematically equivalent to the dense VQRS implementation:
    it accumulates the same numerator `sum(min(S_ij, same_label_ij))` and
    denominator `sum(S_ij) - 1` row by row, then applies the configured lower
    and upper fuzzy quantifiers. It avoids storing the full n x n matrix.

    @param similarity_engine: Dense or blockwise SimilarityEngine instance.
    @param labels: Label vector aligned with the engine samples.
    @param config: Flat or nested model configuration used to build VQRS components.
    @return: VQRSBlockwiseApproximation value object.
    @raises ValueError: If labels are not aligned with the engine samples.
    """
    if not isinstance(similarity_engine, BaseSimilarityEngine):
        raise TypeError("similarity_engine must be a BaseSimilarityEngine instance.")

    labels_array = _as_labels(labels, expected_length=similarity_engine.n_samples)
    lb_fuzzy_quantifier, ub_fuzzy_quantifier, tnorm = build_vqrs_components_from_config(config)

    n_samples = similarity_engine.n_samples
    numerator_acc = np.zeros(n_samples, dtype=np.float64)
    denominator_acc = np.zeros(n_samples, dtype=np.float64)

    for block in similarity_engine.iter_blocks():
        row_labels = labels_array[block.row_slice]
        col_labels = labels_array[block.col_slice]
        label_mask = (row_labels[:, None] == col_labels[None, :]).astype(float)
        values = np.asarray(block.values, dtype=np.float64)

        tnorm_vals = tnorm(values, label_mask)

        diagonal_rows, diagonal_cols = _diagonal_positions_for_block(block.row_slice, block.col_slice)
        if diagonal_rows.size:
            # Dense VQRS excludes self-comparisons from the numerator by forcing
            # the t-norm diagonal to zero, while denominator exclusion is handled
            # once at the end through `sum(S_i*) - 1`.
            tnorm_vals[diagonal_rows, diagonal_cols] = 0.0

        if values.shape[1] > 0:
            numerator_acc[block.row_slice] += np.sum(tnorm_vals, axis=1)
            denominator_acc[block.row_slice] += np.sum(values, axis=1)

    denominator = denominator_acc - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        interim = numerator_acc / denominator

    lower = np.asarray(lb_fuzzy_quantifier(interim))
    upper = np.asarray(ub_fuzzy_quantifier(interim))
    boundary = upper - lower
    positive_region = lower.copy()
    return VQRSBlockwiseApproximation(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
        interim=interim,
    )


def compute_owafrs_blockwise(
    similarity_engine: BaseSimilarityEngine,
    labels: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> OWAFRSBlockwiseApproximation:
    """
    @brief Compute exact OWAFRS approximations from similarity blocks.

    OWAFRS requires row-wise sorting before applying OWA weights, so it cannot
    use only min/max or sum accumulators. This Phase 5 implementation keeps one
    `row_block_size x n_samples` lower/upper buffer at a time, fills it from
    similarity blocks, sorts it exactly like the dense OWAFRS model, writes the
    result rows, and releases the buffer before moving to the next row block.

    @param similarity_engine: Dense or blockwise SimilarityEngine instance.
    @param labels: Label vector aligned with the engine samples.
    @param config: Flat or nested model configuration used to build OWAFRS components.
    @return: OWAFRSBlockwiseApproximation value object.
    @raises ValueError: If labels are not aligned with the engine samples.
    """
    if not isinstance(similarity_engine, BaseSimilarityEngine):
        raise TypeError("similarity_engine must be a BaseSimilarityEngine instance.")

    labels_array = _as_labels(labels, expected_length=similarity_engine.n_samples)
    ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(config)

    n_samples = similarity_engine.n_samples
    lower_weights = lb_owa_method.weights(n_samples - 1, order="asc")
    upper_weights = ub_owa_method.weights(n_samples - 1, order="desc")

    lower = np.zeros(n_samples, dtype=np.result_type(np.float64, lower_weights.dtype))
    upper = np.zeros(n_samples, dtype=np.result_type(np.float64, upper_weights.dtype))

    current_row_slice: Optional[slice] = None
    lower_buffer: Optional[np.ndarray] = None
    upper_buffer: Optional[np.ndarray] = None

    for block in similarity_engine.iter_blocks():
        if current_row_slice != block.row_slice:
            if current_row_slice is not None and lower_buffer is not None and upper_buffer is not None:
                _finalize_owafrs_row_buffer(
                    row_slice=current_row_slice,
                    lower_buffer=lower_buffer,
                    upper_buffer=upper_buffer,
                    lower_weights=lower_weights,
                    upper_weights=upper_weights,
                    lower_out=lower,
                    upper_out=upper,
                )

            current_row_slice = block.row_slice
            n_rows = len(range(block.row_slice.start or 0, block.row_slice.stop or 0))
            lower_buffer = np.zeros((n_rows, n_samples), dtype=np.float64)
            upper_buffer = np.zeros((n_rows, n_samples), dtype=np.float64)

        row_labels = labels_array[block.row_slice]
        col_labels = labels_array[block.col_slice]
        label_mask = (row_labels[:, None] == col_labels[None, :]).astype(float)
        values = np.asarray(block.values, dtype=np.float64)

        lower_values = lb_implicator(values, label_mask)
        upper_values = ub_tnorm(values, label_mask)

        diagonal_rows, diagonal_cols = _diagonal_positions_for_block(block.row_slice, block.col_slice)
        if diagonal_rows.size:
            lower_values[diagonal_rows, diagonal_cols] = 0.0
            upper_values[diagonal_rows, diagonal_cols] = 0.0

        lower_buffer[:, block.col_slice] = lower_values
        upper_buffer[:, block.col_slice] = upper_values

    if current_row_slice is not None and lower_buffer is not None and upper_buffer is not None:
        _finalize_owafrs_row_buffer(
            row_slice=current_row_slice,
            lower_buffer=lower_buffer,
            upper_buffer=upper_buffer,
            lower_weights=lower_weights,
            upper_weights=upper_weights,
            lower_out=lower,
            upper_out=upper,
        )

    boundary = upper - lower
    positive_region = lower.copy()
    return OWAFRSBlockwiseApproximation(
        lower=lower,
        upper=upper,
        boundary=boundary,
        positive_region=positive_region,
    )


__all__ = [
    "ITFRSBlockwiseApproximation",
    "VQRSBlockwiseApproximation",
    "OWAFRSBlockwiseApproximation",
    "build_itfrs_components_from_config",
    "build_vqrs_components_from_config",
    "build_owafrs_components_from_config",
    "compute_itfrs_blockwise",
    "compute_vqrs_blockwise",
    "compute_owafrs_blockwise",
]
