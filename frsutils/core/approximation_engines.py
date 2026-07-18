# SPDX-License-Identifier: BSD-3-Clause
"""Exact dense-compatible blockwise fuzzy-rough approximation engines.

ITFRS and VQRS can keep supported accumulators backend-resident during
blockwise CuPy execution. OWAFRS uses exact NumPy row buffers because its OWA
aggregation requires sorting all non-self evidence for each row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from frsutils.core.backends import is_cupy_backend
from frsutils.core.models.itfrs_components import build_itfrs_components_from_config
from frsutils.core.models.owafrs_components import build_owafrs_components_from_config
from frsutils.core.models.vqrs_components import (
    build_default_vqrs_flat_config,
    build_vqrs_components_from_config,
)
from frsutils.core.models.vqrs_math import compute_vqrs_interim_ratio
from frsutils.core.similarity_engine import BaseSimilarityEngine
from frsutils.utils.init_helpers import normalize_flat_config_to_nested


@dataclass(frozen=True)
class ITFRSBlockwiseApproximation:
    """Immutable exact ITFRS blockwise outputs.

    Attributes
    ----------
    lower, upper : ndarray of shape (n_samples,)
        Lower and upper approximation values.
    boundary : ndarray of shape (n_samples,)
        Signed values computed as ``upper - lower``.
    positive_region : ndarray of shape (n_samples,)
        Positive-region values under the current lower-score contract.
    execution_backend : str
        Backend that owned the ITFRS accumulators.
    used_gpu_approximation_accumulators : bool
        Whether CuPy owned the approximation accumulators and reductions.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    execution_backend: str = "numpy"
    used_gpu_approximation_accumulators: bool = False


@dataclass(frozen=True)
class VQRSBlockwiseApproximation:
    """Immutable exact VQRS blockwise outputs.

    Attributes
    ----------
    lower, upper : ndarray of shape (n_samples,)
        Quantified lower and upper approximation values.
    boundary : ndarray of shape (n_samples,)
        Signed values computed as ``upper - lower``.
    positive_region : ndarray of shape (n_samples,)
        Positive-region values under the current lower-score contract.
    interim : ndarray of shape (n_samples,)
        Non-self support-to-similarity ratios before quantification.
    execution_backend : str
        Backend that owned the VQRS accumulators.
    used_gpu_approximation_accumulators : bool
        Whether CuPy owned the approximation accumulators and reductions.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    interim: np.ndarray
    execution_backend: str = "numpy"
    used_gpu_approximation_accumulators: bool = False


@dataclass(frozen=True)
class OWAFRSBlockwiseApproximation:
    """Immutable exact OWAFRS row-buffered outputs.

    Attributes
    ----------
    lower, upper : ndarray of shape (n_samples,)
        OWA-aggregated lower and upper approximation values.
    boundary : ndarray of shape (n_samples,)
        Signed values computed as ``upper - lower``.
    positive_region : ndarray of shape (n_samples,)
        Positive-region values under the current lower-score contract.

    Notes
    -----
    CuPy-backed runs may generate similarity blocks on the GPU, but OWAFRS
    sorting and aggregation remain NumPy-resident.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return whether a mapping already contains nested FRsutils sections."""
    return isinstance(config.get("fr_model"), Mapping) or isinstance(
        config.get("similarity"), Mapping
    )


def _as_labels(labels: Any, *, expected_length: int) -> np.ndarray:
    """Return a validated one-dimensional NumPy label array.

    Parameters
    ----------
    labels : array-like
        Candidate label vector.
    expected_length : int
        Required number of labels and samples.

    Returns
    -------
    ndarray of shape (expected_length,)
        Validated label vector.

    Raises
    ------
    ValueError
        If fewer than two samples are available or labels are missing,
        non-one-dimensional, or length-mismatched.
    """
    if expected_length < 2:
        raise ValueError(
            "Fuzzy-rough approximation engines require at least two samples."
        )
    if labels is None:
        raise ValueError("labels must be provided as a 1D array-like vector.")

    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array-like vector.")
    if len(labels_array) != expected_length:
        raise ValueError(
            "Length of labels must match the number of samples in the "
            "similarity engine."
        )
    return labels_array


def _as_nested_config(
    config: Optional[Mapping[str, Any]],
    *,
    default_model_type: str = "itfrs",
) -> Mapping[str, Any]:
    """Normalize flat or absent configuration into nested FRsutils form.

    Parameters
    ----------
    config : mapping or None
        Optional flat or nested configuration.
    default_model_type : {"itfrs", "vqrs", "owafrs"}, default="itfrs"
        Model whose defaults are used when ``config`` is omitted.

    Returns
    -------
    mapping
        Nested FRsutils configuration.

    Raises
    ------
    TypeError
        If ``config`` is not mapping-like.
    """
    if config is None:
        if default_model_type == "vqrs":
            config = {"type": "vqrs", **build_default_vqrs_flat_config()}
        elif default_model_type == "owafrs":
            config = {
                "type": "owafrs",
                "ub_tnorm_name": "minimum",
                "lb_implicator_name": "lukasiewicz",
                "ub_owa_method_name": "linear",
                "lb_owa_method_name": "linear",
            }
        else:
            config = {
                "type": "itfrs",
                "ub_tnorm_name": "minimum",
                "lb_implicator_name": "lukasiewicz",
            }
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")
    if _is_nested_frs_config(config):
        return config
    return normalize_flat_config_to_nested(dict(config))


def _backend_index_array(indices: np.ndarray, *, xp: Any):
    """Convert NumPy integer indices to the active backend namespace."""
    return indices if xp is np else xp.asarray(indices, dtype=int)


def _diagonal_positions_for_block(
    row_slice: slice,
    col_slice: slice,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return local positions where block row and column sample ids match.

    Parameters
    ----------
    row_slice, col_slice : slice
        Global row and column slices for one similarity block.

    Returns
    -------
    row_positions, column_positions : tuple of ndarray
        Local integer indices for diagonal self-comparison cells.
    """
    row_indices = np.arange(row_slice.start or 0, row_slice.stop or 0)
    col_indices = np.arange(col_slice.start or 0, col_slice.stop or 0)
    common, row_local, col_local = np.intersect1d(
        row_indices,
        col_indices,
        return_indices=True,
    )
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
    """Sort one OWAFRS row buffer and write its exact outputs.

    Parameters
    ----------
    row_slice : slice
        Global output rows represented by the buffers.
    lower_buffer, upper_buffer : ndarray
        Complete lower and upper evidence rows before OWA aggregation.
    lower_weights, upper_weights : ndarray of shape (n_samples - 1,)
        Dense-compatible OWA weight vectors.
    lower_out, upper_out : ndarray of shape (n_samples,)
        Full output arrays updated in place.
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
    """Compute exact ITFRS approximations from similarity blocks.

    Parameters
    ----------
    similarity_engine : BaseSimilarityEngine
        Dense or blockwise engine whose blocks follow the
        ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Labels aligned with the engine samples.
    config : mapping or None, default=None
        Flat or nested ITFRS component configuration.

    Returns
    -------
    ITFRSBlockwiseApproximation
        NumPy public outputs and backend-execution metadata.

    Notes
    -----
    The result is numerically equivalent to the dense ITFRS reference model.
    CuPy-backed engines keep supported similarity blocks, component values, and
    min/max accumulators on the device until the final NumPy conversion.
    """
    if not isinstance(similarity_engine, BaseSimilarityEngine):
        raise TypeError("similarity_engine must be a BaseSimilarityEngine instance.")

    labels_array = _as_labels(labels, expected_length=similarity_engine.n_samples)
    ub_tnorm, lb_implicator = build_itfrs_components_from_config(config)

    backend = getattr(similarity_engine, "backend", None)
    xp = getattr(backend, "xp", np)
    backend_name = getattr(backend, "name", "numpy")
    use_gpu_accumulators = bool(backend is not None and is_cupy_backend(backend))

    n_samples = similarity_engine.n_samples
    lower_acc = xp.ones(n_samples, dtype=np.float64)
    upper_acc = xp.zeros(n_samples, dtype=np.float64)

    block_iterator = (
        similarity_engine.iter_backend_blocks()
        if use_gpu_accumulators and hasattr(similarity_engine, "iter_backend_blocks")
        else similarity_engine.iter_blocks()
    )

    for block in block_iterator:
        row_labels = labels_array[block.row_slice]
        col_labels = labels_array[block.col_slice]
        label_mask_np = (row_labels[:, None] == col_labels[None, :]).astype(np.float64)
        values = xp.asarray(block.values, dtype=np.float64)
        label_mask = xp.asarray(label_mask_np, dtype=np.float64)

        implication_vals = lb_implicator.compute_backend(values, label_mask, xp=xp, validate_inputs=False)
        tnorm_vals = ub_tnorm.compute_backend(values, label_mask, xp=xp)

        diagonal_rows, diagonal_cols = _diagonal_positions_for_block(block.row_slice, block.col_slice)
        if diagonal_rows.size:
            row_idx = _backend_index_array(diagonal_rows, xp=xp)
            col_idx = _backend_index_array(diagonal_cols, xp=xp)
            implication_vals[row_idx, col_idx] = 1.0
            tnorm_vals[row_idx, col_idx] = 0.0

        if implication_vals.shape[1] > 0:
            lower_acc[block.row_slice] = xp.minimum(lower_acc[block.row_slice], xp.min(implication_vals, axis=1))
        if tnorm_vals.shape[1] > 0:
            upper_acc[block.row_slice] = xp.maximum(upper_acc[block.row_slice], xp.max(tnorm_vals, axis=1))

    if backend is not None:
        lower_out = backend.to_numpy(lower_acc)
        upper_out = backend.to_numpy(upper_acc)
    else:
        lower_out = np.asarray(lower_acc)
        upper_out = np.asarray(upper_acc)

    boundary_out = upper_out - lower_out
    positive_region_out = lower_out.copy()

    return ITFRSBlockwiseApproximation(
        lower=lower_out,
        upper=upper_out,
        boundary=boundary_out,
        positive_region=positive_region_out,
        execution_backend=backend_name,
        used_gpu_approximation_accumulators=use_gpu_accumulators,
    )


def compute_vqrs_blockwise(
    similarity_engine: BaseSimilarityEngine,
    labels: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> VQRSBlockwiseApproximation:
    """Compute exact VQRS approximations from similarity blocks.

    Parameters
    ----------
    similarity_engine : BaseSimilarityEngine
        Dense or blockwise engine whose blocks follow the
        ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Labels aligned with the engine samples.
    config : mapping or None, default=None
        Flat or nested VQRS component configuration.

    Returns
    -------
    VQRSBlockwiseApproximation
        Quantified NumPy outputs, interim ratios, and backend metadata.

    Notes
    -----
    The engine accumulates non-self numerator and denominator masses exactly.
    CuPy-backed execution may keep similarity blocks, T-norm values, sums,
    interim ratios, and quantifier application on the device until final NumPy
    conversion.
    """
    if not isinstance(similarity_engine, BaseSimilarityEngine):
        raise TypeError("similarity_engine must be a BaseSimilarityEngine instance.")

    labels_array = _as_labels(labels, expected_length=similarity_engine.n_samples)
    lb_fuzzy_quantifier, ub_fuzzy_quantifier, tnorm = build_vqrs_components_from_config(config)

    backend = getattr(similarity_engine, "backend", None)
    xp = getattr(backend, "xp", np)
    backend_name = getattr(backend, "name", "numpy")
    use_gpu_accumulators = bool(backend is not None and is_cupy_backend(backend))

    n_samples = similarity_engine.n_samples
    numerator_acc = xp.zeros(n_samples, dtype=np.float64)
    denominator_acc = xp.zeros(n_samples, dtype=np.float64)

    block_iterator = (
        similarity_engine.iter_backend_blocks()
        if use_gpu_accumulators and hasattr(similarity_engine, "iter_backend_blocks")
        else similarity_engine.iter_blocks()
    )

    for block in block_iterator:
        row_labels = labels_array[block.row_slice]
        col_labels = labels_array[block.col_slice]
        label_mask_np = (row_labels[:, None] == col_labels[None, :]).astype(np.float64)
        values = xp.asarray(block.values, dtype=np.float64)
        label_mask = xp.asarray(label_mask_np, dtype=np.float64)

        tnorm_vals = tnorm.compute_backend(values, label_mask, xp=xp)

        diagonal_rows, diagonal_cols = _diagonal_positions_for_block(block.row_slice, block.col_slice)
        denominator_values = values.copy()
        if diagonal_rows.size:
            # Exclude self-comparisons from both numerator and denominator while
            # preserving the original similarity block. This avoids assuming a
            # unit diagonal for precomputed similarity engines.
            row_idx = _backend_index_array(diagonal_rows, xp=xp)
            col_idx = _backend_index_array(diagonal_cols, xp=xp)
            tnorm_vals[row_idx, col_idx] = 0.0
            denominator_values[row_idx, col_idx] = 0.0

        block_denominator = (
            xp.sum(denominator_values, axis=1)
            if values.shape[1] > 0
            else xp.zeros(values.shape[0], dtype=np.float64)
        )
        if values.shape[1] > 0:
            numerator_acc[block.row_slice] = (
                numerator_acc[block.row_slice] + xp.sum(tnorm_vals, axis=1)
            )
            denominator_acc[block.row_slice] = (
                denominator_acc[block.row_slice] + block_denominator
            )

    interim_acc = compute_vqrs_interim_ratio(
        numerator_acc,
        denominator_acc,
        xp=xp,
    )

    lower_acc = lb_fuzzy_quantifier.compute_backend(interim_acc, xp=xp, validate_inputs=lb_fuzzy_quantifier.validate_inputs)
    upper_acc = ub_fuzzy_quantifier.compute_backend(interim_acc, xp=xp, validate_inputs=ub_fuzzy_quantifier.validate_inputs)
    if backend is not None:
        lower_out = backend.to_numpy(lower_acc)
        upper_out = backend.to_numpy(upper_acc)
        interim_out = backend.to_numpy(interim_acc)
    else:
        lower_out = np.asarray(lower_acc)
        upper_out = np.asarray(upper_acc)
        interim_out = np.asarray(interim_acc)

    boundary_out = upper_out - lower_out
    positive_region_out = lower_out.copy()

    return VQRSBlockwiseApproximation(
        lower=lower_out,
        upper=upper_out,
        boundary=boundary_out,
        positive_region=positive_region_out,
        interim=interim_out,
        execution_backend=backend_name,
        used_gpu_approximation_accumulators=use_gpu_accumulators,
    )


def compute_owafrs_blockwise(
    similarity_engine: BaseSimilarityEngine,
    labels: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> OWAFRSBlockwiseApproximation:
    """Compute exact OWAFRS approximations from similarity blocks.

    Parameters
    ----------
    similarity_engine : BaseSimilarityEngine
        Dense or blockwise engine whose blocks follow the
        ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Labels aligned with the engine samples.
    config : mapping or None, default=None
        Flat or nested OWAFRS component configuration.

    Returns
    -------
    OWAFRSBlockwiseApproximation
        Exact NumPy lower, upper, signed-boundary, and positive-region outputs.

    Notes
    -----
    OWAFRS requires sorting all non-self evidence for each row. The exact
    blockwise path therefore uses one ``row_block_size x n_samples`` NumPy buffer
    at a time. Optional CuPy support is limited to upstream similarity-block
    generation and does not make OWAFRS aggregation GPU-resident.
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
