# SPDX-License-Identifier: BSD-3-Clause
"""Exact approximation engines for fuzzy-rough computations.

This module contains dense-compatible blockwise approximation routines. The
ITFRS and VQRS blockwise paths are the scalable backend-aware counterparts to
their dense NumPy reference models and can use backend-resident accumulators
when the similarity engine is configured with an optional CuPy backend. OWAFRS
uses an exact row-buffered blockwise path; optional CuPy support is limited to
similarity-block generation and does not claim GPU-resident OWAFRS aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from FRsutils.core.backends import is_cupy_backend
from FRsutils.core.similarity_engine import BaseSimilarityEngine
from FRsutils.core.models.itfrs_components import build_itfrs_components_from_config
from FRsutils.core.models.vqrs_components import build_vqrs_components_from_config
from FRsutils.core.models.owafrs_components import build_owafrs_components_from_config
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


@dataclass(frozen=True)
class ITFRSBlockwiseApproximation:
    """Immutable exact ITFRS blockwise approximation outputs.
    
    Parameters
    ----------
    lower : object
        Lower approximation values.
    upper : object
        Upper approximation values.
    boundary : object
        Boundary values computed from the public NumPy upper/lower outputs.
    positive_region : object
        Positive-region values copied from the public NumPy lower output.
    execution_backend : object
        Backend that owned the ITFRS accumulators.
    used_gpu_approximation_accumulators : object
        True when CuPy held ITFRS accumulators/reductions.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray
    execution_backend: str = "numpy"
    used_gpu_approximation_accumulators: bool = False


@dataclass(frozen=True)
class VQRSBlockwiseApproximation:
    """Immutable exact VQRS blockwise approximation outputs.
    
    Parameters
    ----------
    lower : object
        Lower approximation values after the lower fuzzy quantifier.
    upper : object
        Upper approximation values after the upper fuzzy quantifier.
    boundary : object
        Boundary values computed from the public NumPy upper/lower outputs.
    positive_region : object
        Positive-region values copied from the public NumPy lower output.
    interim : object
        Raw VQRS ratio values before fuzzy quantifier application.
    execution_backend : object
        Backend that owned the VQRS accumulators.
    used_gpu_approximation_accumulators : object
        True when CuPy held VQRS accumulators/reductions.
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
    """Immutable exact OWAFRS blockwise approximation outputs.

    OWAFRS blockwise execution keeps the OWA sorting/aggregation buffers in
    NumPy to preserve the dense reference contract. CuPy-backed blockwise runs
    may use GPU-resident similarity blocks, but this result object intentionally
    does not claim GPU-resident OWAFRS approximation accumulators.
    
    Parameters
    ----------
    lower : object
        Lower approximation values after OWA aggregation.
    upper : object
        Upper approximation values after OWA aggregation.
    boundary : object
        Boundary values computed as upper - lower.
    positive_region : object
        Positive-region values, identical to lower by the base model contract.
    """

    lower: np.ndarray
    upper: np.ndarray
    boundary: np.ndarray
    positive_region: np.ndarray


def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when config already looks like FRsutils nested config.
        
        Parameters
        ----------
        config : Mapping[str, Any]
            Candidate config mapping.
        
        Returns
        -------
        bool
            True when fuzzy-rough nested sections are present.
        
    """
    return isinstance(config.get("fr_model"), Mapping) or isinstance(config.get("similarity"), Mapping)


def _as_labels(labels: Any, *, expected_length: int) -> np.ndarray:
    """Convert and validate label input for blockwise approximations.
        
        Parameters
        ----------
        labels : Any
            Candidate label vector.
        expected_length : int
            Required number of labels.
        
        Returns
        -------
        np.ndarray
            One-dimensional NumPy label array.
        
        Raises
        ------
        ValueError
            If labels are not one-dimensional or length-matched.
        
    """
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array-like vector.")
    if len(labels_array) != expected_length:
        raise ValueError("Length of labels must match the number of samples in the similarity engine.")
    return labels_array


def _as_nested_config(config: Optional[Mapping[str, Any]], *, default_model_type: str = "itfrs") -> Mapping[str, Any]:
    """Normalize a flat or nested config into nested FRsutils form.
        
        Parameters
        ----------
        config : Optional[Mapping[str, Any]]
            Optional flat or nested config mapping.
        default_model_type : str
            Model alias used when config is omitted.
        
        Returns
        -------
        Mapping[str, Any]
            Nested FRsutils config mapping.
        
        Raises
        ------
        TypeError
            If config is not mapping-like.
        
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


def _backend_index_array(indices: np.ndarray, *, xp: Any):
    """Convert NumPy integer indices to a backend-compatible index array.
        
        Parameters
        ----------
        indices : np.ndarray
            One-dimensional NumPy integer index array.
        xp : Any
            Array namespace used by the active backend.
        
        Returns
        -------
        object
            Backend-compatible index array.
        
    """
    return indices if xp is np else xp.asarray(indices, dtype=int)


def _set_block_diagonal_values(values: Any, row_indices: np.ndarray, col_indices: np.ndarray, *, xp: Any, lower_value: float, upper_value: Optional[float] = None):
    """Set local diagonal entries for one or two backend arrays.
        
        Parameters
        ----------
        values : Any
            First backend array to mutate.
        row_indices : np.ndarray
            Local row positions as NumPy integers.
        col_indices : np.ndarray
            Local column positions as NumPy integers.
        xp : Any
            Array namespace used by the active backend.
        lower_value : float
            Value assigned to `values`.
        upper_value : Optional[float]
            Reserved for readability at call sites; ignored here.
        
        Returns
        -------
        object
            None.
        
    """
    if row_indices.size == 0:
        return
    row_idx = _backend_index_array(row_indices, xp=xp)
    col_idx = _backend_index_array(col_indices, xp=xp)
    values[row_idx, col_idx] = lower_value


def _diagonal_positions_for_block(row_slice: slice, col_slice: slice) -> Tuple[np.ndarray, np.ndarray]:
    """Return local diagonal positions for overlapping row/column slices.
        
        Parameters
        ----------
        row_slice : slice
            Global row slice for a similarity block.
        col_slice : slice
            Global column slice for a similarity block.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of local row indices and local column indices where global sample ids match.
        
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
    """Sort one OWAFRS row buffer and write final lower/upper values.
        
        Dense OWAFRS sets the diagonal to zero, sorts every row descending, removes
        one zero-valued self-comparison column, and applies model-specific OWA
        weights. This helper mirrors that exact behavior for one row block.
        
        Parameters
        ----------
        row_slice : slice
            Global output row slice represented by the buffers.
        lower_buffer : np.ndarray
            Lower implication values for all columns in the row block.
        upper_buffer : np.ndarray
            Upper T-norm values for all columns in the row block.
        lower_weights : np.ndarray
            Dense-compatible lower OWA weights.
        upper_weights : np.ndarray
            Dense-compatible upper OWA weights.
        lower_out : np.ndarray
            Full lower output array to mutate.
        upper_out : np.ndarray
            Full upper output array to mutate.
        
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

        This function is mathematically equivalent to the dense ITFRS reference
        model but keeps only row-level lower and upper accumulators in memory.
        When the similarity engine resolves backend="cupy" and exposes
        ``iter_backend_blocks()``, similarity blocks, implicator/T-norm values,
        and min/max accumulators remain backend-resident until the final NumPy
        conversion at the public result boundary.

        Parameters
        ----------
        similarity_engine : BaseSimilarityEngine
            Dense or blockwise SimilarityEngine instance.
        labels : Any
            Label vector aligned with the engine samples.
        config : Optional[Mapping[str, Any]]
            Flat or nested model configuration used to build ITFRS components.
        
        Returns
        -------
        ITFRSBlockwiseApproximation
            ITFRSBlockwiseApproximation value object.
        
        Raises
        ------
        ValueError
            If labels are not aligned with the engine samples.
        
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
        
        This function is the public scalable counterpart to the dense VQRS
        reference model: it accumulates the same numerator
        `sum(min(S_ij, same_label_ij))` and denominator `sum(S_ij) - 1` row by
        row, then applies the configured lower and upper fuzzy quantifiers. When
        the similarity engine resolves
        backend='cupy' and exposes iter_backend_blocks(), this Phase 4 path keeps
        similarity blocks, minimum T-norm values, numerator/denominator sums, and
        fuzzy-quantifier application backend-resident until the final public NumPy
        conversion of lower, upper, and interim outputs. Boundary and positive
        region outputs are derived from those public NumPy arrays. The function
        avoids storing the full n x n matrix in either CPU or GPU memory.
        
        Parameters
        ----------
        similarity_engine : BaseSimilarityEngine
            Dense or blockwise SimilarityEngine instance.
        labels : Any
            Label vector aligned with the engine samples.
        config : Optional[Mapping[str, Any]]
            Flat or nested model configuration used to build VQRS components.
        
        Returns
        -------
        VQRSBlockwiseApproximation
            VQRSBlockwiseApproximation value object.
        
        Raises
        ------
        ValueError
            If labels are not aligned with the engine samples.
        
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
        if diagonal_rows.size:
            # Dense VQRS excludes self-comparisons from the numerator by forcing
            # the T-norm diagonal to zero, while denominator exclusion is handled
            # once at the end through `sum(S_i*) - 1`.
            row_idx = _backend_index_array(diagonal_rows, xp=xp)
            col_idx = _backend_index_array(diagonal_cols, xp=xp)
            tnorm_vals[row_idx, col_idx] = 0.0

        if values.shape[1] > 0:
            numerator_acc[block.row_slice] = numerator_acc[block.row_slice] + xp.sum(tnorm_vals, axis=1)
            denominator_acc[block.row_slice] = denominator_acc[block.row_slice] + xp.sum(values, axis=1)

    denominator = denominator_acc - 1.0
    errstate = getattr(xp, "errstate", np.errstate)
    with errstate(divide="ignore", invalid="ignore"):
        interim_acc = numerator_acc / denominator

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
        
        OWAFRS requires row-wise sorting before applying OWA weights, so it cannot
        use only min/max or sum accumulators like ITFRS or VQRS. This exact
        implementation keeps one `row_block_size x n_samples` lower/upper NumPy
        buffer at a time, fills it from similarity blocks, sorts it exactly like
        the dense OWAFRS model, writes the result rows, and releases the buffer
        before moving to the next row block. Optional CuPy support remains limited
        to similarity-block generation; OWAFRS aggregation is intentionally
        conservative and NumPy-compatible.
        
        Parameters
        ----------
        similarity_engine : BaseSimilarityEngine
            Dense or blockwise SimilarityEngine instance.
        labels : Any
            Label vector aligned with the engine samples.
        config : Optional[Mapping[str, Any]]
            Flat or nested model configuration used to build OWAFRS components.
        
        Returns
        -------
        OWAFRSBlockwiseApproximation
            OWAFRSBlockwiseApproximation value object.
        
        Raises
        ------
        ValueError
            If labels are not aligned with the engine samples.
        
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
