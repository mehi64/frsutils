"""
@file similarity_engine.py
@brief Similarity-engine abstraction for dense, blockwise, and optional GPU block computation.

This module adds a non-invasive execution layer above the existing similarity
functions. It does not change the current public dense similarity behavior;
instead, it provides reusable engine objects that future approximation code can
consume block by block.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# SimilarityBlock                      Immutable row/column block plus values
# BaseSimilarityEngine                 Shared validation and component resolution
# DenseSimilarityEngine                Compatibility wrapper around dense matrix building
# BlockwiseSimilarityEngine            Exact block iterator for future streaming models
# calculate_similarity_block           Compute one pairwise block from two feature matrices
# iter_backend_blocks                  Yield backend-resident blocks for GPU-aware accumulators
# build_similarity_engine              Factory for dense/blockwise engine construction
# backend='cupy'                       Optional GPU acceleration for block similarity/ITFRS block execution

# ✅ Design Patterns & Clean Code Notes
# - Strategy Pattern: dense and blockwise engines share a common public contract
# - Factory Method: build_similarity_engine resolves engine aliases
# - Adapter Pattern: converts flat/nested config into concrete similarity components
# - Conservative Refactor: current dense API remains untouched and regression-tested
# - Optional Dependency Boundary: CuPy is imported only when backend='cupy' is requested
# - DRY: block computation delegates backend formulas to Similarity/TNorm components
# - GPU Residency Boundary: public iter_blocks() returns NumPy, while iter_backend_blocks() may keep CuPy values resident
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from FRsutils.core.similarity_engine import build_similarity_engine
#
# engine = build_similarity_engine(X, engine="blockwise", similarity="linear")
# for block in engine.iter_blocks():
#     print(block.row_slice, block.col_slice, block.values.shape)
#
# dense = engine.to_dense()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from FRsutils.core.backends import ArrayBackend, build_array_backend, is_cupy_backend
from FRsutils.core.similarities import Similarity, build_similarity_matrix
from FRsutils.core.tnorms import TNorm
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested


@dataclass(frozen=True)
class SimilarityBlock:
    """
    @brief Immutable pairwise similarity block produced by a similarity engine.

    @param row_slice: Row sample slice represented by this block.
    @param col_slice: Column sample slice represented by this block.
    @param values: Similarity values with shape `(len(row_slice), len(col_slice))`.
    @param values_backend: Name of the array backend that owns `values`.
    @param values_are_backend_resident: True when `values` may be a non-NumPy backend array.
    """

    row_slice: slice
    col_slice: slice
    values: Any
    values_backend: str = "numpy"
    values_are_backend_resident: bool = False



def _as_2d_feature_matrix(X: Any) -> np.ndarray:
    """
    @brief Convert and validate feature matrix input for similarity engines.

    @param X: Candidate feature matrix.
    @return: Two-dimensional NumPy array.
    @raises ValueError: If X is missing or not two-dimensional.
    """
    if X is None:
        raise ValueError("X must be provided when building a similarity engine.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like feature matrix.")
    return X_array



def _validate_block_size(block_size: int) -> int:
    """
    @brief Validate and normalize a block size.

    @param block_size: Candidate positive integer block size.
    @return: Normalized block size.
    @raises TypeError: If block_size is not an integer.
    @raises ValueError: If block_size is less than one.
    """
    if not isinstance(block_size, int):
        raise TypeError("block_size must be an integer.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    return block_size



def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """
    @brief Return True when config already looks like a nested FRsutils config.

    @param config: Candidate configuration mapping.
    @return: True if nested similarity sections are present.
    """
    return isinstance(config.get("similarity"), Mapping) or isinstance(config.get("similarity_tnorm"), Mapping)



def _prepare_similarity_config(
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    @brief Prepare an effective flat or nested similarity config.

    @param config: Optional flat or nested mapping.
    @param flat_config: Additional flat keyword parameters.
    @return: Defensive configuration dictionary.
    @raises TypeError: If config is not mapping-like.
    @raises ValueError: If nested config is mixed with extra flat kwargs.
    """
    if config is not None and not isinstance(config, Mapping):
        raise TypeError("config must be a mapping when provided.")

    if config is not None and _is_nested_frs_config(config):
        if flat_config:
            raise ValueError("Do not mix nested config with extra flat keyword parameters.")
        return dict(config)

    effective = dict(config or {})
    effective.update(dict(flat_config))
    return effective



def build_similarity_components(
    config: Optional[Mapping[str, Any]] = None,
    **flat_config: Any,
) -> Tuple[Similarity, TNorm, Dict[str, Any]]:
    """
    @brief Build similarity and T-norm components from flat or nested config.

    This mirrors `FRsutils.core.similarities.build_similarity_matrix` component
    resolution so Phase 1 engines remain equivalent to the existing dense path.

    @param config: Optional flat or nested configuration mapping.
    @param flat_config: Additional flat config values.
    @return: Tuple of `(similarity_func, tnorm_func, effective_config)`.
    """
    effective_config = _prepare_similarity_config(config, flat_config)
    nested = effective_config if _is_nested_frs_config(effective_config) else normalize_flat_config_to_nested(effective_config)

    sim_cfg = nested.get("similarity", {}) if isinstance(nested, Mapping) else {}
    tnorm_cfg = nested.get("similarity_tnorm", {}) if isinstance(nested, Mapping) else {}

    similarity_type = sim_cfg.get("name") or effective_config.get("similarity") or "gaussian"
    similarity_params = dict(sim_cfg.get("params") if isinstance(sim_cfg.get("params"), dict) else {})
    if str(similarity_type).lower() in {"gaussian", "gauss"} and "sigma" in effective_config and "sigma" not in similarity_params:
        # Backward-compatible legacy flat alias used by older tests/examples.
        similarity_params["sigma"] = effective_config["sigma"]

    tnorm_type = tnorm_cfg.get("name") or effective_config.get("similarity_tnorm") or "minimum"
    tnorm_params = tnorm_cfg.get("params") if isinstance(tnorm_cfg.get("params"), dict) else {}

    similarity_func = Similarity.create(similarity_type, **similarity_params)
    tnorm_func = TNorm.create(tnorm_type, **tnorm_params)
    return similarity_func, tnorm_func, dict(effective_config)



def _registered_component_name(registry_cls: Any, component: Any) -> str:
    """
    @brief Return the primary registered alias for a pluggable component.

    @param registry_cls: Registry base class such as Similarity or TNorm.
    @param component: Concrete component instance.
    @return: Registered alias or the component's fallback `name` property.
    """
    try:
        return registry_cls.get_registered_name(component)
    except Exception:
        return str(getattr(component, "name", component.__class__.__name__)).lower()


def _compute_similarity_from_diff(diff: Any, similarity_func: Similarity, backend: ArrayBackend):
    """
    @brief Compute feature-level similarity on a backend array.

    Phase 1 moved backend-specific formulas into the Similarity components, so
    this helper now acts only as a small engine-to-component adapter.

    @param diff: Pairwise backend-array difference matrix.
    @param similarity_func: Built similarity component.
    @param backend: Resolved array backend.
    @return: Backend-array similarity matrix.
    @raises NotImplementedError: If a similarity has no backend formula yet.
    """
    try:
        return similarity_func.compute_backend(diff, xp=backend.xp)
    except NotImplementedError as exc:
        similarity_name = _registered_component_name(Similarity, similarity_func)
        raise NotImplementedError(
            f"backend='{backend.name}' does not yet support similarity='{similarity_name}'. "
            "Use backend='numpy' or add compute_backend(...) to this similarity."
        ) from exc


def _apply_tnorm_backend(a: Any, b: Any, tnorm: TNorm, backend: ArrayBackend):
    """
    @brief Apply a T-norm component on backend arrays.

    Phase 1 moved backend-specific T-norm formulas into the TNorm components, so
    this helper now acts only as a small engine-to-component adapter.

    @param a: First backend array.
    @param b: Second backend array.
    @param tnorm: Built T-norm component.
    @param backend: Resolved array backend.
    @return: Backend-array T-norm result.
    @raises NotImplementedError: If a T-norm has no backend formula yet.
    """
    try:
        return tnorm.compute_backend(a, b, xp=backend.xp)
    except NotImplementedError as exc:
        tnorm_name = _registered_component_name(TNorm, tnorm)
        raise NotImplementedError(
            f"backend='{backend.name}' does not yet support similarity_tnorm='{tnorm_name}'. "
            "Use backend='numpy' or add compute_backend(...) to this T-norm."
        ) from exc


def calculate_similarity_block(
    X_rows: Any,
    X_cols: Any,
    similarity_func: Similarity,
    tnorm,
    *,
    backend: Optional[ArrayBackend] = None,
    return_backend_array: bool = False,
) -> Any:
    """
    @brief Compute an exact pairwise similarity block between two feature matrices.

    @param X_rows: Row-side feature matrix with shape `(n_rows, n_features)`.
    @param X_cols: Column-side feature matrix with shape `(n_cols, n_features)`.
    @param similarity_func: Built similarity component.
    @param tnorm: Built binary T-norm component/callable.
    @param backend: Optional ArrayBackend. NumPy is used when omitted.
    @param return_backend_array: If True, keep CuPy values on GPU instead of converting to NumPy.
    @return: Similarity block with shape `(n_rows, n_cols)` as NumPy or backend array.
    @raises ValueError: If feature dimensions do not match.
    """
    X_rows_array = _as_2d_feature_matrix(X_rows)
    X_cols_array = _as_2d_feature_matrix(X_cols)

    if X_rows_array.shape[1] != X_cols_array.shape[1]:
        raise ValueError("X_rows and X_cols must have the same number of features.")

    n_rows, n_features = X_rows_array.shape
    n_cols = X_cols_array.shape[0]
    if n_rows == 0 or n_cols == 0:
        return np.zeros((n_rows, n_cols), dtype=np.float64)

    effective_backend = backend or build_array_backend("numpy")

    if not is_cupy_backend(effective_backend):
        sim_block = np.ones((n_rows, n_cols), dtype=np.float64)
        for feature_idx in range(n_features):
            row_col = X_rows_array[:, feature_idx].reshape(-1, 1)
            col_row = X_cols_array[:, feature_idx].reshape(1, -1)
            feature_sim = similarity_func(row_col, col_row)
            sim_block = tnorm(sim_block, feature_sim)
        return sim_block

    X_rows_backend = effective_backend.asarray(X_rows_array, dtype=np.float64)
    X_cols_backend = effective_backend.asarray(X_cols_array, dtype=np.float64)
    sim_block_backend = effective_backend.ones((n_rows, n_cols), dtype=np.float64)

    for feature_idx in range(n_features):
        row_col = X_rows_backend[:, feature_idx].reshape(-1, 1)
        col_row = X_cols_backend[:, feature_idx].reshape(1, -1)
        feature_sim = _compute_similarity_from_diff(row_col - col_row, similarity_func, effective_backend)
        sim_block_backend = _apply_tnorm_backend(sim_block_backend, feature_sim, tnorm, effective_backend)

    if return_backend_array:
        return sim_block_backend
    return effective_backend.to_numpy(sim_block_backend)


class BaseSimilarityEngine:
    """
    @brief Base class shared by concrete similarity-engine strategies.

    @param X: Feature matrix used to compute pairwise similarities.
    @param config: Optional flat or nested similarity configuration.
    @param backend: Array backend alias: "numpy", "auto", or explicit "cupy".
    @param flat_config: Additional flat sklearn-style config values.
    """

    engine_name = "base"

    def __init__(
        self,
        X: Any,
        *,
        config: Optional[Mapping[str, Any]] = None,
        backend: str = "numpy",
        **flat_config: Any,
    ) -> None:
        self.X = _as_2d_feature_matrix(X)
        self.backend: ArrayBackend = build_array_backend(backend)
        self.similarity_func, self.tnorm_func, self.config = build_similarity_components(config, **flat_config)

    @property
    def n_samples(self) -> int:
        """
        @brief Number of samples in the engine feature matrix.

        @return: Number of rows in X.
        """
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """
        @brief Number of features in the engine feature matrix.

        @return: Number of columns in X.
        """
        return self.X.shape[1]

    def iter_blocks(self) -> Iterator[SimilarityBlock]:
        """
        @brief Yield exact similarity blocks as NumPy arrays.

        @return: Iterator over NumPy-backed SimilarityBlock values.
        """
        raise NotImplementedError("Concrete similarity engines must implement iter_blocks().")

    def iter_backend_blocks(self) -> Iterator[SimilarityBlock]:
        """
        @brief Yield exact similarity blocks in the engine backend when supported.

        The default implementation preserves compatibility by delegating to
        iter_blocks(), which yields NumPy-backed values. Blockwise engines can
        override this to keep CuPy block values resident for GPU-aware
        approximation accumulators.

        @return: Iterator over SimilarityBlock values.
        """
        yield from self.iter_blocks()

    def to_dense(self) -> np.ndarray:
        """
        @brief Materialize this engine as a dense pairwise similarity matrix.

        @return: Dense `(n_samples, n_samples)` similarity matrix.
        """
        dense = np.zeros((self.n_samples, self.n_samples), dtype=np.float64)
        for block in self.iter_blocks():
            dense[block.row_slice, block.col_slice] = block.values
        if self.n_samples:
            np.fill_diagonal(dense, 1.0)
        return dense


class DenseSimilarityEngine(BaseSimilarityEngine):
    """
    @brief Compatibility engine that materializes the existing dense matrix path.

    DenseSimilarityEngine intentionally delegates `to_dense()` to the current
    `build_similarity_matrix` implementation so Phase 1 does not alter dense
    behavior. Its `iter_blocks()` yields the full matrix as a single block.
    """

    engine_name = "dense"

    def to_dense(self) -> np.ndarray:
        """
        @brief Build the dense similarity matrix using the existing legacy path.

        @return: Dense `(n_samples, n_samples)` similarity matrix.
        """
        if _is_nested_frs_config(self.config):
            return build_similarity_matrix(self.X, config=self.config)
        return build_similarity_matrix(self.X, **self.config)

    def iter_blocks(self) -> Iterator[SimilarityBlock]:
        """
        @brief Yield the full dense matrix as a single similarity block.

        @return: Iterator with one SimilarityBlock.
        """
        yield SimilarityBlock(
            row_slice=slice(0, self.n_samples),
            col_slice=slice(0, self.n_samples),
            values=self.to_dense(),
            values_backend="numpy",
            values_are_backend_resident=False,
        )


class BlockwiseSimilarityEngine(BaseSimilarityEngine):
    """
    @brief Exact blockwise similarity engine for future streaming approximations.

    Phase 1 uses this engine only for equivalence tests and dense materialization.
    Future phases can consume `iter_blocks()` directly in ITFRS/VQRS/OWAFRS
    accumulators without allocating the full pairwise matrix.

    @param X: Feature matrix used to compute pairwise similarities.
    @param block_size: Positive row/column block size.
    @param config: Optional flat or nested similarity configuration.
    @param backend: Array backend alias: "numpy", "auto", or explicit "cupy".
    @param flat_config: Additional flat sklearn-style config values.
    """

    engine_name = "blockwise"

    def __init__(
        self,
        X: Any,
        *,
        block_size: int = 1024,
        config: Optional[Mapping[str, Any]] = None,
        backend: str = "numpy",
        **flat_config: Any,
    ) -> None:
        super().__init__(X, config=config, backend=backend, **flat_config)
        self.block_size = _validate_block_size(block_size)

    def _iter_blocks_impl(self, *, return_backend_array: bool) -> Iterator[SimilarityBlock]:
        """
        @brief Shared row-major block iterator for NumPy and backend-resident values.

        @param return_backend_array: If True, keep CuPy values resident when backend='cupy'.
        @return: Iterator over SimilarityBlock values.
        """
        n = self.n_samples
        for row_start in range(0, n, self.block_size):
            row_stop = min(row_start + self.block_size, n)
            row_slice = slice(row_start, row_stop)
            X_rows = self.X[row_slice]

            for col_start in range(0, n, self.block_size):
                col_stop = min(col_start + self.block_size, n)
                col_slice = slice(col_start, col_stop)
                X_cols = self.X[col_slice]

                values = calculate_similarity_block(
                    X_rows,
                    X_cols,
                    self.similarity_func,
                    self.tnorm_func,
                    backend=self.backend,
                    return_backend_array=return_backend_array,
                )

                if row_start == col_start and row_stop == col_stop and getattr(values, "size", 0):
                    self.backend.xp.fill_diagonal(values, 1.0)

                values_are_backend_resident = bool(return_backend_array and is_cupy_backend(self.backend))
                if not return_backend_array and is_cupy_backend(self.backend):
                    values_backend = "numpy"
                else:
                    values_backend = self.backend.name

                yield SimilarityBlock(
                    row_slice=row_slice,
                    col_slice=col_slice,
                    values=values,
                    values_backend=values_backend,
                    values_are_backend_resident=values_are_backend_resident,
                )

    def iter_blocks(self) -> Iterator[SimilarityBlock]:
        """
        @brief Yield exact pairwise similarity blocks as NumPy arrays.

        @return: Iterator over NumPy-backed SimilarityBlock values.
        """
        for block in self._iter_blocks_impl(return_backend_array=False):
            if block.values_are_backend_resident:
                yield SimilarityBlock(
                    row_slice=block.row_slice,
                    col_slice=block.col_slice,
                    values=self.backend.to_numpy(block.values),
                    values_backend="numpy",
                    values_are_backend_resident=False,
                )
            else:
                yield block

    def iter_backend_blocks(self) -> Iterator[SimilarityBlock]:
        """
        @brief Yield exact pairwise blocks using the resolved backend when possible.

        For backend='cupy', block values remain on GPU so GPU-aware approximation
        engines can apply implicator/T-norm reductions without an immediate
        CPU round-trip. For NumPy, this is equivalent to iter_blocks().

        @return: Iterator over SimilarityBlock values.
        """
        yield from self._iter_blocks_impl(return_backend_array=True)


def build_similarity_engine(
    X: Any,
    *,
    engine: str = "dense",
    block_size: int = 1024,
    config: Optional[Mapping[str, Any]] = None,
    backend: str = "numpy",
    **flat_config: Any,
) -> BaseSimilarityEngine:
    """
    @brief Build a similarity engine by alias.

    @param X: Feature matrix used to compute pairwise similarities.
    @param engine: Engine alias, currently "dense" or "blockwise".
    @param block_size: Positive block size for blockwise engines.
    @param config: Optional flat or nested similarity configuration.
    @param backend: Array backend alias: "numpy", "auto", or explicit "cupy".
    @param flat_config: Additional flat sklearn-style config values.
    @return: Concrete similarity engine instance.
    @raises TypeError: If engine is not a string.
    @raises ValueError: If engine is unknown.
    """
    if not isinstance(engine, str) or not engine.strip():
        raise TypeError("engine must be a non-empty string.")

    normalized = engine.strip().lower()
    if normalized in {"dense", "full", "matrix"}:
        return DenseSimilarityEngine(X, config=config, backend=backend, **flat_config)
    if normalized in {"blockwise", "chunkwise", "blocked"}:
        return BlockwiseSimilarityEngine(
            X,
            block_size=block_size,
            config=config,
            backend=backend,
            **flat_config,
        )

    raise ValueError("Unknown similarity engine. Use engine='dense' or engine='blockwise'.")


__all__ = [
    "BaseSimilarityEngine",
    "BlockwiseSimilarityEngine",
    "DenseSimilarityEngine",
    "SimilarityBlock",
    "build_similarity_components",
    "build_similarity_engine",
    "calculate_similarity_block",
]
