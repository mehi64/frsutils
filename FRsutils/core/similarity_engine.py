# SPDX-License-Identifier: BSD-3-Clause
"""Similarity-matrix execution engine with dense and blockwise computation paths.

This module belongs to the core fuzzy-rough computation layer.
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
    """Immutable pairwise similarity block produced by a similarity engine.
    
    Parameters
    ----------
    row_slice : object
        Row sample slice represented by this block.
    col_slice : object
        Column sample slice represented by this block.
    values : object
        Similarity values with shape `(len(row_slice), len(col_slice))`.
    values_backend : object
        Name of the array backend that owns `values`.
    values_are_backend_resident : object
        True when `values` may be a non-NumPy backend array.
    """

    row_slice: slice
    col_slice: slice
    values: Any
    values_backend: str = "numpy"
    values_are_backend_resident: bool = False



def _as_2d_feature_matrix(X: Any) -> np.ndarray:
    """Convert and validate feature matrix input for similarity engines.
        
        Parameters
        ----------
        X : Any
            Candidate feature matrix.
        
        Returns
        -------
        np.ndarray
            Two-dimensional NumPy array.
        
        Raises
        ------
        ValueError
            If X is missing or not two-dimensional.
        
    """
    if X is None:
        raise ValueError("X must be provided when building a similarity engine.")

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like feature matrix.")
    return X_array



def _validate_block_size(block_size: int) -> int:
    """Validate and normalize a block size.
        
        Parameters
        ----------
        block_size : int
            Candidate positive integer block size.
        
        Returns
        -------
        int
            Normalized block size.
        
        Raises
        ------
        TypeError
            If block_size is not an integer.
        ValueError
            If block_size is less than one.
        
    """
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise TypeError("block_size must be an integer.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    return block_size



def _is_nested_frs_config(config: Mapping[str, Any]) -> bool:
    """Return True when config already looks like a nested FRsutils config.
        
        Parameters
        ----------
        config : Mapping[str, Any]
            Candidate configuration mapping.
        
        Returns
        -------
        bool
            True if nested similarity sections are present.
        
    """
    return isinstance(config.get("similarity"), Mapping) or isinstance(config.get("similarity_tnorm"), Mapping)



def _prepare_similarity_config(
    config: Optional[Mapping[str, Any]],
    flat_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Prepare an effective flat or nested similarity config.
        
        Parameters
        ----------
        config : Optional[Mapping[str, Any]]
            Optional flat or nested mapping.
        flat_config : Mapping[str, Any]
            Additional flat keyword parameters.
        
        Returns
        -------
        Dict[str, Any]
            Defensive configuration dictionary.
        
        Raises
        ------
        TypeError
            If config is not mapping-like.
        ValueError
            If nested config is mixed with extra flat kwargs.
        
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
    """Build similarity and T-norm components from flat or nested config.
        
        This mirrors `FRsutils.core.similarities.build_similarity_matrix` component
        resolution so blockwise engines remain equivalent to the dense path.
        
        Parameters
        ----------
        config : Optional[Mapping[str, Any]]
            Optional flat or nested configuration mapping.
        flat_config : Any
            Additional flat config values.
        
        Returns
        -------
        Tuple[Similarity, TNorm, Dict[str, Any]]
            Tuple of `(similarity_func, tnorm_func, effective_config)`.
        
    """
    effective_config = _prepare_similarity_config(config, flat_config)
    nested = effective_config if _is_nested_frs_config(effective_config) else normalize_flat_config_to_nested(effective_config)

    sim_cfg = nested.get("similarity", {}) if isinstance(nested, Mapping) else {}
    tnorm_cfg = nested.get("similarity_tnorm", {}) if isinstance(nested, Mapping) else {}

    flat_similarity = effective_config.get("similarity")
    if isinstance(flat_similarity, Mapping):
        flat_similarity = None
    flat_similarity_tnorm = effective_config.get("similarity_tnorm")
    if isinstance(flat_similarity_tnorm, Mapping):
        flat_similarity_tnorm = None

    similarity_type = sim_cfg.get("name") or flat_similarity or "gaussian"
    similarity_params = dict(sim_cfg.get("params") if isinstance(sim_cfg.get("params"), dict) else {})
    if str(similarity_type).lower() in {"gaussian", "gauss"} and "sigma" in effective_config and "sigma" not in similarity_params:
        # Backward-compatible legacy flat alias used by older tests/examples.
        similarity_params["sigma"] = effective_config["sigma"]

    tnorm_type = tnorm_cfg.get("name") or flat_similarity_tnorm or "minimum"
    tnorm_params = tnorm_cfg.get("params") if isinstance(tnorm_cfg.get("params"), dict) else {}

    similarity_func = Similarity.create(similarity_type, **similarity_params)
    tnorm_func = TNorm.create(tnorm_type, **tnorm_params)
    return similarity_func, tnorm_func, dict(effective_config)



def _registered_component_name(registry_cls: Any, component: Any) -> str:
    """Return the primary registered alias for a pluggable component.
        
        Parameters
        ----------
        registry_cls : Any
            Registry base class such as Similarity or TNorm.
        component : Any
            Concrete component instance.
        
        Returns
        -------
        str
            Registered alias or the component's fallback `name` property.
        
    """
    try:
        return registry_cls.get_registered_name(component)
    except Exception:
        return str(getattr(component, "name", component.__class__.__name__)).lower()


def _compute_similarity_from_diff(diff: Any, similarity_func: Similarity, backend: ArrayBackend):
    """Compute feature-level similarity on a backend array.
        
        Backend-specific formulas live in Similarity components, so this helper acts
        only as a small engine-to-component adapter.
        
        Parameters
        ----------
        diff : Any
            Pairwise backend-array difference matrix.
        similarity_func : Similarity
            Built similarity component.
        backend : ArrayBackend
            Resolved array backend.
        
        Returns
        -------
        object
            Backend-array similarity matrix.
        
        Raises
        ------
        NotImplementedError
            If a similarity has no backend formula yet.
        
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
    """Apply a T-norm component on backend arrays.
        
        Backend-specific T-norm formulas live in TNorm components, so this helper acts
        only as a small engine-to-component adapter.
        
        Parameters
        ----------
        a : Any
            First backend array.
        b : Any
            Second backend array.
        tnorm : TNorm
            Built T-norm component.
        backend : ArrayBackend
            Resolved array backend.
        
        Returns
        -------
        object
            Backend-array T-norm result.
        
        Raises
        ------
        NotImplementedError
            If a T-norm has no backend formula yet.
        
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
    """Compute an exact pairwise similarity block between two feature matrices.
        
        Parameters
        ----------
        X_rows : Any
            Row-side feature matrix with shape `(n_rows, n_features)`.
        X_cols : Any
            Column-side feature matrix with shape `(n_cols, n_features)`.
        similarity_func : Similarity
            Built similarity component.
        tnorm : object
            Built binary T-norm component/callable.
        backend : Optional[ArrayBackend]
            Optional ArrayBackend. NumPy is used when omitted.
        return_backend_array : bool
            If True, keep CuPy values on GPU instead of converting to NumPy.
        
        Returns
        -------
        Any
            Similarity block with shape `(n_rows, n_cols)` as NumPy or backend array.
        
        Raises
        ------
        ValueError
            If feature dimensions do not match.
        
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
    """Base class shared by concrete similarity-engine strategies.
    
    Parameters
    ----------
    X : object
        Feature matrix used to compute pairwise similarities.
    config : object
        Optional flat or nested similarity configuration.
    backend : object
        Array backend alias: "numpy", "auto", or explicit "cupy".
    flat_config : object
        Additional flat sklearn-style config values.
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
        """Initialize the BaseSimilarityEngine instance."""
        self.X = _as_2d_feature_matrix(X)
        self.backend: ArrayBackend = build_array_backend(backend)
        self.similarity_func, self.tnorm_func, self.config = build_similarity_components(config, **flat_config)

    @property
    def n_samples(self) -> int:
        """Number of samples in the engine feature matrix.
                
                Returns
                -------
                int
                    Number of rows in X.
                
        """
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features in the engine feature matrix.
                
                Returns
                -------
                int
                    Number of columns in X.
                
        """
        return self.X.shape[1]

    def iter_blocks(self) -> Iterator[SimilarityBlock]:
        """Yield exact similarity blocks as NumPy arrays.
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator over NumPy-backed SimilarityBlock values.
                
        """
        raise NotImplementedError("Concrete similarity engines must implement iter_blocks().")

    def iter_backend_blocks(self) -> Iterator[SimilarityBlock]:
        """Yield exact similarity blocks in the engine backend when supported.
                
                The default implementation preserves compatibility by delegating to
                iter_blocks(), which yields NumPy-backed values. Blockwise engines can
                override this to keep CuPy block values resident for GPU-aware
                approximation accumulators.
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator over SimilarityBlock values.
                
        """
        yield from self.iter_blocks()

    def to_dense(self) -> np.ndarray:
        """Materialize this engine as a dense pairwise similarity matrix.
                
                Returns
                -------
                np.ndarray
                    Dense `(n_samples, n_samples)` similarity matrix.
                
        """
        dense = np.zeros((self.n_samples, self.n_samples), dtype=np.float64)
        for block in self.iter_blocks():
            dense[block.row_slice, block.col_slice] = block.values
        if self.n_samples:
            np.fill_diagonal(dense, 1.0)
        return dense


class DenseSimilarityEngine(BaseSimilarityEngine):
    """Compatibility engine that materializes the existing dense matrix path.
    
    DenseSimilarityEngine intentionally delegates `to_dense()` to the current
    `build_similarity_matrix` implementation so dense behavior remains the
    reference path. Its `iter_blocks()` yields the full matrix as a single block.
    """

    engine_name = "dense"

    def to_dense(self) -> np.ndarray:
        """Build the dense similarity matrix using the existing legacy path.
                
                Returns
                -------
                np.ndarray
                    Dense `(n_samples, n_samples)` similarity matrix.
                
        """
        if _is_nested_frs_config(self.config):
            return build_similarity_matrix(self.X, config=self.config)
        return build_similarity_matrix(self.X, **self.config)

    def iter_blocks(self) -> Iterator[SimilarityBlock]:
        """Yield the full dense matrix as a single similarity block.
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator with one SimilarityBlock.
                
        """
        yield SimilarityBlock(
            row_slice=slice(0, self.n_samples),
            col_slice=slice(0, self.n_samples),
            values=self.to_dense(),
            values_backend="numpy",
            values_are_backend_resident=False,
        )


class BlockwiseSimilarityEngine(BaseSimilarityEngine):
    """Exact blockwise similarity engine for streaming-friendly approximations.
    
    The engine exposes `iter_blocks()` for ITFRS/VQRS/OWAFRS accumulators that
    should avoid allocating the full pairwise matrix.
    
    Parameters
    ----------
    X : object
        Feature matrix used to compute pairwise similarities.
    block_size : object
        Positive row/column block size.
    config : object
        Optional flat or nested similarity configuration.
    backend : object
        Array backend alias: "numpy", "auto", or explicit "cupy".
    flat_config : object
        Additional flat sklearn-style config values.
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
        """Initialize the BlockwiseSimilarityEngine instance."""
        super().__init__(X, config=config, backend=backend, **flat_config)
        self.block_size = _validate_block_size(block_size)

    def _iter_blocks_impl(self, *, return_backend_array: bool) -> Iterator[SimilarityBlock]:
        """Shared row-major block iterator for NumPy and backend-resident values.
                
                Parameters
                ----------
                return_backend_array : bool
                    If True, keep CuPy values resident when backend='cupy'.
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator over SimilarityBlock values.
                
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
                    if return_backend_array and is_cupy_backend(self.backend):
                        self.backend.xp.fill_diagonal(values, 1.0)
                    else:
                        np.fill_diagonal(values, 1.0)

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
        """Yield exact pairwise similarity blocks as NumPy arrays.
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator over NumPy-backed SimilarityBlock values.
                
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
        """Yield exact pairwise blocks using the resolved backend when possible.
                
                For backend='cupy', block values remain on GPU so GPU-aware approximation
                engines can apply implicator/T-norm reductions without an immediate
                CPU round-trip. For NumPy, this is equivalent to iter_blocks().
                
                Returns
                -------
                Iterator[SimilarityBlock]
                    Iterator over SimilarityBlock values.
                
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
    """Build a similarity engine by alias.
        
        Parameters
        ----------
        X : Any
            Feature matrix used to compute pairwise similarities.
        engine : str
            Engine alias, currently "dense" or "blockwise".
        block_size : int
            Positive block size for blockwise engines.
        config : Optional[Mapping[str, Any]]
            Optional flat or nested similarity configuration.
        backend : str
            Array backend alias: "numpy", "auto", or explicit "cupy".
        flat_config : Any
            Additional flat sklearn-style config values.
        
        Returns
        -------
        BaseSimilarityEngine
            Concrete similarity engine instance.
        
        Raises
        ------
        TypeError
            If engine is not a string.
        ValueError
            If engine is unknown.
        
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
