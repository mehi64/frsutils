# SPDX-License-Identifier: BSD-3-Clause
"""Core behavior tests for similarity-engine execution strategies."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import builtins
import sys

import numpy as np
import pytest

from frsutils.core.backends import ArrayBackend, build_array_backend
from frsutils.core.similarities import Similarity, build_similarity_matrix
from frsutils.core.similarity_engine import (
    BaseSimilarityEngine,
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    SimilarityBlock,
    _apply_tnorm_backend,
    _compute_similarity_from_diff,
    _as_2d_feature_matrix,
    _validate_block_size,
    build_similarity_components,
    build_similarity_engine,
    calculate_similarity_block,
)
from frsutils.core.tnorms import TNorm
from frsutils.utils.init_helpers import normalize_flat_config_to_nested
from tests._fake_cupy_backend import FakeCupyArray, install_fake_cupy_module


X_ENGINE = np.array(
    [
        [0.0, 0.0, 0.2],
        [0.1, 0.2, 0.2],
        [0.8, 0.7, 0.5],
        [0.9, 1.0, 0.4],
        [0.4, 0.5, 0.9],
    ],
    dtype=float,
)


@pytest.mark.parametrize(
    "invalid_x",
    [None, np.array([0.0, 1.0]), np.zeros((1, 2, 3), dtype=float)],
)
def test_as_2d_feature_matrix_rejects_missing_or_non_2d_input(invalid_x):
    with pytest.raises(ValueError):
        _as_2d_feature_matrix(invalid_x)


def test_as_2d_feature_matrix_converts_array_like_input_to_float_array():
    X = [[0, 1], [2, 3]]

    result = _as_2d_feature_matrix(X)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == (2, 2)
    np.testing.assert_allclose(result, np.array(X, dtype=float))


@pytest.mark.parametrize("block_size", [1, 2, 10])
def test_validate_block_size_accepts_positive_integers(block_size):
    assert _validate_block_size(block_size) == block_size


@pytest.mark.parametrize("block_size", [0, -1])
def test_validate_block_size_rejects_non_positive_integers(block_size):
    with pytest.raises(ValueError):
        _validate_block_size(block_size)


@pytest.mark.parametrize("block_size", [1.5, "2", None, True, False])
def test_validate_block_size_rejects_non_integer_values_and_bools(block_size):
    with pytest.raises(TypeError):
        _validate_block_size(block_size)


@pytest.mark.parametrize(
    "flat_config",
    [
        {"similarity": "linear", "similarity_tnorm": "minimum"},
        {"similarity": "linear", "similarity_tnorm": "product"},
        {"similarity": "gaussian", "similarity_sigma": 0.5, "similarity_tnorm": "minimum"},
        {"similarity": "gaussian", "sigma": 0.5, "similarity_tnorm": "product"},
    ],
)
def test_build_similarity_components_matches_dense_builder_for_flat_configs(flat_config):
    similarity_func, tnorm_func, effective_config = build_similarity_components(**flat_config)

    direct = calculate_similarity_block(X_ENGINE, X_ENGINE, similarity_func, tnorm_func)
    expected = build_similarity_matrix(X_ENGINE, **flat_config)

    assert effective_config == flat_config
    np.testing.assert_allclose(direct, expected, atol=1e-12)


def test_build_similarity_components_nested_config_matches_flat_config():
    flat_config = {
        "type": "itfrs",
        "similarity": "linear",
        "similarity_tnorm": "minimum",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    }
    nested_config = normalize_flat_config_to_nested(flat_config)

    flat_similarity, flat_tnorm, _ = build_similarity_components(**flat_config)
    nested_similarity, nested_tnorm, _ = build_similarity_components(config=nested_config)

    flat_block = calculate_similarity_block(X_ENGINE, X_ENGINE, flat_similarity, flat_tnorm)
    nested_block = calculate_similarity_block(X_ENGINE, X_ENGINE, nested_similarity, nested_tnorm)
    np.testing.assert_allclose(nested_block, flat_block, atol=1e-12)


def test_build_similarity_components_rejects_nested_config_mixed_with_flat_kwargs():
    nested_config = normalize_flat_config_to_nested(
        {"similarity": "linear", "similarity_tnorm": "minimum"}
    )

    with pytest.raises(ValueError):
        build_similarity_components(config=nested_config, similarity="gaussian")


@pytest.mark.parametrize("config", [[("similarity", "linear")], "linear"])
def test_build_similarity_components_rejects_non_mapping_config(config):
    with pytest.raises(TypeError):
        build_similarity_components(config=config)


def test_build_similarity_components_does_not_mutate_config_mapping():
    config = {"similarity": "gaussian", "similarity_sigma": 0.75, "similarity_tnorm": "minimum"}
    before = deepcopy(config)

    build_similarity_components(config=config)

    assert config == before


@pytest.mark.parametrize(
    "similarity_name, tnorm_name",
    [
        ("linear", "minimum"),
        ("linear", "product"),
        ("gaussian", "minimum"),
        ("gaussian", "product"),
    ],
)
def test_calculate_similarity_block_matches_dense_submatrix_for_registered_pairs(
    similarity_name,
    tnorm_name,
):
    similarity_kwargs = {"sigma": 0.5} if similarity_name == "gaussian" else {}
    similarity_func = Similarity.create(similarity_name, **similarity_kwargs)
    tnorm_func = TNorm.create(tnorm_name)

    block = calculate_similarity_block(
        X_ENGINE[0:2],
        X_ENGINE[2:5],
        similarity_func,
        tnorm_func,
    )
    dense = build_similarity_matrix(
        X_ENGINE,
        similarity=similarity_name,
        similarity_sigma=0.5,
        similarity_tnorm=tnorm_name,
    )

    np.testing.assert_allclose(block, dense[0:2, 2:5], atol=1e-12)


def test_calculate_similarity_block_supports_rectangular_inputs():
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    block = calculate_similarity_block(X_ENGINE[0:1], X_ENGINE[1:5], similarity_func, tnorm_func)

    assert block.shape == (1, 4)
    expected = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")
    np.testing.assert_allclose(block, expected[0:1, 1:5], atol=1e-12)


@pytest.mark.parametrize(
    "X_rows, X_cols, expected_shape",
    [
        (np.empty((0, 3), dtype=float), X_ENGINE, (0, 5)),
        (X_ENGINE, np.empty((0, 3), dtype=float), (5, 0)),
    ],
)
def test_calculate_similarity_block_handles_empty_sides(X_rows, X_cols, expected_shape):
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    block = calculate_similarity_block(X_rows, X_cols, similarity_func, tnorm_func)

    assert block.shape == expected_shape
    assert block.dtype == np.float64


def test_calculate_similarity_block_rejects_feature_dimension_mismatch():
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    with pytest.raises(ValueError):
        calculate_similarity_block(
            np.zeros((2, 3), dtype=float),
            np.zeros((2, 2), dtype=float),
            similarity_func,
            tnorm_func,
        )


@pytest.mark.parametrize(
    "X_rows, X_cols",
    [
        (None, X_ENGINE),
        (X_ENGINE, None),
        (np.array([0.0, 1.0]), X_ENGINE),
        (X_ENGINE, np.zeros((1, 2, 3), dtype=float)),
    ],
)
def test_calculate_similarity_block_rejects_invalid_row_or_col_matrices(X_rows, X_cols):
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    with pytest.raises(ValueError):
        calculate_similarity_block(X_rows, X_cols, similarity_func, tnorm_func)


def test_calculate_similarity_block_returns_float64_numpy_array():
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    block = calculate_similarity_block(X_ENGINE[0:2], X_ENGINE[0:3], similarity_func, tnorm_func)

    assert isinstance(block, np.ndarray)
    assert block.dtype == np.float64


class _FakeSimilarityEngine(BaseSimilarityEngine):
    """Small test double exposing BaseSimilarityEngine dense reconstruction."""

    engine_name = "fake"

    def iter_blocks(self):
        yield SimilarityBlock(slice(0, 2), slice(0, 2), np.array([[0.2, 0.3], [0.4, 0.5]]))
        yield SimilarityBlock(slice(0, 2), slice(2, 3), np.array([[0.6], [0.7]]))
        yield SimilarityBlock(slice(2, 3), slice(0, 2), np.array([[0.8, 0.9]]))
        yield SimilarityBlock(slice(2, 3), slice(2, 3), np.array([[0.1]]))


def test_base_similarity_engine_exposes_sample_and_feature_counts():
    engine = _FakeSimilarityEngine(X_ENGINE, similarity="linear")

    assert engine.n_samples == 5
    assert engine.n_features == 3


def test_base_similarity_engine_iter_blocks_must_be_implemented():
    engine = BaseSimilarityEngine(X_ENGINE, similarity="linear")

    with pytest.raises(NotImplementedError):
        list(engine.iter_blocks())


def test_base_similarity_engine_iter_backend_blocks_delegates_to_iter_blocks():
    engine = _FakeSimilarityEngine(X_ENGINE[0:3], similarity="linear")

    blocks = list(engine.iter_backend_blocks())

    assert len(blocks) == 4
    assert all(isinstance(block, SimilarityBlock) for block in blocks)


def test_base_similarity_engine_to_dense_reconstructs_blocks_and_sets_diagonal():
    engine = _FakeSimilarityEngine(X_ENGINE[0:3], similarity="linear")

    dense = engine.to_dense()

    expected = np.array(
        [
            [1.0, 0.3, 0.6],
            [0.4, 1.0, 0.7],
            [0.8, 0.9, 1.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(dense, expected, atol=1e-12)


def test_dense_similarity_engine_to_dense_matches_existing_dense_path():
    expected = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")
    engine = DenseSimilarityEngine(X_ENGINE, similarity="linear", similarity_tnorm="minimum")

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)


def test_dense_similarity_engine_iter_blocks_yields_single_full_numpy_block():
    engine = DenseSimilarityEngine(X_ENGINE, similarity="linear", similarity_tnorm="minimum")

    blocks = list(engine.iter_blocks())

    assert len(blocks) == 1
    block = blocks[0]
    assert block.row_slice == slice(0, X_ENGINE.shape[0])
    assert block.col_slice == slice(0, X_ENGINE.shape[0])
    assert block.values_backend == "numpy"
    assert block.values_are_backend_resident is False
    np.testing.assert_allclose(block.values, engine.to_dense(), atol=1e-12)


@pytest.mark.parametrize("block_size", [1, 2, 3, 10])
def test_blockwise_similarity_engine_to_dense_matches_dense_engine_for_block_sizes(block_size):
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=block_size,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)


def test_blockwise_similarity_engine_yields_expected_row_major_slices_and_shapes():
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    blocks = list(engine.iter_blocks())

    assert [(block.row_slice, block.col_slice) for block in blocks] == [
        (slice(0, 2), slice(0, 2)),
        (slice(0, 2), slice(2, 4)),
        (slice(0, 2), slice(4, 5)),
        (slice(2, 4), slice(0, 2)),
        (slice(2, 4), slice(2, 4)),
        (slice(2, 4), slice(4, 5)),
        (slice(4, 5), slice(0, 2)),
        (slice(4, 5), slice(2, 4)),
        (slice(4, 5), slice(4, 5)),
    ]
    assert [block.values.shape for block in blocks] == [
        (2, 2),
        (2, 2),
        (2, 1),
        (2, 2),
        (2, 2),
        (2, 1),
        (1, 2),
        (1, 2),
        (1, 1),
    ]


def test_blockwise_similarity_engine_iter_blocks_returns_numpy_metadata():
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=2, similarity="linear")

    blocks = list(engine.iter_blocks())

    assert blocks
    assert all(isinstance(block.values, np.ndarray) for block in blocks)
    assert all(block.values_backend == "numpy" for block in blocks)
    assert all(block.values_are_backend_resident is False for block in blocks)


def test_blockwise_similarity_engine_iter_backend_blocks_matches_iter_blocks_for_numpy_backend():
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=2, backend="numpy", similarity="linear")

    numpy_blocks = list(engine.iter_blocks())
    backend_blocks = list(engine.iter_backend_blocks())

    assert [(block.row_slice, block.col_slice) for block in backend_blocks] == [
        (block.row_slice, block.col_slice) for block in numpy_blocks
    ]
    for backend_block, numpy_block in zip(backend_blocks, numpy_blocks):
        assert backend_block.values_backend == "numpy"
        assert backend_block.values_are_backend_resident is False
        np.testing.assert_allclose(backend_block.values, numpy_block.values, atol=1e-12)


def test_blockwise_similarity_engine_block_size_one_yields_n_squared_blocks():
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=1, similarity="linear")

    blocks = list(engine.iter_blocks())

    assert len(blocks) == X_ENGINE.shape[0] ** 2
    assert all(block.values.shape == (1, 1) for block in blocks)


def test_blockwise_similarity_engine_large_block_size_yields_single_block():
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=99, similarity="linear")

    blocks = list(engine.iter_blocks())

    assert len(blocks) == 1
    assert blocks[0].row_slice == slice(0, X_ENGINE.shape[0])
    assert blocks[0].col_slice == slice(0, X_ENGINE.shape[0])


def test_blockwise_similarity_engine_empty_input_yields_empty_dense_matrix_and_no_blocks():
    X_empty = np.empty((0, 3), dtype=float)
    engine = BlockwiseSimilarityEngine(X_empty, block_size=2, similarity="linear")

    assert list(engine.iter_blocks()) == []
    assert engine.to_dense().shape == (0, 0)


def test_blockwise_similarity_engine_manual_block_reconstruction_matches_to_dense():
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    reconstructed = np.zeros((engine.n_samples, engine.n_samples), dtype=float)
    for block in engine.iter_blocks():
        reconstructed[block.row_slice, block.col_slice] = block.values

    np.testing.assert_allclose(reconstructed, engine.to_dense(), atol=1e-12)


@pytest.mark.parametrize("engine_alias", ["dense", "full", "matrix", " DENSE ", "FULL"])
def test_build_similarity_engine_dense_aliases_return_dense_engine(engine_alias):
    engine = build_similarity_engine(X_ENGINE, engine=engine_alias, similarity="linear")

    assert isinstance(engine, DenseSimilarityEngine)


@pytest.mark.parametrize("engine_alias", ["blockwise", "chunkwise", "blocked", " BLOCKWISE ", "CHUNKWISE"])
def test_build_similarity_engine_blockwise_aliases_return_blockwise_engine(engine_alias):
    engine = build_similarity_engine(X_ENGINE, engine=engine_alias, similarity="linear", block_size=2)

    assert isinstance(engine, BlockwiseSimilarityEngine)


@pytest.mark.parametrize("engine_alias", ["unknown", "similarity"])
def test_build_similarity_engine_rejects_unknown_engine_aliases(engine_alias):
    with pytest.raises(ValueError):
        build_similarity_engine(X_ENGINE, engine=engine_alias, similarity="linear")


@pytest.mark.parametrize("engine_alias", ["", "   ", 123, None])
def test_build_similarity_engine_rejects_missing_or_non_string_engine_aliases(engine_alias):
    with pytest.raises(TypeError):
        build_similarity_engine(X_ENGINE, engine=engine_alias, similarity="linear")


def test_build_similarity_engine_auto_backend_uses_numpy_backend():
    engine = build_similarity_engine(X_ENGINE, engine="blockwise", backend="auto", similarity="linear")

    assert engine.backend.name == "numpy"
    np.testing.assert_allclose(
        engine.to_dense(),
        build_similarity_matrix(X_ENGINE, similarity="linear"),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["unknown", "tensorflow"])
def test_build_similarity_engine_rejects_unknown_backend_aliases(backend):
    with pytest.raises(ValueError):
        build_similarity_engine(X_ENGINE, engine="blockwise", backend=backend, similarity="linear")


@pytest.mark.parametrize("backend", ["", "   ", None, 123])
def test_build_similarity_engine_rejects_missing_or_non_string_backend_aliases(backend):
    with pytest.raises(TypeError):
        build_similarity_engine(X_ENGINE, engine="blockwise", backend=backend, similarity="linear")


def test_calculate_similarity_block_defaults_to_numpy_backend_when_backend_is_omitted():
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    default_block = calculate_similarity_block(X_ENGINE, X_ENGINE, similarity_func, tnorm_func)
    explicit_block = calculate_similarity_block(
        X_ENGINE,
        X_ENGINE,
        similarity_func,
        tnorm_func,
        backend=build_array_backend("numpy"),
    )

    np.testing.assert_allclose(default_block, explicit_block, atol=1e-12)



def test_similarity_block_is_immutable_after_creation():
    block = SimilarityBlock(slice(0, 1), slice(1, 2), np.array([[0.5]], dtype=float))

    with pytest.raises(FrozenInstanceError):
        block.values_backend = "cupy"


@pytest.mark.parametrize(
    "similarity_name, similarity_kwargs, tnorm_name",
    [
        ("linear", {}, "minimum"),
        ("linear", {}, "product"),
        ("linear", {}, "lukasiewicz"),
        ("gaussian", {"sigma": 0.5}, "minimum"),
        ("gaussian", {"sigma": 0.5}, "product"),
        ("gaussian", {"sigma": 0.5}, "lukasiewicz"),
    ],
)
def test_blockwise_similarity_engine_outputs_core_numerical_invariants(
    similarity_name,
    similarity_kwargs,
    tnorm_name,
):
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        similarity=similarity_name,
        similarity_tnorm=tnorm_name,
        **{f"similarity_{key}": value for key, value in similarity_kwargs.items()},
    )

    matrix = engine.to_dense()

    assert matrix.shape == (X_ENGINE.shape[0], X_ENGINE.shape[0])
    np.testing.assert_allclose(np.diag(matrix), np.ones(X_ENGINE.shape[0]), atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_identical_samples_have_similarity_one_under_blockwise_engine():
    X = np.array(
        [
            [0.2, 0.4, 0.6],
            [0.2, 0.4, 0.6],
            [0.9, 0.8, 0.1],
        ],
        dtype=float,
    )
    engine = BlockwiseSimilarityEngine(X, block_size=2, similarity="linear")

    matrix = engine.to_dense()

    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(1.0)


def test_linear_similarity_clips_large_feature_distances_to_zero():
    X = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=float)
    engine = BlockwiseSimilarityEngine(X, block_size=1, similarity="linear", similarity_tnorm="minimum")

    matrix = engine.to_dense()

    assert matrix[0, 1] == pytest.approx(0.0)
    assert matrix[1, 0] == pytest.approx(0.0)


def test_gaussian_similarity_decreases_when_sigma_decreases_for_fixed_distance():
    X = np.array([[0.0], [0.5]], dtype=float)

    small_sigma = BlockwiseSimilarityEngine(X, block_size=1, similarity="gaussian", similarity_sigma=0.1).to_dense()
    large_sigma = BlockwiseSimilarityEngine(X, block_size=1, similarity="gaussian", similarity_sigma=1.0).to_dense()

    assert small_sigma[0, 1] < large_sigma[0, 1]
    assert small_sigma[0, 1] > 0.0
    assert large_sigma[0, 1] < 1.0


def test_blockwise_diagonal_blocks_are_forced_to_exact_unit_diagonal():
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=2, similarity="gaussian", similarity_sigma=0.5)

    for block in engine.iter_blocks():
        if block.row_slice == block.col_slice:
            np.testing.assert_allclose(np.diag(block.values), np.ones(block.values.shape[0]), atol=0.0)


def test_blockwise_off_diagonal_blocks_are_not_forced_to_unit_diagonal():
    X = np.array([[0.0], [0.2], [0.9], [1.0]], dtype=float)
    engine = BlockwiseSimilarityEngine(X, block_size=2, similarity="linear", similarity_tnorm="minimum")

    off_diagonal_block = next(
        block for block in engine.iter_blocks() if block.row_slice == slice(0, 2) and block.col_slice == slice(2, 4)
    )

    assert off_diagonal_block.values[0, 0] != pytest.approx(1.0)
    assert off_diagonal_block.values[1, 1] != pytest.approx(1.0)


def test_dense_engine_with_nested_config_keeps_nested_config_for_dense_path():
    nested_config = normalize_flat_config_to_nested(
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.5,
            "similarity_tnorm": "product",
        }
    )
    engine = DenseSimilarityEngine(X_ENGINE, config=nested_config)

    expected = build_similarity_matrix(X_ENGINE, config=nested_config)

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)
    assert engine.config == nested_config
    assert isinstance(engine.config["similarity"], dict)


def test_blockwise_engine_with_nested_config_keeps_nested_config_and_matches_dense_path():
    nested_config = normalize_flat_config_to_nested(
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.5,
            "similarity_tnorm": "product",
        }
    )
    engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=2, config=nested_config)

    expected = build_similarity_matrix(X_ENGINE, config=nested_config)

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)
    assert engine.config == nested_config
    assert isinstance(engine.config["similarity"], dict)


def test_flat_kwargs_override_flat_config_mapping_at_component_boundary():
    config = {"similarity": "linear", "similarity_tnorm": "minimum"}

    similarity_func, tnorm_func, effective_config = build_similarity_components(
        config=config,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )
    actual = calculate_similarity_block(X_ENGINE, X_ENGINE, similarity_func, tnorm_func)
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    assert effective_config["similarity"] == "gaussian"
    assert effective_config["similarity_tnorm"] == "product"
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_calculate_similarity_block_does_not_mutate_row_or_col_inputs():
    X_rows = X_ENGINE[0:2].copy()
    X_cols = X_ENGINE[2:5].copy()
    rows_before = X_rows.copy()
    cols_before = X_cols.copy()
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    _ = calculate_similarity_block(X_rows, X_cols, similarity_func, tnorm_func)

    np.testing.assert_array_equal(X_rows, rows_before)
    np.testing.assert_array_equal(X_cols, cols_before)


class _NumPyOnlySimilarity(Similarity):
    """Similarity test double without a non-NumPy backend formula."""

    name = "numpy_only_similarity"

    def _compute(self, diff):
        return np.maximum(0.0, 1.0 - np.abs(diff))

    @classmethod
    def validate_params(cls, **kwargs):
        return None

    def _get_params(self):
        return {}


class _NumPyOnlyTNorm(TNorm):
    """T-norm test double without a non-NumPy backend formula."""

    name = "numpy_only_tnorm"

    def compute_backend(self, a, b, *, xp=np):
        if xp is not np:
            raise NotImplementedError("non-NumPy backend is intentionally unsupported")
        return np.minimum(a, b)

    def reduce_backend(self, arr, *, xp=np):
        if xp is not np:
            raise NotImplementedError("non-NumPy backend is intentionally unsupported")
        return np.min(arr, axis=0)

    @classmethod
    def validate_params(cls, **kwargs):
        return None

    def _get_params(self):
        return {}


def test_compute_similarity_from_diff_raises_clear_error_for_unsupported_backend_formula():
    backend = ArrayBackend(name="fakegpu", xp=object())
    diff = np.array([[0.0, 0.5]], dtype=float)

    with pytest.raises(NotImplementedError, match="backend='fakegpu'.*similarity='numpy_only_similarity'.*backend='numpy'"):
        _compute_similarity_from_diff(diff, _NumPyOnlySimilarity(), backend)


def test_apply_tnorm_backend_raises_clear_error_for_unsupported_backend_formula():
    backend = ArrayBackend(name="fakegpu", xp=object())
    a = np.array([[1.0, 0.5]], dtype=float)
    b = np.array([[0.8, 0.2]], dtype=float)

    with pytest.raises(NotImplementedError, match="backend='fakegpu'.*similarity_tnorm='numpy_only_tnorm'.*backend='numpy'"):
        _apply_tnorm_backend(a, b, _NumPyOnlyTNorm(), backend)


def test_base_similarity_engine_to_dense_uses_float64_dense_output():
    engine = _FakeSimilarityEngine(X_ENGINE[0:3], similarity="linear")

    dense = engine.to_dense()

    assert dense.dtype == np.float64


def test_dense_and_blockwise_engines_are_independent_on_repeated_materialization():
    dense_engine = DenseSimilarityEngine(X_ENGINE, similarity="linear")
    blockwise_engine = BlockwiseSimilarityEngine(X_ENGINE, block_size=2, similarity="linear")

    dense_first = dense_engine.to_dense()
    blockwise_first = blockwise_engine.to_dense()
    dense_second = dense_engine.to_dense()
    blockwise_second = blockwise_engine.to_dense()

    np.testing.assert_allclose(dense_first, dense_second, atol=1e-12)
    np.testing.assert_allclose(blockwise_first, blockwise_second, atol=1e-12)
    np.testing.assert_allclose(blockwise_first, dense_first, atol=1e-12)



def test_build_array_backend_resolves_fake_cupy_module_when_optional_backend_is_available(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)

    backend = build_array_backend("cupy")

    assert backend.name == "cupy"
    assert backend.xp is fake_cupy


def test_build_array_backend_cupy_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("CuPy intentionally unavailable in this test.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match="backend='cupy'.*optional CuPy package"):
        build_array_backend("cupy")


@pytest.mark.parametrize(
    "similarity_name, similarity_kwargs, tnorm_name",
    [
        ("linear", {}, "minimum"),
        ("linear", {}, "product"),
        ("gaussian", {"sigma": 0.5}, "minimum"),
        ("gaussian", {"sigma": 0.5}, "product"),
    ],
)
def test_calculate_similarity_block_cupy_backend_matches_numpy_and_returns_numpy_by_default(
    monkeypatch,
    similarity_name,
    similarity_kwargs,
    tnorm_name,
):
    install_fake_cupy_module(monkeypatch)
    backend = build_array_backend("cupy")
    similarity_func = Similarity.create(similarity_name, **similarity_kwargs)
    tnorm_func = TNorm.create(tnorm_name)

    actual = calculate_similarity_block(
        X_ENGINE[0:3],
        X_ENGINE[1:5],
        similarity_func,
        tnorm_func,
        backend=backend,
    )
    expected = calculate_similarity_block(
        X_ENGINE[0:3],
        X_ENGINE[1:5],
        similarity_func,
        tnorm_func,
        backend=build_array_backend("numpy"),
    )

    assert isinstance(actual, np.ndarray)
    assert not isinstance(actual, FakeCupyArray)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_calculate_similarity_block_cupy_backend_can_return_backend_resident_values(monkeypatch):
    install_fake_cupy_module(monkeypatch)
    backend = build_array_backend("cupy")
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    actual = calculate_similarity_block(
        X_ENGINE[0:3],
        X_ENGINE[1:5],
        similarity_func,
        tnorm_func,
        backend=backend,
        return_backend_array=True,
    )
    expected = calculate_similarity_block(
        X_ENGINE[0:3],
        X_ENGINE[1:5],
        similarity_func,
        tnorm_func,
    )

    assert isinstance(actual, FakeCupyArray)
    np.testing.assert_allclose(backend.to_numpy(actual), expected, atol=1e-12)


def test_blockwise_iter_blocks_with_cupy_backend_returns_numpy_blocks_without_cupy_diagonal_calls(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        backend="cupy",
        similarity="linear",
        similarity_tnorm="minimum",
    )

    blocks = list(engine.iter_blocks())

    assert fake_cupy.fill_diagonal_calls == []
    assert all(isinstance(block.values, np.ndarray) for block in blocks)
    assert all(not isinstance(block.values, FakeCupyArray) for block in blocks)
    assert all(block.values_backend == "numpy" for block in blocks)
    assert all(block.values_are_backend_resident is False for block in blocks)
    np.testing.assert_allclose(engine.to_dense(), build_similarity_matrix(X_ENGINE, similarity="linear"), atol=1e-12)


def test_blockwise_iter_backend_blocks_with_cupy_backend_returns_backend_resident_blocks(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        backend="cupy",
        similarity="linear",
        similarity_tnorm="minimum",
    )

    blocks = list(engine.iter_backend_blocks())

    assert len(fake_cupy.fill_diagonal_calls) == 3
    assert all(isinstance(block.values, FakeCupyArray) for block in blocks)
    assert all(block.values_backend == "cupy" for block in blocks)
    assert all(block.values_are_backend_resident is True for block in blocks)
    for block in blocks:
        if block.row_slice == block.col_slice:
            np.testing.assert_allclose(
                np.diag(np.asarray(block.values)),
                np.ones(block.values.shape[0]),
                atol=0.0,
            )


def test_blockwise_to_dense_with_cupy_backend_matches_numpy_dense(monkeypatch):
    install_fake_cupy_module(monkeypatch)
    engine = BlockwiseSimilarityEngine(
        X_ENGINE,
        block_size=2,
        backend="cupy",
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    actual = engine.to_dense()
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    assert isinstance(actual, np.ndarray)
    assert not isinstance(actual, FakeCupyArray)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_dense_engine_with_cupy_backend_keeps_public_blocks_numpy_backed(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)
    engine = DenseSimilarityEngine(X_ENGINE, backend="cupy", similarity="linear")

    blocks = list(engine.iter_backend_blocks())

    assert engine.backend.xp is fake_cupy
    assert len(blocks) == 1
    assert isinstance(blocks[0].values, np.ndarray)
    assert not isinstance(blocks[0].values, FakeCupyArray)
    assert blocks[0].values_backend == "numpy"
    assert blocks[0].values_are_backend_resident is False
    np.testing.assert_allclose(blocks[0].values, build_similarity_matrix(X_ENGINE, similarity="linear"), atol=1e-12)
