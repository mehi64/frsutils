# SPDX-License-Identifier: BSD-3-Clause
"""Public contract tests for similarity-engine construction."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from frsutils import (
    BlockwiseSimilarityEngine,
    DenseSimilarityEngine,
    build_similarity_engine,
    build_similarity_matrix,
)
from frsutils.core.similarity_engine import calculate_similarity_block
from frsutils.core.similarities import Similarity
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
def test_public_similarity_engine_rejects_missing_or_non_2d_x(invalid_x):
    with pytest.raises(ValueError):
        build_similarity_engine(invalid_x, engine="dense", similarity="linear")


def test_public_similarity_engine_converts_array_like_x_to_float_matrix():
    engine = build_similarity_engine([[0, 1], [2, 3]], engine="blockwise", similarity="linear")

    assert engine.X.dtype == np.float64
    np.testing.assert_allclose(engine.X, np.array([[0.0, 1.0], [2.0, 3.0]]))


def test_public_build_similarity_matrix_accepts_python_list_input():
    matrix = build_similarity_matrix([[0.0, 0.5], [1.0, 0.25]], similarity="linear")

    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(np.diag(matrix), np.ones(2), atol=0.0)


def test_public_build_similarity_matrix_supports_flat_config_mapping_with_kwargs_override():
    config = {"similarity": "linear", "similarity_tnorm": "minimum"}

    actual = build_similarity_matrix(
        X_ENGINE,
        config=config,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert config == {"similarity": "linear", "similarity_tnorm": "minimum"}


def test_public_build_similarity_matrix_rejects_nested_config():
    nested_config = normalize_flat_config_to_nested(
        {"similarity": "linear", "similarity_tnorm": "minimum"}
    )

    with pytest.raises(ValueError):
        build_similarity_matrix(
            X_ENGINE,
            config=nested_config,
            similarity="gaussian",
            similarity_sigma=0.5,
        )


@pytest.mark.parametrize("config", [[("similarity", "linear")], "linear"])
def test_public_build_similarity_matrix_rejects_non_mapping_config(config):
    with pytest.raises(TypeError):
        build_similarity_matrix(X_ENGINE, config=config)


def test_public_build_similarity_matrix_rejects_unknown_similarity_alias():
    with pytest.raises(ValueError):
        build_similarity_matrix(X_ENGINE, similarity="not-a-similarity")


def test_public_build_similarity_matrix_does_not_mutate_flat_config_mapping():
    config = {"similarity": "gaussian", "similarity_sigma": 0.75, "similarity_tnorm": "minimum"}
    before = deepcopy(config)

    _ = build_similarity_matrix(X_ENGINE, config=config)

    assert config == before


@pytest.mark.parametrize(
    "flat_config",
    [
        {"similarity": "linear", "similarity_tnorm": "minimum"},
        {"similarity": "linear", "similarity_tnorm": "product"},
        {"similarity": "gaussian", "similarity_sigma": 0.5, "similarity_tnorm": "minimum"},
        {"similarity": "gaussian", "sigma": 0.5, "similarity_tnorm": "product"},
    ],
)
def test_dense_similarity_engine_matches_existing_dense_builder_for_flat_configs(flat_config):
    expected = build_similarity_matrix(X_ENGINE, **flat_config)
    engine = build_similarity_engine(X_ENGINE, engine="dense", **flat_config)

    assert isinstance(engine, DenseSimilarityEngine)
    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)

    blocks = list(engine.iter_blocks())
    assert len(blocks) == 1
    assert blocks[0].row_slice == slice(0, X_ENGINE.shape[0])
    assert blocks[0].col_slice == slice(0, X_ENGINE.shape[0])
    assert blocks[0].values_backend == "numpy"
    assert blocks[0].values_are_backend_resident is False
    np.testing.assert_allclose(blocks[0].values, expected, atol=1e-12)


@pytest.mark.parametrize("block_size", [1, 2, 3, 10])
def test_blockwise_similarity_engine_materializes_existing_dense_matrix(block_size):
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="linear",
        similarity_tnorm="minimum",
    )
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=block_size,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    assert isinstance(engine, BlockwiseSimilarityEngine)
    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)


@pytest.mark.parametrize(
    "flat_config",
    [
        {"similarity": "linear", "similarity_tnorm": "minimum"},
        {"similarity": "linear", "similarity_tnorm": "product"},
        {"similarity": "gaussian", "similarity_sigma": 0.5, "similarity_tnorm": "minimum"},
        {"similarity": "gaussian", "similarity_sigma": 0.5, "similarity_tnorm": "product"},
    ],
)
def test_blockwise_similarity_engine_matches_dense_for_multiple_registered_configs(flat_config):
    dense_engine = build_similarity_engine(X_ENGINE, engine="dense", **flat_config)
    blockwise_engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        **flat_config,
    )

    np.testing.assert_allclose(blockwise_engine.to_dense(), dense_engine.to_dense(), atol=1e-12)


def test_blockwise_similarity_engine_exposes_expected_block_slices_shapes_and_order():
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    blocks = list(engine.iter_blocks())

    assert len(blocks) == 9
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
    assert all(block.values_backend == "numpy" for block in blocks)
    assert all(block.values_are_backend_resident is False for block in blocks)


def test_blockwise_similarity_engine_reconstructs_dense_matrix_from_public_blocks():
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        similarity="linear",
        similarity_tnorm="minimum",
    )

    reconstructed = np.zeros((X_ENGINE.shape[0], X_ENGINE.shape[0]), dtype=float)
    for block in engine.iter_blocks():
        reconstructed[block.row_slice, block.col_slice] = block.values

    expected = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")
    np.testing.assert_allclose(reconstructed, expected, atol=1e-12)


def test_similarity_engine_rejects_model_configuration_parameters():
    """Similarity endpoints reject fuzzy-rough model configuration parameters."""
    with pytest.raises(ValueError, match="ub_tnorm_name"):
        build_similarity_engine(
            X_ENGINE,
            engine="blockwise",
            block_size=2,
            similarity="linear",
            ub_tnorm_name="minimum",
        )


def test_similarity_engine_rejects_nested_config():
    """Similarity engines accept the documented flat public contract only."""
    nested_config = normalize_flat_config_to_nested(
        {"similarity": "linear", "similarity_tnorm": "minimum"}
    )

    with pytest.raises(ValueError, match="Nested configuration is internal"):
        build_similarity_engine(
            X_ENGINE,
            engine="blockwise",
            block_size=2,
            config=nested_config,
        )


@pytest.mark.parametrize("config", [[("similarity", "linear")], "linear"])
def test_similarity_engine_rejects_non_mapping_config(config):
    with pytest.raises(TypeError):
        build_similarity_engine(X_ENGINE, engine="dense", config=config)


def test_similarity_engine_does_not_mutate_config_mapping():
    config = {"similarity": "gaussian", "similarity_sigma": 0.75, "similarity_tnorm": "minimum"}
    before = deepcopy(config)

    engine = build_similarity_engine(X_ENGINE, engine="blockwise", block_size=2, config=config)
    _ = engine.to_dense()

    assert config == before


def test_calculate_similarity_block_matches_corresponding_dense_submatrix():
    similarity_func = Similarity.create("linear")
    tnorm_func = TNorm.create("minimum")

    block = calculate_similarity_block(
        X_ENGINE[0:2],
        X_ENGINE[2:5],
        similarity_func,
        tnorm_func,
    )
    dense = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")

    np.testing.assert_allclose(block, dense[0:2, 2:5], atol=1e-12)


def test_similarity_engine_does_not_mutate_input_matrix():
    X = X_ENGINE.copy()
    before = X.copy()

    engine = build_similarity_engine(X, engine="blockwise", block_size=2, similarity="linear")
    _ = engine.to_dense()

    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("engine_name", ["dense", "full", "matrix", " DENSE ", "FULL"])
def test_public_dense_engine_aliases_are_supported(engine_name):
    engine = build_similarity_engine(X_ENGINE, engine=engine_name, similarity="linear")

    assert isinstance(engine, DenseSimilarityEngine)


@pytest.mark.parametrize("engine_name", ["blockwise", "chunkwise", "blocked", " BLOCKWISE ", "CHUNKWISE"])
def test_public_blockwise_engine_aliases_are_supported(engine_name):
    engine = build_similarity_engine(X_ENGINE, engine=engine_name, similarity="linear", block_size=2)

    assert isinstance(engine, BlockwiseSimilarityEngine)


@pytest.mark.parametrize("engine_name", ["unknown", "similarity"])
def test_invalid_similarity_engine_name_is_rejected(engine_name):
    with pytest.raises(ValueError):
        build_similarity_engine(X_ENGINE, engine=engine_name, similarity="linear")


@pytest.mark.parametrize("engine_name", ["", "   ", 123, None])
def test_missing_or_non_string_similarity_engine_name_is_rejected(engine_name):
    with pytest.raises(TypeError):
        build_similarity_engine(X_ENGINE, engine=engine_name, similarity="linear")


@pytest.mark.parametrize(
    "block_size, expected_error",
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError), (None, TypeError), (True, TypeError)],
)
def test_invalid_block_size_is_rejected_for_blockwise_engine(block_size, expected_error):
    with pytest.raises(expected_error):
        build_similarity_engine(
            X_ENGINE,
            engine="blockwise",
            block_size=block_size,
            similarity="linear",
        )


def test_similarity_engine_auto_backend_resolves_to_numpy():
    engine = build_similarity_engine(X_ENGINE, engine="blockwise", backend="auto", similarity="linear")

    assert engine.backend.name == "numpy"
    np.testing.assert_allclose(
        engine.to_dense(),
        build_similarity_matrix(X_ENGINE, similarity="linear"),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["unknown", "tensorflow"])
def test_similarity_engine_rejects_unknown_backend_alias(backend):
    with pytest.raises(ValueError):
        build_similarity_engine(X_ENGINE, engine="blockwise", backend=backend, similarity="linear")


@pytest.mark.parametrize("backend", ["", "   ", 123, None])
def test_similarity_engine_rejects_missing_or_non_string_backend_alias(backend):
    with pytest.raises(TypeError):
        build_similarity_engine(X_ENGINE, engine="blockwise", backend=backend, similarity="linear")


def test_similarity_engine_handles_empty_feature_matrix():
    X_empty = np.empty((0, 3), dtype=float)

    dense_engine = build_similarity_engine(X_empty, engine="dense", similarity="linear")
    blockwise_engine = build_similarity_engine(X_empty, engine="blockwise", block_size=2, similarity="linear")

    assert dense_engine.to_dense().shape == (0, 0)
    assert blockwise_engine.to_dense().shape == (0, 0)
    assert list(blockwise_engine.iter_blocks()) == []


def test_similarity_engine_handles_single_sample_matrix():
    X_single = np.array([[0.25, 0.75, 0.5]], dtype=float)

    dense_engine = build_similarity_engine(X_single, engine="dense", similarity="linear")
    blockwise_engine = build_similarity_engine(X_single, engine="blockwise", block_size=2, similarity="linear")

    np.testing.assert_allclose(dense_engine.to_dense(), np.array([[1.0]]), atol=1e-12)
    np.testing.assert_allclose(blockwise_engine.to_dense(), np.array([[1.0]]), atol=1e-12)



@pytest.mark.parametrize(
    "similarity_name, similarity_kwargs, tnorm_name",
    [
        ("linear", {}, "minimum"),
        ("linear", {}, "product"),
        ("linear", {}, "lukasiewicz"),
        ("gaussian", {"similarity_sigma": 0.5}, "minimum"),
        ("gaussian", {"similarity_sigma": 0.5}, "product"),
        ("gaussian", {"similarity_sigma": 0.5}, "lukasiewicz"),
    ],
)
def test_public_blockwise_engine_preserves_similarity_matrix_invariants(
    similarity_name,
    similarity_kwargs,
    tnorm_name,
):
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        similarity=similarity_name,
        similarity_tnorm=tnorm_name,
        **similarity_kwargs,
    )

    matrix = engine.to_dense()

    assert matrix.shape == (X_ENGINE.shape[0], X_ENGINE.shape[0])
    np.testing.assert_allclose(np.diag(matrix), np.ones(X_ENGINE.shape[0]), atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_public_dense_and_blockwise_engines_are_repeatable_across_calls():
    dense_engine = build_similarity_engine(X_ENGINE, engine="dense", similarity="linear")
    blockwise_engine = build_similarity_engine(X_ENGINE, engine="blockwise", block_size=2, similarity="linear")

    dense_first = dense_engine.to_dense()
    blockwise_first = blockwise_engine.to_dense()

    np.testing.assert_allclose(dense_engine.to_dense(), dense_first, atol=1e-12)
    np.testing.assert_allclose(blockwise_engine.to_dense(), blockwise_first, atol=1e-12)
    np.testing.assert_allclose(blockwise_first, dense_first, atol=1e-12)


def test_public_blockwise_engine_with_identical_samples_has_unit_cross_similarity():
    X = np.array([[0.2, 0.3], [0.2, 0.3], [0.9, 1.0]], dtype=float)

    matrix = build_similarity_engine(X, engine="blockwise", block_size=2, similarity="linear").to_dense()

    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(1.0)


def test_public_gaussian_similarity_sigma_controls_off_diagonal_strength():
    X = np.array([[0.0], [0.5]], dtype=float)

    small_sigma = build_similarity_engine(
        X,
        engine="blockwise",
        block_size=1,
        similarity="gaussian",
        similarity_sigma=0.1,
    ).to_dense()
    large_sigma = build_similarity_engine(
        X,
        engine="blockwise",
        block_size=1,
        similarity="gaussian",
        similarity_sigma=1.0,
    ).to_dense()

    assert small_sigma[0, 1] < large_sigma[0, 1]


def test_public_flat_kwargs_override_flat_config_mapping():
    config = {"similarity": "linear", "similarity_tnorm": "minimum"}

    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        config=config,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )
    expected = build_similarity_matrix(
        X_ENGINE,
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    np.testing.assert_allclose(engine.to_dense(), expected, atol=1e-12)
    assert config == {"similarity": "linear", "similarity_tnorm": "minimum"}


def test_public_similarity_blocks_have_exact_unit_diagonal_inside_diagonal_blocks():
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        similarity="gaussian",
        similarity_sigma=0.5,
    )

    for block in engine.iter_blocks():
        if block.row_slice == block.col_slice:
            np.testing.assert_allclose(np.diag(block.values), np.ones(block.values.shape[0]), atol=0.0)



def test_public_blockwise_cupy_backend_materializes_numpy_dense_matrix(monkeypatch):
    install_fake_cupy_module(monkeypatch)
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        backend="cupy",
        similarity="linear",
        similarity_tnorm="minimum",
    )

    actual = engine.to_dense()
    expected = build_similarity_matrix(X_ENGINE, similarity="linear", similarity_tnorm="minimum")

    assert isinstance(actual, np.ndarray)
    assert not isinstance(actual, FakeCupyArray)
    assert engine.backend.name == "cupy"
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_public_blockwise_cupy_iter_blocks_returns_numpy_blocks(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
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


def test_public_blockwise_cupy_iter_backend_blocks_returns_backend_resident_blocks(monkeypatch):
    fake_cupy = install_fake_cupy_module(monkeypatch)
    engine = build_similarity_engine(
        X_ENGINE,
        engine="blockwise",
        block_size=2,
        backend="cupy",
        similarity="gaussian",
        similarity_sigma=0.5,
        similarity_tnorm="product",
    )

    blocks = list(engine.iter_backend_blocks())

    assert len(fake_cupy.fill_diagonal_calls) == 3
    assert all(isinstance(block.values, FakeCupyArray) for block in blocks)
    assert all(block.values_backend == "cupy" for block in blocks)
    assert all(block.values_are_backend_resident is True for block in blocks)
    for block in blocks:
        if block.row_slice == block.col_slice:
            np.testing.assert_allclose(np.diag(np.asarray(block.values)), np.ones(block.values.shape[0]), atol=0.0)


def test_public_dense_engine_with_cupy_backend_keeps_numpy_block_contract(monkeypatch):
    install_fake_cupy_module(monkeypatch)
    engine = build_similarity_engine(X_ENGINE, engine="dense", backend="cupy", similarity="linear")

    blocks = list(engine.iter_backend_blocks())

    assert engine.backend.name == "cupy"
    assert len(blocks) == 1
    assert isinstance(blocks[0].values, np.ndarray)
    assert not isinstance(blocks[0].values, FakeCupyArray)
    assert blocks[0].values_backend == "numpy"
    assert blocks[0].values_are_backend_resident is False
