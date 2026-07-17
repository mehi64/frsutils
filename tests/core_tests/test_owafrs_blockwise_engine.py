# SPDX-License-Identifier: BSD-3-Clause
"""Direct parity and contract tests for the blockwise OWAFRS engine."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from frsutils.core.approximation_engines import (
    OWAFRSBlockwiseApproximation,
    compute_owafrs_blockwise,
)
from frsutils.core.implicators import Implicator
from frsutils.core.models.owafrs import OWAFRS
from frsutils.core.owa_weights import OWAWeights
from frsutils.core.similarity_engine import BaseSimilarityEngine, SimilarityBlock
from frsutils.core.tnorms import TNorm


LOGGER = logging.getLogger("frsutils.tests.owafrs_blockwise")

OWAFRS_RELATION = np.array(
    [
        [0.72, 0.88, 0.41, 0.16, 0.27, 0.34],
        [0.79, 0.81, 0.36, 0.24, 0.19, 0.12],
        [0.46, 0.31, 0.69, 0.77, 0.28, 0.21],
        [0.18, 0.29, 0.68, 0.91, 0.63, 0.49],
        [0.25, 0.17, 0.22, 0.57, 0.83, 0.86],
        [0.33, 0.14, 0.19, 0.52, 0.74, 0.76],
    ],
    dtype=float,
)
OWAFRS_LABELS = np.array(["cold", "cold", "warm", "warm", "hot", "hot"], dtype=object)


class FixedBlockSimilarityEngine(BaseSimilarityEngine):
    """Yield a fixed fuzzy relation in deterministic row-major blocks."""

    def __init__(self, relation: np.ndarray, *, block_size: int) -> None:
        """Store the relation and requested block size without rebuilding it."""
        self.relation = np.asarray(relation, dtype=float)
        self.block_size = block_size
        self.backend = None
        self.X = np.zeros((self.relation.shape[0], 1), dtype=float)

    @property
    def n_samples(self) -> int:
        """Return the relation order."""
        return self.relation.shape[0]

    def iter_blocks(self):
        """Yield all relation blocks in row-major order."""
        for row_start in range(0, self.n_samples, self.block_size):
            row_slice = slice(row_start, min(row_start + self.block_size, self.n_samples))
            for col_start in range(0, self.n_samples, self.block_size):
                col_slice = slice(col_start, min(col_start + self.block_size, self.n_samples))
                yield SimilarityBlock(
                    row_slice=row_slice,
                    col_slice=col_slice,
                    values=self.relation[row_slice, col_slice],
                )


def _build_tnorm(name: str) -> TNorm:
    """Build a registered T-norm with a valid Yager parameter when needed."""
    return TNorm.create(name, p=1.7)


def _build_implicator(name: str) -> Implicator:
    """Build a registered implicator by canonical name."""
    return Implicator.create(name)


def _build_owa(name: str, *, base: float) -> OWAWeights:
    """Build a registered OWA strategy with an exponential base when needed."""
    return OWAWeights.create(name, base=base)


def _component_config(
    ub_tnorm: TNorm,
    lb_implicator: Implicator,
    ub_owa_method: OWAWeights,
    lb_owa_method: OWAWeights,
) -> dict:
    """Return a nested OWAFRS config containing direct component instances."""
    return {
        "fr_model": {
            "type": "owafrs",
            "ub_tnorm": ub_tnorm,
            "lb_implicator": lb_implicator,
            "ub_owa_method": ub_owa_method,
            "lb_owa_method": lb_owa_method,
        }
    }


def _dense_owafrs_outputs(
    relation: np.ndarray,
    labels: np.ndarray,
    ub_tnorm: TNorm,
    lb_implicator: Implicator,
    ub_owa_method: OWAWeights,
    lb_owa_method: OWAWeights,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all dense OWAFRS reference outputs for one component set."""
    model = OWAFRS(
        relation,
        labels,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
        logger=LOGGER,
    )
    lower = model.lower_approximation()
    upper = model.upper_approximation()
    return lower, upper, upper - lower, lower.copy()


def _assert_matches_dense(
    result: OWAFRSBlockwiseApproximation,
    expected: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Assert equality of every blockwise OWAFRS output with dense reference values."""
    expected_lower, expected_upper, expected_boundary, expected_positive = expected
    np.testing.assert_allclose(result.lower, expected_lower, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.upper, expected_upper, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.boundary, expected_boundary, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result.positive_region,
        expected_positive,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.boundary, result.upper - result.lower, atol=1e-12)
    np.testing.assert_allclose(result.positive_region, result.lower, atol=1e-12)


def _deterministic_relation(n_samples: int) -> np.ndarray:
    """Return a finite asymmetric relation with a deliberately non-unit diagonal."""
    rng = np.random.default_rng(10_000 + n_samples)
    relation = rng.uniform(0.05, 0.95, size=(n_samples, n_samples))
    np.fill_diagonal(relation, rng.uniform(0.55, 0.95, size=n_samples))
    return relation


@pytest.mark.parametrize("block_size", [1, 2, 4, 10])
def test_compute_owafrs_blockwise_matches_dense_for_multiple_block_sizes(block_size):
    """Direct blockwise execution should match dense OWAFRS for every result field."""
    ub_tnorm = _build_tnorm("minimum")
    lb_implicator = _build_implicator("lukasiewicz")
    ub_owa_method = _build_owa("linear", base=2.0)
    lb_owa_method = _build_owa("linear", base=2.0)
    expected = _dense_owafrs_outputs(
        OWAFRS_RELATION,
        OWAFRS_LABELS,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )
    engine = FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=block_size)

    result = compute_owafrs_blockwise(
        engine,
        OWAFRS_LABELS,
        config=_component_config(
            ub_tnorm,
            lb_implicator,
            ub_owa_method,
            lb_owa_method,
        ),
    )

    assert isinstance(result, OWAFRSBlockwiseApproximation)
    _assert_matches_dense(result, expected)


def test_compute_owafrs_blockwise_resolves_custom_flat_configuration():
    """Direct blockwise execution should honor flat component selectors and parameters."""
    config = {
        "type": "owafrs",
        "ub_tnorm_name": "yager",
        "ub_tnorm_p": 1.7,
        "lb_implicator_name": "goedel",
        "ub_owa_method_name": "exponential",
        "ub_owa_method_base": 1.35,
        "lb_owa_method_name": "harmonic",
    }
    ub_tnorm = _build_tnorm("yager")
    lb_implicator = _build_implicator("goedel")
    ub_owa_method = _build_owa("exponential", base=1.35)
    lb_owa_method = _build_owa("harmonic", base=2.0)
    expected = _dense_owafrs_outputs(
        OWAFRS_RELATION,
        OWAFRS_LABELS,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    result = compute_owafrs_blockwise(
        FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=2),
        OWAFRS_LABELS,
        config=config,
    )

    _assert_matches_dense(result, expected)


@pytest.mark.parametrize(
    "labels",
    [
        np.array([10, 10, 20, 20, 30, 30], dtype=int),
        np.array(["a", "a", "b", "b", "c", "c"], dtype=str),
        np.array(["a", 1, "a", 1, None, None], dtype=object),
    ],
)
def test_compute_owafrs_blockwise_supports_numeric_string_and_object_labels(labels):
    """Direct blockwise OWAFRS should preserve dense behavior for common label types."""
    ub_tnorm = _build_tnorm("product")
    lb_implicator = _build_implicator("kleenedienes")
    ub_owa_method = _build_owa("harmonic", base=2.0)
    lb_owa_method = _build_owa("exponential", base=1.4)
    expected = _dense_owafrs_outputs(
        OWAFRS_RELATION,
        labels,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    result = compute_owafrs_blockwise(
        FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=3),
        labels,
        config=_component_config(
            ub_tnorm,
            lb_implicator,
            ub_owa_method,
            lb_owa_method,
        ),
    )

    _assert_matches_dense(result, expected)


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["same"] * 6, dtype=object),
        np.array(["a", "b", "c", "d", "e", "f"], dtype=object),
    ],
)
def test_compute_owafrs_blockwise_handles_degenerate_label_partitions(labels):
    """All-same and all-singleton class partitions should remain finite and exact."""
    ub_tnorm = _build_tnorm("minimum")
    lb_implicator = _build_implicator("lukasiewicz")
    ub_owa_method = _build_owa("linear", base=2.0)
    lb_owa_method = _build_owa("harmonic", base=2.0)
    expected = _dense_owafrs_outputs(
        OWAFRS_RELATION,
        labels,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    result = compute_owafrs_blockwise(
        FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=2),
        labels,
        config=_component_config(
            ub_tnorm,
            lb_implicator,
            ub_owa_method,
            lb_owa_method,
        ),
    )

    _assert_matches_dense(result, expected)
    assert np.all(np.isfinite(result.lower))
    assert np.all(np.isfinite(result.upper))


@pytest.mark.parametrize("n_samples", [2, 3, 5, 8, 11])
def test_compute_owafrs_blockwise_matches_dense_across_dataset_sizes(n_samples):
    """Direct parity should hold across small and non-block-aligned sample counts."""
    relation = _deterministic_relation(n_samples)
    labels = np.array([f"class-{index % 3}" for index in range(n_samples)], dtype=object)
    ub_tnorm = _build_tnorm("hamacher")
    lb_implicator = _build_implicator("reichenbach")
    ub_owa_method = _build_owa("exponential", base=1.2)
    lb_owa_method = _build_owa("linear", base=2.0)
    expected = _dense_owafrs_outputs(
        relation,
        labels,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )
    config = _component_config(
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    for block_size in (1, 2, n_samples, n_samples + 3):
        result = compute_owafrs_blockwise(
            FixedBlockSimilarityEngine(relation, block_size=block_size),
            labels,
            config=config,
        )
        _assert_matches_dense(result, expected)


def test_compute_owafrs_blockwise_does_not_mutate_relation_or_labels():
    """Blockwise sorting and diagonal removal must not alter caller-owned inputs."""
    relation = OWAFRS_RELATION.copy()
    labels = OWAFRS_LABELS.copy()
    relation_before = relation.copy()
    labels_before = labels.copy()

    compute_owafrs_blockwise(
        FixedBlockSimilarityEngine(relation, block_size=2),
        labels,
        config={"type": "owafrs"},
    )

    np.testing.assert_array_equal(relation, relation_before)
    np.testing.assert_array_equal(labels, labels_before)


def test_compute_owafrs_blockwise_supports_exponential_weights_above_twenty_comparisons():
    """Dense and blockwise OWAFRS should support exponential OWA on realistic sizes."""
    n_samples = 64
    relation = _deterministic_relation(n_samples)
    labels = np.array([index % 4 for index in range(n_samples)], dtype=int)
    ub_tnorm = _build_tnorm("minimum")
    lb_implicator = _build_implicator("lukasiewicz")
    ub_owa_method = _build_owa("exponential", base=1.3)
    lb_owa_method = _build_owa("exponential", base=1.7)
    expected = _dense_owafrs_outputs(
        relation,
        labels,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    result = compute_owafrs_blockwise(
        FixedBlockSimilarityEngine(relation, block_size=7),
        labels,
        config=_component_config(
            ub_tnorm,
            lb_implicator,
            ub_owa_method,
            lb_owa_method,
        ),
    )

    _assert_matches_dense(result, expected)
    assert np.all(np.isfinite(result.lower))
    assert np.all(np.isfinite(result.upper))


def test_compute_owafrs_blockwise_rejects_invalid_engine():
    """The direct engine boundary should reject non-engine objects clearly."""
    with pytest.raises(TypeError, match="BaseSimilarityEngine"):
        compute_owafrs_blockwise(object(), OWAFRS_LABELS)


@pytest.mark.parametrize(
    "labels, match",
    [
        (OWAFRS_LABELS[:-1], "Length of labels"),
        (OWAFRS_LABELS.reshape(2, 3), "1D"),
    ],
)
def test_compute_owafrs_blockwise_rejects_invalid_label_shapes(labels, match):
    """The direct engine boundary should reject misaligned or non-vector labels."""
    engine = FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=2)

    with pytest.raises(ValueError, match=match):
        compute_owafrs_blockwise(engine, labels)


@pytest.mark.slow
@pytest.mark.parametrize("tnorm_name", list(TNorm.list_available()))
@pytest.mark.parametrize("implicator_name", list(Implicator.list_available()))
@pytest.mark.parametrize("upper_owa_name", list(OWAWeights.list_available()))
@pytest.mark.parametrize("lower_owa_name", list(OWAWeights.list_available()))
def test_all_registered_owafrs_components_match_dense_blockwise(
    tnorm_name,
    implicator_name,
    upper_owa_name,
    lower_owa_name,
):
    """Every canonical OWAFRS component combination should preserve exact parity."""
    ub_tnorm = _build_tnorm(tnorm_name)
    lb_implicator = _build_implicator(implicator_name)
    ub_owa_method = _build_owa(upper_owa_name, base=1.3)
    lb_owa_method = _build_owa(lower_owa_name, base=1.7)
    expected = _dense_owafrs_outputs(
        OWAFRS_RELATION,
        OWAFRS_LABELS,
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )
    config = _component_config(
        ub_tnorm,
        lb_implicator,
        ub_owa_method,
        lb_owa_method,
    )

    for block_size in (1, 2, 4, 10):
        result = compute_owafrs_blockwise(
            FixedBlockSimilarityEngine(OWAFRS_RELATION, block_size=block_size),
            OWAFRS_LABELS,
            config=config,
        )
        _assert_matches_dense(result, expected)
