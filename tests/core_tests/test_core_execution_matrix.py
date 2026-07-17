# SPDX-License-Identifier: BSD-3-Clause
"""Exhaustive dense/blockwise parity matrix for public core execution."""

from __future__ import annotations

from itertools import product
import logging
from typing import Any

import numpy as np
import pytest

from frsutils import compute_approximations


X_EXECUTION_MATRIX = np.array(
    [
        [0.00, 0.20, 0.40],
        [0.15, 0.10, 0.35],
        [0.35, 0.40, 0.20],
        [0.65, 0.70, 0.80],
        [0.82, 0.75, 0.68],
        [0.95, 0.90, 0.85],
    ],
    dtype=float,
)
Y_EXECUTION_MATRIX = np.array(
    ["class-a", "class-a", "class-b", "class-b", "class-c", "class-c"],
    dtype=object,
)

SIMILARITIES = ("linear", "gaussian")
TNORMS = (
    "minimum",
    "product",
    "lukasiewicz",
    "drastic",
    "einstein",
    "hamacher",
    "nilpotent",
    "yager",
)
IMPLICATORS = (
    "lukasiewicz",
    "goedel",
    "kleenedienes",
    "reichenbach",
    "goguen",
    "rescher",
    "yager",
    "weber",
    "fodor",
)
OWA_METHODS = ("linear", "exponential", "harmonic")
FUZZY_QUANTIFIERS = ("linear", "quadratic")


def _similarity_kwargs(similarity: str, similarity_tnorm: str) -> dict[str, Any]:
    """Return a valid flat similarity configuration for one canonical pair."""
    config: dict[str, Any] = {
        "similarity": similarity,
        "similarity_tnorm": similarity_tnorm,
    }
    if similarity == "gaussian":
        config["similarity_sigma"] = 0.37
    if similarity_tnorm == "yager":
        config["similarity_tnorm_p"] = 1.7
    return config


def _tnorm_kwargs(prefix: str, name: str) -> dict[str, Any]:
    """Return a flat T-norm selector and any required Yager parameter."""
    config: dict[str, Any] = {f"{prefix}_tnorm_name": name}
    if name == "yager":
        config[f"{prefix}_tnorm_p"] = 1.7
    return config


def _owa_kwargs(prefix: str, name: str, *, base: float) -> dict[str, Any]:
    """Return a flat OWA selector and any required exponential base."""
    config: dict[str, Any] = {f"{prefix}_owa_method_name": name}
    if name == "exponential":
        config[f"{prefix}_owa_method_base"] = base
    return config


def _assert_dense_blockwise_parity(config: dict[str, Any]) -> None:
    """Assert parity of every public approximation field for one configuration."""
    dense = compute_approximations(
        X_EXECUTION_MATRIX,
        Y_EXECUTION_MATRIX,
        engine="dense",
        **config,
    )
    blockwise = compute_approximations(
        X_EXECUTION_MATRIX,
        Y_EXECUTION_MATRIX,
        engine="blockwise",
        backend="numpy",
        block_size=2,
        **config,
    )

    for field in ("lower", "upper", "boundary", "positive_region"):
        np.testing.assert_allclose(
            getattr(blockwise, field),
            getattr(dense, field),
            rtol=1e-11,
            atol=1e-12,
            err_msg=f"Dense/blockwise mismatch for {field}: {config!r}",
        )

    assert dense.backend == "numpy"
    assert blockwise.backend == "numpy"
    assert dense.used_blockwise is False
    assert blockwise.used_blockwise is True


@pytest.mark.slow
def test_itfrs_execution_matrix_matches_dense_and_blockwise() -> None:
    """Check every canonical ITFRS execution configuration for NumPy parity."""
    logging.disable(logging.CRITICAL)
    try:
        for similarity, similarity_tnorm in product(SIMILARITIES, TNORMS):
            similarity_config = _similarity_kwargs(similarity, similarity_tnorm)
            for upper_tnorm, lower_implicator in product(TNORMS, IMPLICATORS):
                config = {
                    **similarity_config,
                    **_tnorm_kwargs("ub", upper_tnorm),
                    "model": "itfrs",
                    "lb_implicator_name": lower_implicator,
                }
                _assert_dense_blockwise_parity(config)
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.slow
def test_vqrs_execution_matrix_matches_dense_and_blockwise() -> None:
    """Check every canonical VQRS execution configuration for NumPy parity."""
    logging.disable(logging.CRITICAL)
    try:
        for similarity, similarity_tnorm in product(SIMILARITIES, TNORMS):
            similarity_config = _similarity_kwargs(similarity, similarity_tnorm)
            for lower_quantifier, upper_quantifier in product(
                FUZZY_QUANTIFIERS,
                FUZZY_QUANTIFIERS,
            ):
                config = {
                    **similarity_config,
                    "model": "vqrs",
                    "lb_fuzzy_quantifier_name": lower_quantifier,
                    "lb_fuzzy_quantifier_alpha": 0.05,
                    "lb_fuzzy_quantifier_beta": 0.73,
                    "ub_fuzzy_quantifier_name": upper_quantifier,
                    "ub_fuzzy_quantifier_alpha": 0.12,
                    "ub_fuzzy_quantifier_beta": 0.88,
                }
                _assert_dense_blockwise_parity(config)
    finally:
        logging.disable(logging.NOTSET)


def _representative_owafrs_component_configs() -> tuple[dict[str, Any], ...]:
    """Return a balanced cross-layer sample of canonical OWAFRS components."""
    configs = []
    for index in range(9):
        upper_tnorm = TNORMS[index % len(TNORMS)]
        lower_implicator = IMPLICATORS[index % len(IMPLICATORS)]
        upper_owa = OWA_METHODS[(index // len(OWA_METHODS)) % len(OWA_METHODS)]
        lower_owa = OWA_METHODS[index % len(OWA_METHODS)]
        configs.append(
            {
                **_tnorm_kwargs("ub", upper_tnorm),
                "lb_implicator_name": lower_implicator,
                **_owa_kwargs("ub", upper_owa, base=2.3),
                **_owa_kwargs("lb", lower_owa, base=1.8),
            }
        )
    return tuple(configs)


OWAFRS_COMPONENT_CONFIGS = _representative_owafrs_component_configs()


@pytest.mark.slow
def test_owafrs_cross_layer_execution_matrix_matches_dense_and_blockwise() -> None:
    """Check balanced canonical OWAFRS cross-layer configurations for NumPy parity."""
    logging.disable(logging.CRITICAL)
    try:
        for similarity, similarity_tnorm in product(SIMILARITIES, TNORMS):
            similarity_config = _similarity_kwargs(similarity, similarity_tnorm)
            for model_config in OWAFRS_COMPONENT_CONFIGS:
                config = {
                    **similarity_config,
                    **model_config,
                    "model": "owafrs",
                }
                _assert_dense_blockwise_parity(config)
    finally:
        logging.disable(logging.NOTSET)

