# SPDX-License-Identifier: BSD-3-Clause
"""Real-CUDA parity matrix for public fuzzy-rough approximation models."""

import numpy as np
import pytest

from frsutils import compute_approximations
from tests._cupy_test_support import require_usable_cupy


X_REAL_CUPY = np.asarray(
    [
        [0.00, 0.00, 0.10],
        [0.08, 0.18, 0.12],
        [0.24, 0.10, 0.22],
        [0.52, 0.62, 0.58],
        [0.68, 0.72, 0.66],
        [0.84, 0.77, 0.81],
        [0.93, 0.88, 0.95],
        [1.00, 0.82, 0.91],
    ],
    dtype=np.float64,
)
Y_REAL_CUPY = np.asarray(["cold", "cold", "cold", "warm", "warm", "warm", "hot", "hot"])

MODEL_CASES = [
    pytest.param("itfrs", {"similarity": "linear"}, True, id="itfrs-default"),
    pytest.param(
        "itfrs",
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "yager",
            "ub_tnorm_p": 1.7,
            "lb_implicator_name": "lukasiewicz",
        },
        True,
        id="itfrs-custom-components",
    ),
    pytest.param("vqrs", {"similarity": "linear"}, True, id="vqrs-default"),
    pytest.param(
        "vqrs",
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "lb_fuzzy_quantifier_name": "linear",
            "lb_fuzzy_quantifier_alpha": 0.2,
            "lb_fuzzy_quantifier_beta": 1.0,
            "ub_fuzzy_quantifier_name": "quadratic",
            "ub_fuzzy_quantifier_alpha": 0.0,
            "ub_fuzzy_quantifier_beta": 0.6,
        },
        True,
        id="vqrs-custom-quantifiers",
    ),
    pytest.param("owafrs", {"similarity": "linear"}, False, id="owafrs-default"),
    pytest.param(
        "owafrs",
        {
            "similarity": "gaussian",
            "similarity_sigma": 0.35,
            "similarity_tnorm": "minimum",
            "ub_tnorm_name": "yager",
            "ub_tnorm_p": 1.7,
            "lb_implicator_name": "lukasiewicz",
            "lb_owa_method_name": "harmonic",
            "ub_owa_method_name": "exponential",
            "ub_owa_method_base": 1.3,
        },
        False,
        id="owafrs-custom-components",
    ),
]


@pytest.mark.parametrize("block_size", [1, 3, 16])
@pytest.mark.parametrize("model, config, expected_gpu_accumulators", MODEL_CASES)
def test_real_cupy_blockwise_parity_matrix(
    model,
    config,
    expected_gpu_accumulators,
    block_size,
):
    """Real CuPy blockwise results must match dense NumPy across model cases."""
    require_usable_cupy()

    dense = compute_approximations(
        X_REAL_CUPY,
        Y_REAL_CUPY,
        model=model,
        engine="dense",
        backend="numpy",
        **config,
    )
    gpu = compute_approximations(
        X_REAL_CUPY,
        Y_REAL_CUPY,
        model=model,
        engine="blockwise",
        backend="cupy",
        block_size=block_size,
        **config,
    )

    assert gpu.backend == "cupy"
    assert gpu.used_blockwise is True
    assert gpu.used_gpu_similarity_blocks is True
    assert gpu.used_gpu_approximation_accumulators is expected_gpu_accumulators
    assert all(
        isinstance(value, np.ndarray)
        for value in (gpu.lower, gpu.upper, gpu.boundary, gpu.positive_region)
    )
    np.testing.assert_allclose(gpu.lower, dense.lower, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(gpu.upper, dense.upper, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(gpu.boundary, dense.boundary, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(gpu.positive_region, dense.positive_region, atol=1e-12, rtol=0.0)
