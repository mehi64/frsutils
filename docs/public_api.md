# frsutils public API

frsutils exposes its stable user-facing API through the package root, `frsutils`. Downstream code, examples, notebooks, and documentation should import from this namespace instead of importing directly from internal `frsutils.core` modules.

```python
from frsutils import compute_approximations
```

The package root, `frsutils`, exposes the intended stable public objects while keeping internal implementation details out of the public contract.

## Main capabilities

The public API is organized around a small set of task-oriented entry points:

| Task                                                       | Public API                                                                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Compute lower, upper, boundary, and positive-region values | `compute_approximations`                                                                                           |
| Compute only one approximation output                      | `compute_lower_approximation`, `compute_upper_approximation`, `compute_boundary_region`, `compute_positive_region` |
| Build a pairwise similarity matrix                         | `build_similarity_matrix`                                                                                          |
| Build a dense or blockwise similarity engine               | `build_similarity_engine`                                                                                          |
| Build a dense fuzzy-rough model object                     | `build_fuzzy_rough_model`                                                                                          |
| Use a scikit-learn-style positive-region scorer            | `FuzzyRoughPositiveRegionScorer`                                                                                   |

Supported fuzzy-rough model aliases are:

- `"itfrs"` for implicator/t-norm fuzzy rough sets.
- `"vqrs"` for vaguely quantified rough sets.
- `"owafrs"` for ordered weighted averaging fuzzy rough sets.

## Quick start

```python
import numpy as np
from frsutils import compute_approximations

X = np.array(
    [
        [0.00, 0.10],
        [0.08, 0.18],
        [0.15, 0.12],
        [0.80, 0.82],
        [0.88, 0.90],
        [0.95, 0.86],
    ],
    dtype=float,
)
y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
)

print(result.positive_region)
```

For a runnable example, see `examples/public_api_quickstart.py`.

## Computing approximations

`compute_approximations` is the main functional entry point.

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="vqrs",
    similarity="linear",
    engine="dense",
)
```

It returns a `FuzzyRoughApproximationResult` with these arrays:

- `lower`
- `upper`
- `boundary`
- `positive_region`

For the current public contracts:

```python
np.allclose(result.boundary, result.upper - result.lower)
np.allclose(result.positive_region, result.lower)
```

Public result arrays are always returned as NumPy arrays, even when an optional CuPy-backed blockwise path is used internally.

### Dense execution

Dense execution builds or consumes a full pairwise similarity matrix and then uses the dense model implementation.

```python
result = compute_approximations(
    X,
    y,
    model="owafrs",
    similarity="linear",
    engine="dense",
)
```

A precomputed similarity matrix can be supplied in dense mode:

```python
from frsutils import build_similarity_matrix, compute_approximations

S = build_similarity_matrix(X, similarity="linear")
result = compute_approximations(
    None,
    y,
    model="itfrs",
    similarity_matrix=S,
    engine="dense",
)
```

### Blockwise execution

Blockwise execution computes exact approximations without materializing the full similarity matrix by default.

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    block_size=256,
    backend="numpy",
)
```

Blockwise mode requires `X` and does not accept a precomputed `similarity_matrix`. To inspect the matrix produced by blockwise execution, set `return_similarity_matrix=True`.

```python
result = compute_approximations(
    X,
    y,
    model="vqrs",
    similarity="linear",
    engine="blockwise",
    block_size=256,
    return_similarity_matrix=True,
)

S = result.similarity_matrix
```

## Approximation wrapper functions

Wrapper functions are available when only one output is needed:

```python
from frsutils import (
    compute_lower_approximation,
    compute_upper_approximation,
    compute_boundary_region,
    compute_positive_region,
)

positive = compute_positive_region(X, y, model="itfrs", similarity="linear")
```

These wrappers delegate to `compute_approximations` and return plain NumPy arrays, not result objects.

## Result object and metadata

`FuzzyRoughApproximationResult` is the stable public result container.

```python
data = result.as_dict()
```

By default, `as_dict()` does not include `similarity_matrix`, because that matrix can be large. Include it explicitly when needed:

```python
data = result.as_dict(include_similarity_matrix=True)
```

Important metadata fields include:

- `model`
- `similarity`
- `engine`
- `backend`
- `block_size`
- `used_blockwise`
- `used_gpu_similarity_blocks`
- `used_gpu_approximation_accumulators`

## Similarity public API

Use `build_similarity_matrix` for ordinary dense pairwise matrices:

```python
from frsutils import build_similarity_matrix

S = build_similarity_matrix(X, similarity="linear")
```

Use `build_similarity_engine` when blockwise iteration is needed:

```python
from frsutils import build_similarity_engine

engine = build_similarity_engine(
    X,
    engine="blockwise",
    block_size=256,
    similarity="linear",
)

for block in engine.iter_blocks():
    print(block.row_slice, block.col_slice, block.values.shape)
```

## Building dense model objects

`build_fuzzy_rough_model` constructs dense model objects from a precomputed similarity matrix and labels.

```python
from frsutils import build_fuzzy_rough_model, build_similarity_matrix

S = build_similarity_matrix(X, similarity="linear")
model = build_fuzzy_rough_model(
    "itfrs",
    similarity_matrix=S,
    labels=y,
)

lower = model.lower_approximation()
```

This builder supports the same model aliases as `compute_approximations`: `"itfrs"`, `"vqrs"`, and `"owafrs"`.

## Scikit-learn-style positive-region scorer

`FuzzyRoughPositiveRegionScorer` provides an estimator-like interface with `fit`, `fit_score`, `score_samples`, `as_result`, `get_params`, and
`set_params`.

```python
from frsutils import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(
    model="owafrs",
    similarity="linear",
    engine="dense",
)

scores = scorer.fit_score(X, y)
result = scorer.as_result()
```

The scorer follows scikit-learn estimator conventions for parameter handling and cloning. It computes scores for the fitted samples; it is not a predictor for unseen samples.

## Flat and nested configuration

The public API accepts flat, sklearn-friendly parameters:

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="gaussian",
    similarity_sigma=0.5,
    ub_tnorm_name="minimum",
    lb_implicator_name="lukasiewicz",
)
```

Internally, frsutils may normalize these flat parameters into a nested component configuration. Nested configuration is also accepted at the public API boundary:

```python
config = {
    "similarity": {"name": "linear", "params": {}},
    "similarity_tnorm": {"name": "minimum", "params": {}},
    "fr_model": {
        "type": "itfrs",
        "ub_tnorm": {"name": "minimum", "params": {}},
        "lb_implicator": {"name": "lukasiewicz", "params": {}},
    },
}

result = compute_approximations(X, y, config=config)
```

Do not mix nested config with additional flat keyword parameters in the same call. This keeps the public API unambiguous.

## Backend and CuPy behavior

The stable backend is NumPy. CuPy support is optional and only used through explicit blockwise execution:

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    backend="cupy",
)
```

Backend claims are model-specific:

- ITFRS: blockwise execution can use CuPy-backed similarity blocks and GPU-resident approximation accumulators.
- VQRS: blockwise execution can use CuPy-backed similarity blocks and GPU-resident approximation accumulators.
- OWAFRS: blockwise execution can use CuPy-backed similarity blocks, but does not currently claim GPU-resident OWAFRS approximation accumulators.

Regardless of backend, public outputs remain NumPy arrays for compatibility with scientific Python and scikit-learn-style workflows. See [`cupy_info.md`](cupy_info.md), [`backend_execution_status.md`](backend_execution_status.md), and
[`owafrs_non_gpu_resident_decision.md`](owafrs_non_gpu_resident_decision.md) for the detailed backend contract and OWAFRS decision record.

## Recommended imports

Recommended:

```python
from frsutils import compute_approximations
from frsutils import FuzzyRoughPositiveRegionScorer
```

Avoid relying on internal modules in downstream user code:

```python
# Avoid this in user-facing examples and downstream packages.
from frsutils.core.models.itfrs import ITFRS
```

Internal modules may change more often than the canonical public API namespace.

## JOSS and release documentation

For release notes, software-paper wording, and benchmark claims, use the conservative wording in `paper_claims.md`. Before tagging or submitting, check `release_checklist.md`, `release_validation_commands.md`, `documentation_smoke_check.md`, and `joss_metadata_check.md`.
