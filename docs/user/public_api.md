# frsutils public API

FRsutils exposes its stable user-facing and downstream-package API through the package root:

```python
from frsutils import compute_approximations
```

User examples, notebooks, tests, and downstream packages should prefer imports from `frsutils` instead of deep internal paths such as `frsutils.core` or `frsutils.utils`. Internal modules may change more often than the package-root public API.

## Purpose

FRsutils is the fuzzy-rough core package. It provides reusable fuzzy-rough building blocks and task-oriented helpers for:

- similarity-matrix construction,
- lower approximation,
- upper approximation,
- boundary-region computation,
- positive-region computation,
- reusable positive-region scoring workflows,
- fuzzy-rough model construction for advanced users and downstream packages.

## Main public entry points

The public API is organized around a small set of task-oriented names:

| Task                                                       | Public API                                                                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Compute lower, upper, boundary, and positive-region values | `compute_approximations`                                                                                           |
| Compute one approximation output                           | `compute_lower_approximation`, `compute_upper_approximation`, `compute_boundary_region`, `compute_positive_region` |
| Build a pairwise similarity matrix                         | `build_similarity_matrix`                                                                                          |
| Build a dense or blockwise similarity engine               | `build_similarity_engine`                                                                                          |
| Build a dense fuzzy-rough model object                     | `build_fuzzy_rough_model`                                                                                          |
| Use a scikit-learn-style positive-region scorer            | `FuzzyRoughPositiveRegionScorer`                                                                                   |

Preferred imports:

```python
from frsutils import (
    FuzzyRoughApproximationResult,
    FuzzyRoughPositiveRegionScorer,
    build_fuzzy_rough_model,
    build_similarity_engine,
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)
```

Supported fuzzy-rough model aliases are:

- `"itfrs"` for implicator/t-norm fuzzy-rough sets,
- `"vqrs"` for vaguely quantified rough sets,
- `"owafrs"` for ordered weighted averaging fuzzy-rough sets.

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

For a runnable example, see `examples/public_api_quickstart.py` in the project root.

## Computing approximations

`compute_approximations` is the main functional entry point:

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

It returns a `FuzzyRoughApproximationResult`, not a positional tuple. Downstream code should access named fields:

```python
result.lower
result.upper
result.boundary
result.positive_region
```

The current public contracts are:

```python
np.allclose(result.boundary, result.upper - result.lower)
np.allclose(result.positive_region, result.lower)
```

Public result arrays are always **NumPy arrays**, even when optional CuPy-backed blockwise execution is used internally.

### Dense execution

Dense execution builds or consumes a full pairwise similarity matrix and then uses the dense model implementation:

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

Blockwise execution computes exact approximations without materializing the full similarity matrix by default:

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

Blockwise mode requires `X` and does not accept a precomputed `similarity_matrix`. To inspect the matrix produced by blockwise execution, set
`return_similarity_matrix=True`:

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

## Result object and execution metadata

`FuzzyRoughApproximationResult` is the stable public result container.

```python
data = result.as_dict()
```

By default, `as_dict()` does not include `similarity_matrix`, because that matrix can be large. Include it explicitly when needed:

```python
data = result.as_dict(include_similarity_matrix=True)
```

Important metadata fields include:

| Field                                 | Meaning                                                                                                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `model`                               | Canonical fuzzy-rough model alias.                                                                           |
| `similarity`                          | Canonical similarity configuration or alias.                                                                 |
| `engine`                              | Canonical approximation engine, usually `"dense"` or `"blockwise"`.                                          |
| `backend`                             | Canonical resolved backend. <mark>Dense execution reports</mark> `"numpy"`.                                  |
| `block_size`                          | Positive integer for blockwise execution; `None` for dense execution.                                        |
| `used_blockwise`                      | Whether blockwise approximation execution was used.                                                          |
| `used_gpu_similarity_blocks`          | Whether similarity blocks used the optional CuPy backend.                                                    |
| `used_gpu_approximation_accumulators` | Whether model-specific approximation accumulators stayed CuPy-resident before final NumPy output conversion. |

**NOTE**: The standard dense path uses NumPy and reports the backend as "numpy". CuPy may be used only by supported backend-aware paths when explicitly requested and available.

The same above fields are included in `FuzzyRoughApproximationResult.as_dict()`.

## Similarity API

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

`build_fuzzy_rough_model` constructs dense model objects from a precomputed similarity matrix and labels:

```python
from frsutils import build_fuzzy_rough_model, build_similarity_matrix

S = build_similarity_matrix(X, similarity="linear")
model = build_fuzzy_rough_model(
    model_type="itfrs",
    similarity_matrix=S,
    labels=y,
)

lower = model.lower_approximation()
```

This builder supports the same model aliases as `compute_approximations`:
`"itfrs"`, `"vqrs"`, and `"owafrs"`.

## Positive-region scorer

`FuzzyRoughPositiveRegionScorer` provides a reusable estimator-like API for fuzzy-rough positive-region scores:

```python
from frsutils import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    backend="numpy",
    block_size=512,
)

scores = scorer.fit_score(X, y)
result = scorer.as_result()
```

The scorer exposes `fit`, `fit_score`, `score_samples`, `as_result`,
`get_params`, and `set_params`. It follows scikit-learn estimator conventions for parameter handling, cloning, and grid-search-style workflows.

The scorer computes scores for fitted training samples. It does not yet score unseen samples, because the current fuzzy-rough approximation models are based
on a fitted pairwise similarity matrix.

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

Internally, FRsutils may normalize these flat parameters into a nested component configuration. Nested configuration is also accepted at the public API boundary for advanced users:

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

## Advanced and semi-internal exports

The following names are public but intended mainly for advanced users, experiments, and downstream packages:

```python
from frsutils import (
    build_fuzzy_rough_model,
    get_fuzzy_rough_model_class,
    list_fuzzy_rough_models,
    list_similarities,
    normalize_flat_config_to_nested,
)
```

Some low-level classes and helpers may also be importable from `frsutils` for inspection, backwards compatibility, or advanced experimentation:

- `Similarity`
- `FuzzyRoughModel`
- `ITFRS`
- `OWAFRS`
- `VQRS`
- `apply_config_aliases`
- `extract_prefixed_params`
- `calculate_similarity_matrix`

These should not be the first choice for tutorials, README examples, or external package integration. Prefer the task-level API unless there is a clear advanced use case.

## Downstream-package rule

Downstream packages should depend on `frsutils` only:

```python
from frsutils import build_similarity_matrix, compute_positive_region
```

Avoid relying on internal modules in downstream user code:

```python
# Avoid this in user-facing examples and downstream packages.
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.similarities import build_similarity_matrix
from frsutils.utils.init_helpers import normalize_flat_config_to_nested
```

The `frsutils` namespace is the compatibility boundary. Internal modules may change between releases.

## Backend behavior

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

Regardless of the backend, public outputs remain NumPy arrays for compatibility with scientific Python and scikit-learn-style workflows. See [Backends](backends.md) for the detailed model-specific backend contract.

## Minimum public contract tests

The repository should keep tests that verify:

1. public imports work from `frsutils`,
2. `compute_approximations` returns a named result object,
3. `compute_positive_region` matches `compute_approximations(...).positive_region`,
4. A precomputed similarity matrix can feed approximation helpers,
5. downstream-style code can use only `frsutils` without importing internals.
