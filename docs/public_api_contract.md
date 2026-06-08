# FRsutils Public API Contract

FRsutils exposes its stable user-facing and downstream-package API through:

```python
from FRsutils.api import ...
```

This document defines the public compatibility boundary for users, examples,
tests, and downstream packages such as `frsampling`.

## Purpose

FRsutils is the fuzzy-rough core package. It provides reusable fuzzy-rough
building blocks and task-oriented helpers for:

- similarity-matrix construction,
- lower approximation,
- upper approximation,
- boundary-region computation,
- positive-region computation,
- reusable positive-region scoring workflows,
- fuzzy-rough model construction for advanced users and downstream packages.

Oversampling algorithms such as FRSMOTE are intentionally outside FRsutils core
and should live in downstream packages such as `frsampling`.

## Stable task-level API

These names are the preferred public API for normal users and documentation
examples:

```python
from FRsutils.api import (
    FuzzyRoughApproximationResult,
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_approximations,
    compute_boundary_region,
    compute_lower_approximation,
    compute_positive_region,
    compute_upper_approximation,
)
```

### `compute_approximations`

Use this when users need the complete fuzzy-rough approximation result.

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
)

result.lower
result.upper
result.boundary
result.positive_region
```

The return value is a `FuzzyRoughApproximationResult`, not a positional tuple.
Downstream code should access named fields instead of relying on tuple order.

`compute_approximations` also owns the public execution-engine boundary:

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    block_size=1024,
    backend="numpy",
)
```

Supported engine aliases are `"dense"` and `"blockwise"`. Supported backend
aliases are `"numpy"`, `"auto"`, and explicit optional `"cupy"`. The CuPy path
currently accelerates similarity-block calculation only; public outputs remain
NumPy arrays.

### `compute_positive_region`

Use this shortcut when only positive-region scores are needed.

```python
scores = compute_positive_region(
    X,
    y,
    model="itfrs",
    similarity="linear",
)
```

### `build_similarity_matrix` and `build_similarity_engine`

Use `build_similarity_matrix` when a user or downstream package needs to build
and reuse the pairwise similarity matrix explicitly.

```python
similarity_matrix = build_similarity_matrix(
    X,
    similarity="linear",
)
```

Use `build_similarity_engine` when advanced code needs the exact dense/blockwise
similarity-engine abstraction directly:

```python
engine = build_similarity_engine(
    X,
    engine="blockwise",
    block_size=512,
    similarity="linear",
    backend="numpy",
)
```

A precomputed matrix can then be fed into the approximation helpers:

```python
scores = compute_positive_region(
    X=None,
    y=y,
    model="itfrs",
    similarity_matrix=similarity_matrix,
)
```

### `FuzzyRoughPositiveRegionScorer`

Use this for reusable fitted workflows and sklearn-style parameter handling.

```python
scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="linear",
)

scores = scorer.fit_score(X, y)
result = scorer.as_result()
```

Current limitation: the scorer wraps the dense/default approximation path and
does not yet expose explicit `engine`, `backend`, or `block_size` constructor
parameters. Those are tracked as the next public API metadata/scorer phase.

## Advanced public API

The following names are public but intended mainly for advanced users,
experiments, and downstream packages:

```python
from FRsutils.api import (
    build_fuzzy_rough_model,
    get_fuzzy_rough_model_class,
    list_fuzzy_rough_models,
    list_similarities,
    normalize_flat_config_to_nested,
)
```

### `build_fuzzy_rough_model`

Use this only when a downstream package needs a constructed fuzzy-rough model
object rather than task-level approximation arrays.

```python
from FRsutils.api import build_fuzzy_rough_model, build_similarity_matrix

similarity_matrix = build_similarity_matrix(X, similarity="linear")
model = build_fuzzy_rough_model(
    model_type="itfrs",
    similarity_matrix=similarity_matrix,
    labels=y,
)
```

## Semi-internal exports

Some low-level classes and helpers may be importable from `FRsutils.api` for
inspection, backwards compatibility, or advanced experimentation:

- `Similarity`
- `FuzzyRoughModel`
- `ITFRS`
- `OWAFRS`
- `VQRS`
- `apply_config_aliases`
- `extract_prefixed_params`
- `calculate_similarity_matrix`

These should not be the first choice for tutorials, README examples, or
external package integration. Prefer the task-level API unless there is a clear
advanced use case.

## Downstream-package rule

Downstream packages should depend on `FRsutils.api` only:

```python
from FRsutils.api import build_similarity_matrix, compute_positive_region
```

They should not import from deep internal paths such as:

```python
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.similarities import build_similarity_matrix
from FRsutils.utils.init_helpers import normalize_flat_config_to_nested
```

The `FRsutils.api` namespace is the compatibility boundary. Internal modules may
change between releases.

## Configuration policy

FRsutils keeps external parameters flat and sklearn-friendly, while internal
code may normalize those parameters into nested component configuration.

Preferred public style:

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

Nested dictionaries are internal implementation details unless explicitly passed
through advanced APIs that document support for them.

## Backend and blockwise status

See [`backend_execution_status.md`](backend_execution_status.md) for the frozen
implementation status, historical phase mapping, and the reduced next-phase
roadmap.

## Versioning expectation

For pre-1.0 releases, the public API may still evolve. However, changes to
`FRsutils.api` should be treated as compatibility-relevant and should be covered
by public API tests.

When changing the public API, update:

- `README.md`,
- this document,
- public API tests,
- downstream contract tests.

## Minimum contract tests

The repository should keep tests that verify:

1. public imports work from `FRsutils.api`,
2. `compute_approximations` returns a named result object,
3. `compute_positive_region` matches `compute_approximations(...).positive_region`,
4. a precomputed similarity matrix can feed approximation helpers,
5. downstream-style code can use only `FRsutils.api` without importing internals.
