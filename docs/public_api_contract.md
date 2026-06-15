# frsutils Public API Contract

frsutils exposes its stable user-facing and downstream-package API through:

```python
from frsutils import ...
```

This document defines the public compatibility boundary for users, examples,
tests, and downstream packages such as `frsampling`.

## Purpose

frsutils is the fuzzy-rough core package. It provides reusable fuzzy-rough
building blocks and task-oriented helpers for:

- similarity-matrix construction,
- lower approximation,
- upper approximation,
- boundary-region computation,
- positive-region computation,
- reusable positive-region scoring workflows,
- fuzzy-rough model construction for advanced users and downstream packages.

Oversampling algorithms such as FRSMOTE are intentionally outside frsutils core
and should live in downstream packages such as `frsampling`.

## Stable task-level API

These names are the preferred public API for normal users and documentation
examples:

```python
from frsutils import (
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
accelerates similarity-block calculation and, for blockwise ITFRS, can keep the
ITFRS approximation accumulator on CuPy until final NumPy output conversion.
Public outputs remain NumPy arrays.

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

The scorer exposes `engine`, `backend`, and `block_size` constructor parameters
and forwards them to `compute_approximations(...)`. The fitted result object
contains execution metadata including `used_gpu_similarity_blocks` and
`used_gpu_approximation_accumulators`. The latter is true for the experimental
CuPy-resident blockwise ITFRS and VQRS accumulator paths. It remains false for
OWAFRS by design after the OWAFRS non-GPU-resident decision, even when OWAFRS uses
CuPy similarity-block computation.

## Benchmark and execution-reporting contract

frsutils includes a public-API benchmark harness at:

```text
benchmarks/benchmark_fuzzy_rough_execution.py
```

The benchmark script intentionally depends on `frsutils` rather than private
core modules. This keeps benchmark results aligned with the same compatibility
boundary used by downstream packages such as `frsampling`.

The benchmark outputs JSON and CSV rows containing execution metadata from
`FuzzyRoughApproximationResult`, including `engine`, `backend`, `block_size`,
`used_blockwise`, `used_gpu_similarity_blocks`, and
`used_gpu_approximation_accumulators`. Downstream benchmark consumers should use
these fields instead of inferring execution paths from command-line arguments.

See [`benchmark_suite.md`](benchmark_suite.md) for the benchmark
matrix and interpretation rules.

## Advanced public API

The following names are public but intended mainly for advanced users,
experiments, and downstream packages:

```python
from frsutils import (
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
from frsutils import build_fuzzy_rough_model, build_similarity_matrix

similarity_matrix = build_similarity_matrix(X, similarity="linear")
model = build_fuzzy_rough_model(
    model_type="itfrs",
    similarity_matrix=similarity_matrix,
    labels=y,
)
```

## Semi-internal exports

Some low-level classes and helpers may be importable from `frsutils` for
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

## OWAFRS backend boundary

OWAFRS is supported by the dense and exact blockwise public approximation APIs.
However, OWAFRS approximation accumulators are not GPU-resident in the current
cycle. With `model="owafrs"`, `engine="blockwise"`, and `backend="cupy"`, only
the similarity-block calculation may use CuPy. The OWA row-buffer, row-wise
sorting, and weighted aggregation remain NumPy.

This keeps the public execution claim conservative and avoids implying full
GPU-native OWAFRS support before sorting and memory benchmarks exist.

## Downstream-package rule

Downstream packages should depend on `frsutils` only:

```python
from frsutils import build_similarity_matrix, compute_positive_region
```

They should not import from deep internal paths such as:

```python
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.similarities import build_similarity_matrix
from frsutils.utils.init_helpers import normalize_flat_config_to_nested
```

The `frsutils` namespace is the compatibility boundary. Internal modules may
change between releases.

## Configuration policy

frsutils keeps external parameters flat and sklearn-friendly, while internal
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

See [`backend_execution_status.md`](backend_execution_status.md) for the current
implementation status and model-specific backend boundary.

## Versioning expectation

For pre-1.0 releases, the public API may still evolve. However, changes to
`frsutils` should be treated as compatibility-relevant and should be covered
by public API tests.

When changing the public API, update:

- `README.md`,
- this document,
- public API tests,
- downstream contract tests.

## Minimum contract tests

The repository should keep tests that verify:

1. public imports work from `frsutils`,
2. `compute_approximations` returns a named result object,
3. `compute_positive_region` matches `compute_approximations(...).positive_region`,
4. a precomputed similarity matrix can feed approximation helpers,
5. downstream-style code can use only `frsutils` without importing internals.

## Execution metadata

`FuzzyRoughApproximationResult` records public execution provenance in addition
to approximation arrays:

- `engine`: canonical execution engine, usually `"dense"` or `"blockwise"`.
- `backend`: canonical resolved backend. Dense execution reports `"numpy"`.
- `block_size`: block size used by blockwise execution, or `None` for dense.
- `used_blockwise`: whether blockwise approximation execution was used.
- `used_gpu_similarity_blocks`: whether similarity blocks were computed through
  the optional CuPy backend.
- `used_gpu_approximation_accumulators`: whether model-specific approximation
  accumulators remained CuPy-resident until final NumPy public output conversion.

`FuzzyRoughPositiveRegionScorer` accepts `engine`, `backend`, and `block_size`
and forwards them to `compute_approximations(...)` while preserving sklearn-style
parameter compatibility.

## Release and paper claim boundary

The release-hardening docs freeze the wording that should be used in releases and software-paper
material. Public examples and downstream packages should continue to depend on
`frsutils`; private `frsutils.core` modules are not part of the stable
compatibility boundary.

Safe release claim:

```text
frsutils provides dense and exact blockwise fuzzy-rough approximation APIs with
optional CuPy-accelerated similarity blocks and experimental CuPy-resident
ITFRS/VQRS blockwise approximation accumulators. Public outputs remain NumPy
arrays, and OWAFRS remains on the conservative exact blockwise NumPy row-buffer
path in the current release.
```

See also:

- `docs/paper_claims.md`
- `docs/release_checklist.md`
- `docs/release_paper_hardening.md`

