# frsutils public API

`frsutils` exposes its stable user-facing API from the package root:

```python
from frsutils import compute_approximations
```

The grouped `frsutils.api` namespace exposes the same facade objects. User code,
notebooks, examples, and downstream research packages should prefer one of these
public import paths instead of importing from `frsutils.core` or `frsutils.utils`.

## Purpose

FRsutils is a fuzzy-rough computation library for research workflows. Its public
API provides task-oriented helpers for:

- pairwise similarity construction,
- lower and upper fuzzy-rough approximations,
- boundary and positive regions,
- dense and exact blockwise execution,
- dense fuzzy-rough model construction,
- reusable positive-region scoring.

The supported fuzzy-rough model aliases are:

- `"itfrs"` — implicator/T-norm fuzzy-rough sets,
- `"owafrs"` — ordered weighted averaging fuzzy-rough sets,
- `"vqrs"` — vaguely quantified rough sets.

## Main public entry points

| Task | Public API |
| --- | --- |
| Compute all approximation outputs | `compute_approximations` |
| Compute one output | `compute_lower_approximation`, `compute_upper_approximation`, `compute_boundary_region`, `compute_positive_region` |
| Build a pairwise similarity matrix | `build_similarity_matrix` |
| Build a dense or blockwise similarity engine | `build_similarity_engine` |
| Build a dense fuzzy-rough model object | `build_fuzzy_rough_model` |
| Compute and cache positive-region scores | `FuzzyRoughPositiveRegionScorer` |

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

A runnable version is available in `examples/public_api_quickstart.py`.

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

It returns a `FuzzyRoughApproximationResult`, so downstream code accesses named
fields rather than positional tuple entries:

```python
result.lower
result.upper
result.boundary
result.positive_region
```

The public result contract is:

```python
np.allclose(result.boundary, result.upper - result.lower)
np.allclose(result.positive_region, result.lower)
```

Public result arrays are NumPy arrays, including results produced by optional
CuPy-backed blockwise internals.

### Input validation and minimum dataset size

All public approximation entry points use the same minimum-size contract:
`X` and `y`, or a precomputed similarity matrix and `y`, must describe at least
two aligned samples. Empty and single-sample inputs raise `ValueError` for all
three models and for both dense and blockwise execution. The approximation
wrapper functions and `FuzzyRoughPositiveRegionScorer` follow the same rule.

Feature data supplied to the public similarity and approximation builders must
be a finite two-dimensional numeric array. Values such as `NaN`, positive
infinity, and negative infinity are rejected. A precomputed similarity matrix
must be finite, square, aligned with the one-dimensional label array, and at
least `2 x 2`.

This validation is performed at the public boundary so mathematically undefined
or model-dependent edge behavior does not leak into research workflows.

### Dense execution

Dense execution materializes or consumes a full pairwise similarity matrix and
uses the dense reference model implementations:

```python
result = compute_approximations(
    X,
    y,
    model="owafrs",
    similarity="linear",
    engine="dense",
)
```

Dense approximation execution is NumPy-only. `backend="numpy"` and
`backend="auto"` are accepted; `backend="cupy"` raises a clear error rather
than being silently ignored. Use blockwise execution for optional CuPy-backed
computation.

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

When a precomputed matrix is supplied, do not also pass `similarity` or
`similarity_*` settings. FRsutils did not construct that matrix and therefore
reports `result.similarity is None` rather than claiming an unknown provenance.

### Blockwise execution

Blockwise execution computes exact approximations without materializing the full
similarity matrix by default:

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

Blockwise mode requires `X` and does not accept a precomputed
`similarity_matrix`. To materialize the matrix for inspection, set
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

## Model examples

The three public model aliases use the same flat configuration contract. These
examples intentionally configure model-specific components so the routing rules
are visible.

### ITFRS

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="gaussian",
    similarity_sigma=0.4,
    similarity_tnorm="yager",
    similarity_tnorm_p=2.0,
    ub_tnorm_name="yager",
    ub_tnorm_p=1.7,
    lb_implicator_name="goguen",
)
```

### OWAFRS

```python
result = compute_approximations(
    X,
    y,
    model="owafrs",
    similarity="linear",
    ub_tnorm_name="minimum",
    lb_implicator_name="lukasiewicz",
    ub_owa_method_name="exponential",
    ub_owa_method_base=2.5,
    lb_owa_method_name="harmonic",
)
```

### VQRS

```python
result = compute_approximations(
    X,
    y,
    model="vqrs",
    similarity="linear",
    lb_fuzzy_quantifier_name="linear",
    lb_fuzzy_quantifier_alpha=0.0,
    lb_fuzzy_quantifier_beta=0.5,
    ub_fuzzy_quantifier_name="quadratic",
    ub_fuzzy_quantifier_alpha=0.1,
    ub_fuzzy_quantifier_beta=0.8,
)
```

The public defaults are ITFRS with a minimum upper T-norm and Lukasiewicz lower
implicator; linear similarity aggregated by the minimum T-norm is used when
similarity settings are omitted. OWAFRS defaults to linear upper and lower OWA
weights. VQRS defaults to linear lower and upper quantifiers with
`alpha=0.1` and `beta=0.6`.

See [Public configuration contract](configuration.md) for the complete selector,
prefix, alias, parameter, default, validation, and backward-compatibility rules.

## Approximation wrapper functions

Wrapper functions are available when only one output is needed:

```python
from frsutils import compute_positive_region

positive = compute_positive_region(
    X,
    y,
    model="itfrs",
    similarity="linear",
)
```

The wrappers delegate to `compute_approximations` and return NumPy arrays.

## Result object and execution metadata

`FuzzyRoughApproximationResult` is a frozen public result container:

```python
data = result.as_dict()
```

`as_dict()` omits `similarity_matrix` by default because the matrix can be large.
Include it explicitly when needed:

```python
data = result.as_dict(include_similarity_matrix=True)
```

`frozen` prevents reassigning result attributes; contained NumPy arrays and the
configuration dictionary are not deep-immutable objects.

Important metadata fields are:

| Field | Meaning |
| --- | --- |
| `model` | Canonical fuzzy-rough model alias. |
| `similarity` | Similarity alias used by FRsutils to build the matrix, or `None` for externally supplied matrices. |
| `engine` | Canonical approximation engine: `"dense"` or `"blockwise"`. |
| `backend` | Canonical resolved backend. Dense execution reports `"numpy"`. |
| `block_size` | Positive integer for blockwise execution; `None` for dense execution. |
| `used_blockwise` | Whether blockwise approximation execution was used. |
| `used_gpu_similarity_blocks` | Whether similarity blocks used the optional CuPy backend. |
| `used_gpu_approximation_accumulators` | Whether model-specific approximation accumulators stayed CuPy-resident before final NumPy conversion. |

For OWAFRS, CuPy-backed blockwise execution currently accelerates similarity
blocks; the OWA approximation accumulators are not claimed to be GPU-resident.

## Similarity API

Use `build_similarity_matrix` for an ordinary dense pairwise matrix:

```python
from frsutils import build_similarity_matrix

S = build_similarity_matrix(
    X,
    similarity="gaussian",
    similarity_sigma=0.5,
)
```

Use `build_similarity_engine` when block iteration is needed:

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

Similarity builders accept only similarity-related flat configuration. Model
parameters such as `ub_tnorm_name` are rejected instead of being ignored.

## Building dense model objects

`build_fuzzy_rough_model` constructs a dense model object from a precomputed
similarity matrix and labels:

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

The builder applies the same authoritative model defaults as
`compute_approximations`. It accepts model-related flat configuration only;
similarity settings are rejected because the matrix has already been built.

## Positive-region scorer

`FuzzyRoughPositiveRegionScorer` provides an estimator-like API for fitted-sample
positive-region scores:

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

The scorer follows scikit-learn conventions for parameter handling, cloning,
`get_params`, and `set_params`. It computes fuzzy-rough scores for the fitted
training relation; unseen-sample scoring is not currently defined. After `fit`,
request the cached fitted scores with no argument:

```python
scores = scorer.score_samples()
```

Passing `X` to `score_samples(X)` raises `ValueError` instead of silently
returning training scores for unrelated samples.

`extra_params` may hold contract-defined flat keys not represented by an
explicit scorer constructor parameter. It cannot duplicate or alias an explicit
constructor parameter, because hidden overrides would break reproducibility.

## Flat public configuration

The stable public configuration boundary is **flat only**. A selector chooses a
registered component, and component parameters use the same component prefix:

```text
<component>_name = <registered alias>
<component>_<constructor parameter> = <value>
```

For example:

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="gaussian",
    similarity_sigma=0.5,
    ub_tnorm_name="yager",
    ub_tnorm_p=2.0,
    lb_implicator_name="lukasiewicz",
)
```

Similarity selectors use the shorter public names `similarity` and
`similarity_tnorm`; their parameters still use `similarity_` and
`similarity_tnorm_` prefixes.

| Component role | Selector | Parameter prefix |
| --- | --- | --- |
| Similarity | `similarity` | `similarity_` |
| Similarity aggregation T-norm | `similarity_tnorm` | `similarity_tnorm_` |
| Upper T-norm | `ub_tnorm_name` | `ub_tnorm_` |
| Lower implicator | `lb_implicator_name` | `lb_implicator_` |
| Upper OWA weighting | `ub_owa_method_name` | `ub_owa_method_` |
| Lower OWA weighting | `lb_owa_method_name` | `lb_owa_method_` |
| Lower fuzzy quantifier | `lb_fuzzy_quantifier_name` | `lb_fuzzy_quantifier_` |
| Upper fuzzy quantifier | `ub_fuzzy_quantifier_name` | `ub_fuzzy_quantifier_` |

A flat configuration dictionary can be passed through `config`:

```python
config = {
    "type": "itfrs",
    "similarity": "gaussian",
    "similarity_sigma": 0.5,
    "ub_tnorm_name": "minimum",
    "lb_implicator_name": "lukasiewicz",
}

result = compute_approximations(X, y, config=config)
```

Additional flat keyword arguments override ordinary values from `config`.
Explicit model sources are stricter: `model="vqrs"` and `config={"type":
"itfrs"}` are a conflict and raise `ValueError` rather than silently choosing one.
If no model source is supplied, the public default is `"itfrs"`.

Nested component dictionaries are an internal normalized representation and are
rejected at public API boundaries. This is intentional: one flat configuration
contract keeps parameter names searchable, typo-checkable, reproducible, and
compatible with estimator-style parameter workflows.

The detailed source of truth is [Public configuration contract](configuration.md).

## Advanced public exports

The facade also exposes registry discovery and concrete model/similarity types
for advanced inspection and downstream research code:

```python
from frsutils import (
    FuzzyRoughModel,
    ITFRS,
    OWAFRS,
    Similarity,
    VQRS,
    get_fuzzy_rough_model_class,
    list_fuzzy_rough_models,
    list_similarities,
)
```

Similarity engine classes, `SimilarityBlock`, and
`calculate_similarity_matrix` are also exported for advanced workflows. They are
not the preferred starting point for tutorials; use the task-level builders and
computation functions when possible.

Internal flat-to-nested construction helpers such as
`normalize_flat_config_to_nested`, `apply_config_aliases`, and
`extract_prefixed_params` are deliberately **not** exported by `frsutils` or
`frsutils.api`.

## Downstream-package rule

Downstream packages should depend on the public facade:

```python
from frsutils import build_similarity_matrix, compute_positive_region
```

Avoid deep internal imports in downstream user-facing code:

```python
# Avoid this in user-facing examples and downstream packages.
from frsutils.core.models.itfrs import ITFRS
from frsutils.core.similarities import build_similarity_matrix
from frsutils.utils.init_helpers import normalize_flat_config_to_nested
```

The `frsutils` namespace is the compatibility boundary. Internal modules may
change between releases.

## Backend behavior

NumPy is the stable backend. CuPy support is optional and used only through
explicit supported blockwise paths:

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

Public outputs remain NumPy arrays regardless of the internal backend. See
[Backends](backends.md) for the model-specific backend contract.
