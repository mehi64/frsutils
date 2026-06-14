# Public execution metadata and scorer backend parameters

This document describes the public execution metadata contract for dense and blockwise approximation results.

## Purpose

`compute_approximations(...)` accepts execution controls such as `engine`,
`block_size`, and `backend`. These controls are recorded on the returned
`FuzzyRoughApproximationResult` so downstream packages, tests, benchmark scripts,
and paper artifacts can prove which execution path produced a result.

FRsutils exposes explicit, stable result metadata:

| Field | Meaning |
| --- | --- |
| `engine` | Canonical approximation engine used for the result: `"dense"` or `"blockwise"`. |
| `backend` | Canonical resolved backend used for similarity-block execution. Dense execution reports `"numpy"`. |
| `block_size` | Positive integer for blockwise execution; `None` for dense execution. |
| `used_blockwise` | Boolean flag for blockwise approximation execution. |
| `used_gpu_similarity_blocks` | Boolean flag for CuPy-backed similarity-block computation. |
| `used_gpu_approximation_accumulators` | Boolean flag for model-specific CuPy-resident approximation accumulators. |

The fields are also included in `FuzzyRoughApproximationResult.as_dict()`.

## Scorer Contract

`FuzzyRoughPositiveRegionScorer` now exposes the same execution controls:

```python
from FRsutils.api import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    backend="numpy",
    block_size=512,
)

scores = scorer.fit_score(X, y)
assert scorer.result_.engine == "blockwise"
```

These parameters are constructor parameters, so they are compatible with
scikit-learn-style `get_params`, `set_params`, cloning, and grid-search workflows.

## Boundary Decision

Dense execution remains NumPy-based. Passing `backend="cupy"` is meaningful only
for `engine="blockwise"`, where similarity blocks can use the optional CuPy
backend. The result field `used_gpu_similarity_blocks` is the safe public flag
for checking whether GPU similarity-block execution actually happened.

For blockwise CuPy execution, ITFRS and VQRS may report
`used_gpu_approximation_accumulators=True`. OWAFRS deliberately keeps this field
`False` because exact OWA sorting and aggregation remain on the conservative
NumPy row-buffer path.
