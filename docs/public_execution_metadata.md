# Phase 2 - Public Execution Metadata and Scorer Backend Parameters

Phase 2 stabilizes the public execution contract after the backend-aware
component formulas introduced in Phase 1.

## Purpose

`compute_approximations(...)` already accepts execution controls such as
`engine`, `block_size`, and `backend`. Before this phase, these controls affected
execution but were not visible in the returned `FuzzyRoughApproximationResult`.
That made it harder for downstream packages, tests, benchmark scripts, and paper
artifacts to prove which execution path produced a result.

Phase 2 adds explicit, stable result metadata:

| Field | Meaning |
| --- | --- |
| `engine` | Canonical approximation engine used for the result: `"dense"` or `"blockwise"`. |
| `backend` | Canonical resolved backend used for similarity-block execution. Dense execution reports `"numpy"`. |
| `block_size` | Positive integer for blockwise execution; `None` for dense execution. |
| `used_blockwise` | Boolean flag for blockwise approximation execution. |
| `used_gpu_similarity_blocks` | Boolean flag for CuPy-backed similarity-block computation. |

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

Phase 2 does **not** make ITFRS/VQRS/OWAFRS accumulators GPU-resident. That is a
later phase.
