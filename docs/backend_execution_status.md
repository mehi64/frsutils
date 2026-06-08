# Backend and Blockwise Execution Status

This document freezes the current implementation status after the backend and
blockwise work. It is the reference point for the next backend phases.

## Current public contract

FRsutils exposes execution control through the public `FRsutils.api` facade:

```python
from FRsutils.api import compute_approximations, build_similarity_engine

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    block_size=1024,
    backend="numpy",
)

engine = build_similarity_engine(
    X,
    engine="blockwise",
    block_size=1024,
    similarity="linear",
    backend="numpy",
)
```

Supported execution-engine aliases:

- `engine="dense"`: historical full similarity-matrix path.
- `engine="blockwise"`: exact blockwise approximation path.

Supported backend aliases:

- `backend="numpy"`: stable CPU path.
- `backend="auto"`: currently resolves to NumPy for conservative behavior.
- `backend="cupy"`: optional experimental CuPy path for similarity-block
  calculation only.

Public outputs remain NumPy arrays. This preserves compatibility with sklearn,
`frsampling`, tests, and downstream code.

## Implemented status

| Area | Status | Notes |
|---|---:|---|
| Public approximation facade | Done | `compute_approximations`, `compute_positive_region`, lower/upper/boundary helpers. |
| Public similarity facade | Done | `build_similarity_matrix`, `build_similarity_engine`. |
| Dense execution | Done | Backward-compatible full-matrix behavior. |
| Blockwise similarity engine | Done | Exact row/column block iterator with `block_size`. |
| ITFRS blockwise approximation | Done | Exact min/max accumulators without full matrix materialization. |
| VQRS blockwise approximation | Done | Exact numerator/denominator accumulators without full matrix materialization. |
| OWAFRS blockwise approximation | Done | Exact row-buffer execution; keeps one row block, not the full matrix. |
| CuPy backend descriptor | Done | Optional dependency is imported lazily only when requested. |
| CuPy similarity-block path | Done, experimental | Supports backend formulas for implemented similarity/T-norm combinations. |
| GPU-resident approximation accumulators | Not done | Approximation accumulators are still NumPy-resident. |
| Backend-aware core components | Not done | Some backend formulas are mirrored in `similarity_engine.py`. |
| Benchmark suite | Not done | Needed before strong performance claims. |

## Important limitation

The current CuPy path should be described as:

> Optional GPU-accelerated similarity-block computation.

It should not yet be described as:

> Full GPU-native fuzzy-rough computation.

Current flow with `backend="cupy"`:

```text
X block
-> CuPy similarity-block computation
-> conversion back to NumPy
-> NumPy approximation accumulators
-> NumPy public result arrays
```

Future GPU-resident flow:

```text
X block
-> CuPy similarity-block computation
-> CuPy fuzzy-rough accumulator work
-> final conversion to NumPy at the public boundary
```

## Phase history freeze

The following earlier implementation phases are now considered complete in the
current codebase:

| Historical phase | Frozen status |
|---|---|
| Phase 1: similarity-engine abstraction | Complete |
| Phase 2: exact ITFRS blockwise approximation | Complete |
| Phase 3: public positive-region scorer/API boundary | Complete, but scorer does not yet expose engine/backend params directly |
| Phase 4: exact VQRS blockwise approximation | Complete |
| Phase 5: exact OWAFRS blockwise approximation and frsampling compatibility | Complete |
| Phase 6: optional CuPy similarity-block backend | Complete as experimental similarity-block support |

## Next implementation phases

The next work should start from this reduced roadmap:

1. Backend-aware component formulas.
   - Move similarity/T-norm/implicator/fuzzy-quantifier backend formulas into
     their own components or a shared formula adapter.
   - Remove long-term formula duplication from `similarity_engine.py`.

2. Public API metadata and scorer execution params.
   - Add `engine`, `backend`, `block_size`, and backend flags to the result
     metadata/config snapshot.
   - Let `FuzzyRoughPositiveRegionScorer` accept and forward `engine`,
     `backend`, and `block_size`.

3. GPU-resident ITFRS accumulator.
   - Keep ITFRS blockwise reductions on CuPy until the final public NumPy
     conversion.

4. GPU-resident VQRS accumulator; defer or separately benchmark OWAFRS.
   - VQRS is sum/reduction-friendly.
   - OWAFRS requires row-wise sorting and should be handled after memory
     benchmarking.

5. Benchmark and documentation.
   - Compare dense NumPy, blockwise NumPy, CuPy similarity-block mode, and future
     GPU-resident modes.
   - Record runtime, peak memory, numerical equivalence, and `block_size`
     sensitivity.

## Validation snapshot

Targeted backend/public API tests were run in the current environment:

```text
FRsutils targeted API/backend/blockwise tests: 72 passed, 4 skipped
```

The skipped tests require CuPy/CUDA and were skipped because CuPy was not
installed in the execution environment.
