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
  calculation. For blockwise ITFRS and VQRS, approximation accumulators can also
  stay CuPy-resident until final NumPy public output conversion.

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
| Backend-aware core components | Done | Similarity/T-norm/implicator/fuzzy-quantifier formulas expose backend hooks. |
| GPU-resident ITFRS accumulator | Done, experimental | Blockwise ITFRS keeps CuPy blocks, implication/T-norm values, and min/max accumulators resident until final output conversion. |
| GPU-resident VQRS accumulator | Done, experimental | Blockwise VQRS keeps CuPy blocks, minimum T-norm values, numerator/denominator sums, interim ratios, and fuzzy-quantifier outputs resident until final conversion. |
| GPU-resident OWAFRS accumulator | Deliberately out of scope | Phase 5 freezes OWAFRS on the conservative NumPy row-buffer path for this release/paper cycle. |
| Benchmark suite | Done | Phase 6 adds a reproducible CLI benchmark harness with JSON/CSV output, numerical-equivalence checks, and optional CuPy skip handling. |

## Important limitation

The current CuPy path should be described as:

> Optional GPU-accelerated similarity-block computation, with experimental
> GPU-resident blockwise ITFRS and VQRS accumulators.

It should not yet be described as:

> Full GPU-native fuzzy-rough computation.

Phase 5 freezes the conservative flow with `backend="cupy"` for OWAFRS:

```text
X block
-> CuPy similarity-block computation
-> conversion back to NumPy
-> NumPy approximation accumulators
-> NumPy public result arrays
```

Current Phase 3/4 ITFRS and VQRS flow with `backend="cupy"`:

```text
X block
-> CuPy similarity-block computation
-> CuPy implicator/T-norm values
-> CuPy row-wise reductions/accumulator work
-> final conversion to NumPy at the public boundary
```

## Phase history freeze

The following earlier implementation phases are now considered complete in the
current codebase:

| Historical phase | Frozen status |
|---|---|
| Phase 1: similarity-engine abstraction | Complete |
| Phase 2: exact ITFRS blockwise approximation | Complete |
| Phase 3: public positive-region scorer/API boundary | Complete; scorer exposes engine/backend/block-size params through the public API. |
| Phase 4: exact VQRS blockwise approximation | Complete |
| Phase 5: exact OWAFRS blockwise approximation and frsampling compatibility | Complete |
| Phase 6: optional CuPy similarity-block backend | Complete as experimental similarity-block support |

## Current cleanup status

Phase 7 release/paper hardening is complete. The current backend cycle should now
be treated as feature-frozen for release/paper cleanup. Remaining work is
operational rather than feature work:

1. Run the benchmark suite on a controlled target machine.
2. Convert benchmark JSON/CSV output into paper/release tables and plots.
3. Keep README, public examples, benchmark docs, and paper-safe claims aligned.
4. Keep the public claim conservative: CuPy similarity blocks plus experimental
   GPU-resident ITFRS/VQRS accumulators, not full GPU-native execution.

OWAFRS GPU-resident execution is not a remaining implementation phase for the
current cycle. It is explicitly out of scope after the Phase 5 decision.

## Validation snapshot

Targeted backend/public API tests were run in the current environment:

```text
FRsutils targeted API/core tests after Phase 4: 364 passed, 4 skipped
Phase 5 changed documentation only; no runtime code changed.
```

The skipped tests require CuPy/CUDA and were skipped because CuPy was not
installed in the execution environment.

## Phase 1 update — backend-aware component formulas

Phase 1 has moved backend-specific formulas into the component classes. The
similarity engine now delegates to `Similarity.compute_backend(...)` and
`TNorm.compute_backend(...)` instead of owning duplicate formula mirrors. This
keeps the current CuPy path conservative while preparing later GPU-resident
approximation accumulators.

## Phase 2 update - result metadata

Public approximation results now expose execution provenance fields:

- `engine`
- `backend`
- `block_size`
- `used_blockwise`
- `used_gpu_similarity_blocks`

These fields are intended for benchmark scripts, downstream package assertions,
and paper artifact reporting. They do not imply GPU-resident approximation
accumulators; `used_gpu_similarity_blocks=True` only means similarity blocks were
computed through the optional CuPy backend.


## Phase 3 update - GPU-resident ITFRS accumulator

Blockwise ITFRS now consumes backend-resident similarity blocks through
`iter_backend_blocks()` when the resolved backend is CuPy. The lower implicator,
upper T-norm, row-wise min/max reductions, and lower/upper accumulators stay on
CuPy until the final public NumPy conversion.

Public result metadata now includes:

- `used_gpu_approximation_accumulators`

This flag is intentionally separate from `used_gpu_similarity_blocks`. The first
means approximation reductions used CuPy; the second means only similarity blocks
were computed through CuPy.


## Phase 4 update - GPU-resident VQRS accumulator

Blockwise VQRS now keeps similarity blocks, same-label masks, T-norm values,
numerator/denominator accumulators, interim ratios, and fuzzy-quantifier outputs
backend-resident when `backend="cupy"` is used. The public result remains NumPy.

## Phase 5 update - OWAFRS remains non-GPU-resident

Phase 5 finalizes the OWAFRS decision for the current release/paper cycle:
OWAFRS will **not** be made GPU-resident. Exact OWAFRS still uses a NumPy
row-buffer path because it requires row-wise sorting and OWA-weighted
aggregation. The optional CuPy path may still accelerate similarity-block
calculation, but OWAFRS approximation accumulation itself remains NumPy.

For OWAFRS, the metadata contract is:

```text
used_gpu_similarity_blocks          may be True with backend="cupy"
used_gpu_approximation_accumulators remains False
```

See [`phase_5_owafrs_non_gpu_resident_decision.md`](phase_5_owafrs_non_gpu_resident_decision.md)
for the decision rationale and benchmark boundary.


## Phase 6 update - benchmark suite

Phase 6 adds `benchmarks/benchmark_fuzzy_rough_execution.py`, a reproducible
benchmark harness for dense NumPy, exact blockwise NumPy, and optional
CuPy-backed blockwise execution. The suite records runtime, lightweight Python
allocator peak memory, numerical-equivalence errors against dense NumPy, and the
public execution metadata fields used by downstream packages and paper artifacts.

The benchmark suite writes JSON and CSV reports and treats unavailable CuPy/CUDA
as skipped rows rather than hard failures. See
[`phase_6_benchmark_suite.md`](phase_6_benchmark_suite.md) for usage and
interpretation guidance.

## Phase 7 release claim status

Phase 7 does not add a new execution path. It freezes the public wording around
existing execution paths so README, paper text, benchmark reports, and downstream
package documentation remain consistent.

Current status to use in release material:

| Area | Release wording |
|---|---|
| Dense execution | Stable public API path. |
| Blockwise execution | Exact public API path for ITFRS, VQRS, and OWAFRS. |
| CuPy similarity blocks | Optional experimental acceleration path. |
| ITFRS/VQRS CuPy accumulators | Experimental GPU-resident blockwise accumulator path. |
| OWAFRS accumulator | Not GPU-resident in this release; exact NumPy row-buffer path. |
| Public outputs | NumPy arrays. |
| frsampling/FRSMOTE | Downstream package; not GPU-native. |

Use `docs/paper_claims.md` as the source of truth for accepted and forbidden
claims. Use `docs/release_checklist.md` before tagging or submitting.



## Phase 8 cleanup update

Phase 8 performs release-cleanup only; it adds no new fuzzy-rough feature. The
cleanup pass fixes stale README links, removes stale VQRS TODO wording, clarifies
test commands, marks exhaustive model-combination tests as `slow`, archives KEEL
Audit WIP copies outside the importable package, and removes local cache files
from generated project artifacts.
