# Phase 5 - OWAFRS GPU-resident non-scope decision

Phase 5 intentionally does **not** make OWAFRS approximation accumulators
GPU-resident.

This is a design-stabilization phase, not a computation-extension phase. The
current exact OWAFRS blockwise implementation remains the supported path:

```text
optional CuPy similarity-block computation
→ conversion to NumPy block
→ NumPy lower/upper row buffers
→ NumPy row-wise sort
→ NumPy OWA-weighted aggregation
→ NumPy public result arrays
```

## Decision

OWAFRS will stay on the conservative NumPy row-buffer approximation path for the
current release/paper cycle.

The public metadata contract is therefore:

```python
result.used_gpu_similarity_blocks          # may be True for backend="cupy"
result.used_gpu_approximation_accumulators # remains False for OWAFRS
```

For OWAFRS, `used_gpu_similarity_blocks=True` only means that similarity blocks
were computed through the optional CuPy backend before conversion back to NumPy.
It does not mean that OWA sorting or approximation accumulation ran on GPU.

## Rationale

ITFRS and VQRS have GPU-friendly accumulator structures:

- ITFRS uses min/max reductions.
- VQRS uses sum-based numerator/denominator accumulators and fuzzy-quantifier
  outputs.

OWAFRS is different. Exact OWAFRS needs row-wise sorting and OWA-weighted
aggregation. Making that path GPU-resident would require separate memory,
chunking, and sorting benchmarks before it could be claimed responsibly. Without
that evidence, a GPU-resident OWAFRS implementation would add complexity and
risk without a clear release/paper benefit.

## Current claim boundary

Allowed claim:

```text
FRsutils supports exact dense and blockwise fuzzy-rough approximation APIs,
optional CuPy-accelerated similarity-block computation, and experimental
GPU-resident blockwise accumulators for ITFRS and VQRS.
```

Not allowed claim:

```text
FRsutils provides full GPU-native execution for all fuzzy-rough models.
```

## Next phase

The next implementation phase should be the benchmark suite. It should compare:

- dense NumPy,
- blockwise NumPy,
- blockwise CuPy similarity-block execution,
- GPU-resident ITFRS,
- GPU-resident VQRS,
- OWAFRS conservative NumPy row-buffer execution.

The benchmark should record runtime, memory behavior, numerical equivalence,
block-size sensitivity, and the CPU/GPU break-even point.
