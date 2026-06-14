# OWAFRS GPU-resident accumulator decision

## Decision

For the current JOSS/release preparation cycle, OWAFRS should not claim
GPU-resident approximation accumulators. OWAFRS may use CuPy-backed similarity
blocks in blockwise execution, but exact OWA sorting and aggregation remain on a
NumPy-compatible row-buffer path.

## Rationale

ITFRS and VQRS blockwise approximations can be accumulated through simpler
reductions such as minima, maxima, and sums. OWAFRS is different: exact OWA
aggregation requires collecting values for each row, sorting them, and applying
OWA weights. Making this fully GPU-resident would require a separate design for
row-wise buffering, sorting, memory layout, and numerical equivalence testing.

Adding that design before JOSS would increase implementation risk without being
necessary for a correct, useful, and testable release.

## Current public claim

Safe claim:

> OWAFRS supports dense NumPy and exact blockwise execution. With
> `backend="cupy"`, similarity blocks can be computed using the CuPy backend,
> but OWAFRS aggregation does not currently claim GPU-resident approximation
> accumulators. Public outputs remain NumPy arrays.

Do not claim:

- full GPU-native OWAFRS,
- GPU-resident OWAFRS OWA aggregation,
- guaranteed speedup for OWAFRS with CuPy.

## Future work

Future performance-focused work can revisit GPU-resident OWAFRS if benchmarks
show that OWAFRS aggregation is a practical bottleneck. That work should include
benchmarks, memory tests, and dense/blockwise/GPU numerical-equivalence tests.
