# Paper and release claim boundaries

This document records the wording that should be used when describing frsutils
in a JOSS paper, release note, README section, or benchmark report.

## Safe project summary

frsutils is a scientific Python library for reusable fuzzy-rough set
computations. It provides backend-aware public APIs for similarity construction,
fuzzy-rough lower and upper approximations, boundary regions, and positive-region
scores. The currently documented model aliases are `itfrs`, `vqrs`, and
`owafrs`.

A safe short claim is:

> frsutils provides dense and exact blockwise fuzzy-rough approximation APIs for
> ITFRS, VQRS, and OWAFRS, with a stable NumPy output contract and optional CuPy
> support in explicit blockwise execution paths.

## Public API scope

Use `frsutils` as the canonical import surface in papers, examples, and
user-facing documentation:

```python
from frsutils import compute_approximations
from frsutils import FuzzyRoughPositiveRegionScorer
```

Do not describe internal modules under `frsutils.core` or `frsutils.utils` as the
stable public API. They may be useful for maintainers, but the JOSS-facing user
entry point is the `frsutils` namespace.

## Execution claim boundary

frsutils supports two main approximation execution modes through the public API:

- `engine="dense"`: builds or consumes a full pairwise similarity matrix and uses
  the dense NumPy reference model implementations.
- `engine="blockwise"`: computes exact approximations by processing similarity
  blocks and avoiding full similarity-matrix materialization by default.

The public output contract is intentionally NumPy-based. Public result arrays are
returned as NumPy arrays even when an optional CuPy-backed blockwise path is used
internally.

## CuPy and GPU wording

Use careful, model-specific wording:

- ITFRS: CuPy-backed blockwise execution can use GPU-backed similarity blocks and
  GPU-resident approximation accumulators.
- VQRS: CuPy-backed blockwise execution can use GPU-backed similarity blocks and
  GPU-resident approximation accumulators.
- OWAFRS: CuPy-backed blockwise execution can use GPU-backed similarity blocks,
  but this release does not claim GPU-resident OWAFRS approximation accumulators.

Avoid these claims unless they are later supported by benchmarks and tests:

- “frsutils is fully GPU-native.”
- “All fuzzy-rough models run fully on the GPU.”
- “CuPy always improves performance.”
- “OWAFRS aggregation is GPU-resident.”

## Benchmark wording

If benchmark results are included, state the exact scenario:

- model alias, such as `itfrs`, `vqrs`, or `owafrs`
- execution engine, such as dense or blockwise
- backend, such as NumPy or CuPy
- sample size, feature count, block size, and hardware
- whether CuPy/CUDA was available or skipped

Prefer wording such as:

> In the tested environment, blockwise execution reduced full-matrix
> materialization and matched dense-reference outputs numerically. CuPy-backed
> runs were optional and were skipped when CuPy/CUDA was unavailable.

## Oversampling boundary

frsutils should be described as the fuzzy-rough core library. Fuzzy-rough
oversampling algorithms such as FRSMOTE live in the standalone downstream
`frsampling` package and depend on the public `frsutils` namespace.

Do not present FRSMOTE as part of the stable frsutils core public API unless the
project intentionally changes that boundary later.

## Recommended JOSS phrasing

A compact JOSS-facing paragraph:

> frsutils is a Python library for fuzzy-rough set computations, including
> similarity construction, lower and upper approximations, boundary regions, and
> positive-region scores. It provides a compact public API through `frsutils`,
> supports ITFRS, VQRS, and OWAFRS model aliases, and offers dense as well as
> exact blockwise execution. Optional CuPy-backed blockwise execution is available
> for selected internal steps while preserving NumPy arrays as the public output
> contract.
