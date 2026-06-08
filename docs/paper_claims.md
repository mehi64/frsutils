# Paper and Release Claims

This document freezes the wording that should be used for FRsutils release notes,
README summaries, benchmark reports, and paper/software-submission text.

## Recommended short claim

FRsutils is a Python library for reusable fuzzy-rough set utilities with a stable
public API for similarity construction, lower/upper approximation, boundary
region, and positive-region computation. It supports dense and exact blockwise
execution, optional CuPy-accelerated similarity-block computation, and
experimental CuPy-resident blockwise approximation accumulators for ITFRS and
VQRS.

## Recommended longer claim

FRsutils separates fuzzy-rough core computation from downstream sampling
algorithms. The package exposes a public `FRsutils.api` facade for dense and exact
blockwise fuzzy-rough approximations. Blockwise execution avoids materializing
the full pairwise similarity matrix during approximation computation. Optional
CuPy support can accelerate similarity-block computation. For ITFRS and VQRS,
blockwise CuPy execution can also keep approximation reductions on the backend
until the final public NumPy output conversion. OWAFRS remains on the
conservative exact blockwise NumPy row-buffer path in the current release because
its row-wise sorting and OWA aggregation need a separate memory/sorting study.

## Do not claim

Do not claim any of the following for the current release/paper cycle:

- full GPU-native fuzzy-rough execution,
- GPU-native FRSMOTE,
- GPU-resident OWAFRS approximation accumulators,
- end-to-end CuPy outputs from public APIs,
- CuPy support as a mandatory dependency,
- benchmark speedups before running the Phase 6 benchmark suite on a stable
  target machine.

## Correct backend wording

Use this wording when describing `backend="cupy"`:

```text
`backend="cupy"` is optional and experimental. It enables CuPy-backed
similarity-block computation and, for blockwise ITFRS/VQRS, CuPy-resident
approximation accumulators. Public result arrays remain NumPy arrays.
```

For OWAFRS, use this wording:

```text
OWAFRS supports exact blockwise execution, but its approximation accumulator is
not GPU-resident in the current release. CuPy may be used for similarity-block
computation only; the OWA row-buffer and sorting path remains NumPy-based.
```

## Relationship to frsampling

FRsutils owns fuzzy-rough core computation. `frsampling` owns FRSMOTE and future
sampling algorithms. Downstream packages should import from `FRsutils.api`, not
from internal modules.

## Evidence expected before final paper submission

Before using numerical speedup claims in a paper, generate benchmark artifacts
with:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs,owafrs \
    --sample-sizes 128,256,512,1024 \
    --n-features 8 \
    --block-sizes 64,128,256 \
    --scenarios dense_numpy,blockwise_numpy,blockwise_cupy \
    --repeats 5 \
    --output-json benchmark_results.json \
    --output-csv benchmark_results.csv
```

Record the machine, Python version, NumPy version, optional CuPy/CUDA version,
GPU model, and command line together with the benchmark artifacts.
