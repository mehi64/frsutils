# Phase 6 — Benchmark Suite

Phase 6 adds a reproducible benchmark suite for the dense, blockwise, and
optional CuPy execution paths exposed through the public `FRsutils.api` facade.
It does **not** change runtime approximation behavior. The goal is to make
performance and equivalence claims measurable instead of relying on informal
manual timing.

## Added artifact

```text
benchmarks/benchmark_fuzzy_rough_execution.py
```

The benchmark script compares:

| Scenario | Meaning |
|---|---|
| `dense_numpy` | Historical dense NumPy full-matrix approximation path. |
| `blockwise_numpy` | Exact blockwise NumPy approximation path. |
| `blockwise_cupy` | Optional CuPy-backed blockwise path. For ITFRS/VQRS this may use GPU-resident approximation accumulators; for OWAFRS it remains non-GPU-resident by Phase 5 decision. |

The script writes machine-readable JSON and CSV reports. These reports are the
preferred source for later paper tables, plots, and release notes.

## Example command

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs,owafrs \
    --sample-sizes 128,256,512 \
    --n-features 8 \
    --block-sizes 64,128 \
    --scenarios dense_numpy,blockwise_numpy,blockwise_cupy \
    --repeats 3 \
    --output-json benchmark_results.json \
    --output-csv benchmark_results.csv
```

CPU-only environments are supported. If `backend="cupy"` is requested but CuPy
or CUDA is unavailable, the corresponding rows are marked as `skipped` rather
than crashing the whole benchmark run.

## Recorded fields

Each benchmark row records:

- status: `success`, `skipped`, or `failed`,
- model, scenario, sample count, feature count, and block size,
- requested backend and resolved backend,
- median/mean/min/max runtime over repeated runs,
- Python allocator peak memory measured by `tracemalloc`,
- numerical-equivalence errors against the dense NumPy reference,
- public result metadata:
  - `used_blockwise`,
  - `used_gpu_similarity_blocks`,
  - `used_gpu_approximation_accumulators`.

`python_peak_memory_bytes` is useful as a lightweight smoke metric, but it does
not fully capture native NumPy/CuPy allocator memory. For final paper-quality
memory claims, run the suite on a controlled machine and supplement it with OS
or GPU memory tooling.

## Expected interpretation

Phase 6 supports these paper-safe comparisons:

```text
dense NumPy
vs blockwise NumPy
vs optional CuPy-backed blockwise execution
```

For CuPy rows, interpret metadata carefully:

```text
ITFRS/VQRS + blockwise_cupy:
    used_gpu_similarity_blocks may be True
    used_gpu_approximation_accumulators may be True

OWAFRS + blockwise_cupy:
    used_gpu_similarity_blocks may be True
    used_gpu_approximation_accumulators remains False
```

This preserves the Phase 5 decision that OWAFRS does not become GPU-resident in
the current release/paper cycle.

## Tests

The benchmark suite has lightweight contract tests:

```text
tests/benchmarks/test_phase6_benchmark_suite_contract.py
```

The tests run tiny CPU-only benchmark cases, verify dense/blockwise numerical
equivalence, and confirm that JSON/CSV artifacts are generated.

## Boundary

Phase 6 creates the benchmark harness, not final benchmark numbers. Final
numbers should be generated later on a stable target machine with the intended
CPU/GPU environment, fixed package versions, and documented run settings.
