# Benchmark suite

This document describes the reproducible benchmark suite for the dense, blockwise, and
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
| `blockwise_cupy` | Optional CuPy-backed blockwise path. For ITFRS/VQRS this may use GPU-resident approximation accumulators; for OWAFRS it remains non-GPU-resident by the OWAFRS non-GPU-resident decision. |

The script writes machine-readable JSON and CSV reports. These reports are the
preferred source for later paper tables, plots, and release notes.

## Synthetic dataset size control

For synthetic benchmarks, control the generated dataset size directly with
`--synthetic-samples` and `--n-features`:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs \
    --synthetic-samples 5000 \
    --n-features 20 \
    --block-sizes 512,1024 \
    --scenarios blockwise_numpy,blockwise_cupy \
    --skip-dense-reference \
    --compare-blockwise-backends \
    --repeats 3 \
    --output-json benchmark_synthetic_large.json \
    --output-csv benchmark_synthetic_large_cases.csv \
    --output-comparison-csv benchmark_synthetic_large_comparison.csv
```

Use `--sample-sizes` instead of `--synthetic-samples` when several synthetic
sizes should be tested in one run:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs \
    --sample-sizes 1000,2000,5000 \
    --n-features 20 \
    --block-sizes 512 \
    --scenarios blockwise_numpy,blockwise_cupy \
    --skip-dense-reference \
    --compare-blockwise-backends \
    --output-json benchmark_size_sweep.json
```

## Basic benchmark command

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

## Paired NumPy/CuPy blockwise comparison

For large datasets, use paired comparison mode. This runs blockwise NumPy and
blockwise CuPy on the same dataset, model, and block size, then records both
runtime and numerical differences:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs \
    --synthetic-samples 10000 \
    --n-features 30 \
    --block-sizes 512,1024 \
    --scenarios blockwise_numpy,blockwise_cupy \
    --skip-dense-reference \
    --compare-blockwise-backends \
    --comparison-backends numpy,cupy \
    --repeats 3 \
    --output-json benchmark_paired.json \
    --output-csv benchmark_paired_cases.csv \
    --output-comparison-csv benchmark_paired_comparison.csv
```

The JSON report contains a top-level `comparisons` list. The comparison CSV
contains one row per `model + sample size + block size` pair with fields such as:

- `reference_median_runtime_seconds`,
- `candidate_median_runtime_seconds`,
- `speedup_reference_over_candidate`,
- `max_abs_diff_lower`,
- `max_abs_diff_upper`,
- `max_abs_diff_boundary`,
- `max_abs_diff_positive_region`.

With the default `--comparison-backends numpy,cupy`, the speedup field is:

```text
NumPy median runtime / CuPy median runtime
```

Values greater than 1 mean the candidate backend, normally CuPy, was faster in
that benchmark case.

## Large real dataset mode

For large CSV datasets, compare only blockwise CPU and GPU execution and skip
the dense NumPy reference:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs \
    --input-csv path/to/dataset.csv \
    --target-column class \
    --block-sizes 512,1024 \
    --scenarios blockwise_numpy,blockwise_cupy \
    --repeats 3 \
    --skip-dense-reference \
    --compare-blockwise-backends \
    --output-json benchmark_large.json \
    --output-csv benchmark_large_cases.csv \
    --output-comparison-csv benchmark_large_comparison.csv
```

The CSV loader expects a header row, one target column, and numeric feature
columns. The target column can be selected by name or by zero-based column
index. NumPy input is also supported:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py \
    --models itfrs,vqrs \
    --input-npy-x X.npy \
    --input-npy-y y.npy \
    --block-sizes 512,1024 \
    --scenarios blockwise_numpy,blockwise_cupy \
    --skip-dense-reference \
    --compare-blockwise-backends \
    --output-json benchmark_large.json \
    --output-comparison-csv benchmark_large_comparison.csv
```

When `--skip-dense-reference` is used, dense-reference error fields in the
case-level `results` rows are left empty. Paired comparison rows still compare
NumPy blockwise directly against CuPy blockwise.

## Recorded fields

Each case-level benchmark row records:

- status: `success`, `skipped`, or `failed`,
- model, scenario, sample count, feature count, and block size,
- requested backend and resolved backend,
- median/mean/min/max runtime over repeated runs,
- Python allocator peak memory measured by `tracemalloc`,
- numerical-equivalence errors against the dense NumPy reference when enabled,
- whether dense reference computation was enabled,
- public result metadata:
  - `used_blockwise`,
  - `used_gpu_similarity_blocks`,
  - `used_gpu_approximation_accumulators`.

Each paired comparison row records:

- reference and candidate backend aliases,
- reference and candidate median/mean runtimes,
- speedup ratio,
- max absolute differences for lower, upper, boundary, and positive-region
  arrays,
- GPU metadata flags for both sides.

`python_peak_memory_bytes` is useful as a lightweight smoke metric, but it does
not fully capture native NumPy/CuPy allocator memory. For final paper-quality
memory claims, run the suite on a controlled machine and supplement it with OS
or GPU memory tooling.

## Expected interpretation

The benchmark suite supports these paper-safe comparisons:

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

This preserves the OWAFRS non-GPU-resident decision in
the current release/paper cycle.

## Tests

The benchmark suite has lightweight contract tests:

```text
tests/benchmarks/test_benchmark_suite_contract.py
```

The tests run tiny CPU-only benchmark cases, verify dense/blockwise numerical
equivalence, verify paired backend comparison behavior, and confirm that
JSON/CSV artifacts are generated.

## Boundary

The benchmark suite provides the harness, not final benchmark numbers. Final
numbers should be generated later on a stable target machine with the intended
CPU/GPU environment, fixed package versions, and documented run settings.
