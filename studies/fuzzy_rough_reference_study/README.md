# Reproducible fuzzy-rough reference study

This study is the public research-use artifact distributed with `frsutils`.
It demonstrates a complete workflow built on the package-root public API:

```python
from frsutils import compute_approximations
```

It does not use or disclose FRSMOTE. Its purpose is to show that the fuzzy-rough
core can support a traceable downstream research workflow independently of any
unpublished oversampling method.

## Research questions

The study addresses three limited questions:

1. Can ITFRS, VQRS, and OWAFRS be applied through one public API to real tabular
   classification data?
2. Do exact blockwise NumPy results agree numerically with dense NumPy results?
3. What runtimes are observed for dense and blockwise execution under a recorded
   environment and fixed configuration?

The study is not intended to rank the three model families. Their approximation
semantics and robustness assumptions differ, so raw scores should not be treated
as interchangeable performance measures.

## Datasets and tasks

All datasets are bundled with scikit-learn and require no external download.
Features are transformed independently to `[0, 1]` with `MinMaxScaler`.

| Study task | Source | Target construction |
| --- | --- | --- |
| `breast_cancer` | scikit-learn breast-cancer dataset | Original binary target |
| `wine_class_0_vs_rest` | scikit-learn wine dataset | Class 0 versus all remaining classes |
| `digits_3_vs_8` | scikit-learn digits dataset | Keep digits 3 and 8; digit 8 is the positive class |

The original target value is retained in `sample_scores.csv` for traceability.

## Model profile

The fixed profile is stored in [`study_config.json`](study_config.json). All
models use Gaussian feature similarity with `sigma=2.0` and NumPy execution.
VQRS uses a stricter lower quantifier (`alpha=0.4`, `beta=0.8`) and a more
permissive upper quantifier (`alpha=0.1`, `beta=0.6`). These fixed values are
declared before execution and are not tuned per dataset.
OWAFRS uses the documented linear OWA strategy for both approximations.

`boundary` is recorded exactly according to the public result contract:

```text
boundary = upper - lower
```

The study treats it as a signed model output and does not add an undocumented
non-negativity assumption. Samples are ranked by `abs(boundary)` only to identify
large lower/upper gaps for inspection.

## Reproduce the complete study

From the repository root:

```bash
python -m pip install -e ".[study]"
python studies/fuzzy_rough_reference_study/run_study.py
```

The command regenerates real-dataset tables, the synthetic execution benchmark,
environment metadata, figures, and checksums under `results/`.

To use a different output directory:

```bash
python studies/fuzzy_rough_reference_study/run_study.py \
  --output-dir ./reference_study_output
```

CPU-only smoke run without figures or the synthetic benchmark:

```bash
python studies/fuzzy_rough_reference_study/run_study.py \
  --skip-benchmark \
  --skip-figures \
  --output-dir ./reference_study_smoke
```

## Generated artifacts

The committed result snapshot contains:

- `approximation_summary.csv`: dataset/model summaries and median runtimes,
- `dense_blockwise_equivalence.csv`: maximum absolute errors and pass status,
- `runtime_results.csv`: repeat-level runtime observations,
- `sample_scores.csv`: per-sample lower, upper, boundary, and positive-region
  values from dense reference execution,
- `highest_boundary_gap_samples.csv`: samples with the largest absolute signed
  boundary gaps,
- `benchmark_results.json` and `benchmark_results.csv`: the repository benchmark
  profile for dense and blockwise NumPy execution,
- `environment.json`: Python, package, platform, and Git metadata,
- `resolved_config.json`: exact configuration snapshot used for the run,
- `requirements-study.txt`: minimal exact versions needed to recreate the
  recorded environment,
- `figures/`: runtime and positive-region visual summaries,
- `study_manifest.json`: file sizes and SHA-256 checksums.

## Numerical acceptance rule

For every real dataset and model, dense and blockwise lower, upper, boundary, and
positive-region arrays must agree within the absolute tolerance recorded in
`study_config.json`. The script exits with an error when any comparison fails or
when a public result contains non-finite or out-of-range lower, upper, or
positive-region values.

## Runtime interpretation

Runtime values describe the recorded machine and software environment; they are
not universal performance guarantees. The benchmark's
`python_peak_memory_bytes` field comes from `tracemalloc` and does not fully
measure native NumPy or CuPy allocator memory. No memory-reduction or GPU-speedup
claim is made from this study.

## Final-release procedure

After the phase-three files are committed, rerun the complete command once from
a clean checkout. Confirm that `environment.json` contains the release commit
and reports `git_worktree_dirty: false`, then commit the regenerated results if
runtime-dependent files changed.
