<img title="" src="https://raw.githubusercontent.com/mehi64/frsutils/main/logo/logo.png" alt="frsutils logo" width="220">

[![CI](https://github.com/mehi64/frsutils/actions/workflows/ci.yml/badge.svg)](https://github.com/mehi64/frsutils/actions/workflows/ci.yml)
[![Documentation](https://github.com/mehi64/frsutils/actions/workflows/docs.yml/badge.svg)](https://github.com/mehi64/frsutils/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/frsutils.svg)](https://pypi.org/project/frsutils/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

# frsutils

`frsutils` is a scientific Python library for fuzzy-rough set computations. It provides reusable building blocks and task-oriented APIs for similarity construction, fuzzy-rough lower and upper approximations, boundary regions, and
positive-region scores.

The package is intended for researchers and downstream libraries that need a small, documented fuzzy-rough core with a stable public API.

## Installation

Install from PyPI:

```bash
pip install frsutils
```

For local development from a source checkout:

```bash
pip install -e .
```

Core requirements are intentionally small:

- Python >= 3.10
- NumPy >= 1.21.0
- scikit-learn

Optional extras are used only for specific workflows:

```bash
pip install -e ".[dev]"
pip install -e ".[docs]"
# or, for CUDA 12.x CuPy experiments:
pip install -e ".[gpu-cuda12x]"
```

## Quick start

Use the package root as the canonical public API:

```python
import numpy as np
from frsutils import compute_approximations, compute_positive_region

X = np.array(
    [
        [0.00, 0.10],
        [0.08, 0.18],
        [0.15, 0.12],
        [0.80, 0.82],
        [0.88, 0.90],
        [0.95, 0.86],
    ],
    dtype=float,
)
y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
)

print(result.lower)
print(result.upper)
print(result.boundary)
print(result.positive_region)

scores = compute_positive_region(X, y, model="itfrs", similarity="linear")
print(scores)
```

A runnable version is available in
[`examples/public_api_quickstart.py`](examples/public_api_quickstart.py).

## Public API

Prefer imports from `frsutils` instead of deep internal modules:

```python
from frsutils import (
    FuzzyRoughPositiveRegionScorer,
    build_similarity_matrix,
    compute_approximations,
    compute_positive_region,
)
```

Main user-facing entry points include:

- `compute_approximations`
- `compute_lower_approximation`
- `compute_upper_approximation`
- `compute_boundary_region`
- `compute_positive_region`
- `build_similarity_matrix`
- `build_similarity_engine`
- `build_fuzzy_rough_model`
- `FuzzyRoughPositiveRegionScorer`

See the [public API guide](docs/user/public_api.md) for the full public boundary
and downstream-package contract. Common terms are defined in the
[glossary](docs/user/glossary.md).

## Supported fuzzy-rough models

Current public model aliases are:

- `"itfrs"` — Implicator/T-norm Fuzzy-Rough Sets
- `"vqrs"` — Vaguely Quantified Rough Sets
- `"owafrs"` — Ordered Weighted Average Fuzzy-Rough Sets

Concept notes are kept separate for review and maintenance:

- [ITFRS](docs/concepts/itfrs_info.md)
- [VQRS](docs/concepts/vqrs_info.md)
- [OWAFRS](docs/concepts/owafrs_info.md)
- [Similarities](docs/concepts/similarities_info.md)
- [T-norms](docs/concepts/tnorms_info.md)
- [Implicators](docs/concepts/implicators_info.md)
- [OWA weights](docs/concepts/owa_weights_info.md)

## Execution modes and backends

`frsutils` supports dense and exact blockwise execution through the public API.
Dense NumPy is the stable reference path. Blockwise execution can reduce memory pressure by avoiding materialization of a full `n x n` similarity matrix.

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    block_size=512,
    backend="numpy",
)
```

CuPy is optional and currently limited to selected backend-aware computation paths. The stable default backend is NumPy. Public result arrays are always NumPy arrays, even when a CuPy-backed blockwise path is used internally. See [backends and execution behavior](docs/user/backends.md) for the precise backend
claim boundaries.

## Benchmarking

The repository includes a benchmark harness for dense NumPy, blockwise NumPy, and optional CuPy-backed blockwise execution:

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

See the [benchmark guide](docs/user/benchmarks.md) for larger synthetic runs,
paired NumPy/CuPy comparisons, and interpretation rules.

## Reproducible reference study

The repository includes a real-dataset research artifact that applies ITFRS,
VQRS, and OWAFRS through the stable package-root API, verifies exact
dense/blockwise agreement, records repeated runtimes and per-sample outputs,
and captures the execution environment:

```bash
python -m pip install -e ".[study]"
python studies/fuzzy_rough_reference_study/run_study.py
```

The study uses public scikit-learn datasets and does not depend on FRSMOTE or
any unpublished downstream code. See the
[reference-study documentation](studies/fuzzy_rough_reference_study/README.md)
and the [committed result snapshot](studies/fuzzy_rough_reference_study/results/README.md).

## Project boundary

`frsutils` is the fuzzy-rough core library. This keeps `frsutils` focused on reusable fuzzy-rough computations that can be
used by multiple research and application packages.

## Documentation

The published documentation is available at
[mehi64.github.io/frsutils](https://mehi64.github.io/frsutils/).

Repository sources:

- [Documentation index](docs/index.md)
- [Public API](docs/user/public_api.md)
- [Backends](docs/user/backends.md)
- [Glossary](docs/user/glossary.md)
- [Benchmarks](docs/user/benchmarks.md)
- [Reproducible reference study](docs/user/reference_study.md)
- [Release and JOSS validation](docs/developer/release.md)
- [Final JOSS submission checklist](docs/developer/joss_submission_checklist.md)
- [Software archive and DOI guide](docs/developer/archive_and_doi.md)

Build the documentation locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
mkdocs serve
```

## Development and validation

Run the main test suite from the repository root:

```bash
python -m pytest tests -q
```

Run public API and backend-focused smoke checks:

```bash
python examples/public_api_quickstart.py
python -m pytest tests/api/test_public_api_examples_smoke.py -q -rs
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
```

Run slow model-combination tests only when needed:

```bash
python -m pytest tests/models_tests -m slow -o addopts="" -q
```

Before tagging or submitting to JOSS, run the automated validator and follow
the maintainer guides:

```bash
python scripts/validate_joss_submission.py
```

- [Release and JOSS validation](docs/developer/release.md)
- [Final JOSS submission checklist](docs/developer/joss_submission_checklist.md)
- [Software archive and DOI guide](docs/developer/archive_and_doi.md)

## Community and project governance

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Support policy](SUPPORT.md)
- [Changelog](CHANGELOG.md)
- [Issue tracker](https://github.com/mehi64/frsutils/issues)

Bug reports, usage questions, and scientific enhancement proposals should use
the structured issue forms. Public contributions are reviewed through pull
requests and the automated CI checks.

## License

This project is licensed under the BSD-3-Clause License. See
[LICENSE](LICENSE) for details.

## Citation

If you use `frsutils` in research, cite the exact software release described in
[`CITATION.cff`](CITATION.cff). The version-specific Zenodo DOI will be added
after release archival. After the JOSS article is accepted, its DOI will be
recorded as the preferred citation while the software DOI continues to identify
the archived executable artifact.

```bibtex
@software{Amiri_frsutils_2026,
  author = {Amiri, Mehran},
  title = {frsutils: Fuzzy-Rough Set Utilities for Python},
  url = {https://github.com/mehi64/frsutils},
  version = {0.1.0},
  year = {2026}
}
```
