<img title="" src="https://raw.githubusercontent.com/mehi64/frsutils/main/logo/logo.png" alt="frsutils logo" width="220">

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
pip install -e .[dev]
# or, for CUDA 12.x CuPy experiments:
pip install -e .[gpu-cuda12x]
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

## Project boundary

`frsutils` is the fuzzy-rough core library. This keeps `frsutils` focused on reusable fuzzy-rough computations that can be
used by multiple research and application packages.

## Documentation

Start here:

- [Documentation index](docs/README.md)
- [Public API](docs/user/public_api.md)
- [Backends](docs/user/backends.md)
- [Glossary](docs/user/glossary.md)
- [Benchmarks](docs/user/benchmarks.md)
- [Release and JOSS validation](docs/developer/release.md)

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

Before tagging or submitting to JOSS, follow the
[release and JOSS validation guide](docs/developer/release.md).

## License

This project is licensed under the BSD-3-Clause License. See
[LICENSE](LICENSE) for details.

## Citation

If you use `frsutils` in research, cite the software metadata in
[`CITATION.cff`](CITATION.cff). After the JOSS paper is accepted, cite the JOSS
paper DOI as the preferred citation.

```bibtex
@software{Amiri_frsutils_2026,
  author = {Amiri, Mehran},
  title = {frsutils: Fuzzy-Rough Set Utilities for Python},
  url = {https://github.com/mehi64/frsutils},
  version = {0.0.5},
  year = {2026}
}
```
