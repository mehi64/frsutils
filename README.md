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

`frsutils` requires Python 3.10 or newer. The package is distributed through
PyPI; Conda may be used to create the environment, but `frsutils` itself is
installed with `pip`.

### For users

Use an isolated environment before installing the package.

#### Option 1: Conda environment

```bash
conda create -n frsutils python=3.12 -y
conda activate frsutils
python -m pip install --upgrade pip
python -m pip install frsutils
```

#### Option 2: `venv` environment

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install frsutils
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install frsutils
```

Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install frsutils
```

Verify the installed package:

```bash
python -c "import frsutils; from importlib.metadata import version; print(version('frsutils'))"
```

Core requirements are intentionally small:

- Python >= 3.10
- NumPy >= 1.21.0
- scikit-learn

#### Optional CUDA 12 backend

The stable default backend is NumPy. CUDA support is optional and currently
applies only to the documented CuPy-backed blockwise paths. A compatible NVIDIA
driver and CUDA-capable GPU are required.

Install the CUDA 12 extra instead of the plain package install:

```bash
python -m pip install "frsutils[gpu-cuda12x]"
```

Verify device discovery and a real CUDA computation with:

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.asnumpy(cp.arange(5) ** 2))"
```

A result containing one or more devices and the array `[0 1 4 9 16]` confirms
that a CUDA computation completed successfully.

For release-grade evidence, capture the environment and the full model parity
matrix in one archiveable JSON artifact:

```bash
python scripts/capture_cuda_validation.py \
  --require-cuda \
  --output-json cuda_validation_report.json
```

The report records Python, NumPy, CuPy, CUDA runtime/driver, GPU metadata, the
`nvidia-smi`/`nvcc` command outputs, and dense NumPy versus blockwise CuPy
numerical differences for ITFRS, VQRS, and OWAFRS. OWAFRS is reported as using
GPU-backed similarity blocks only; its OWA aggregation remains CPU-resident.

If CuPy or the first numerical CUDA operation fails, inspect the detected GPU and
CuPy configuration first:

```bash
nvidia-smi
python -c "import cupy as cp; cp.show_config()"
```

Prefer reinstalling or upgrading the `frsutils` CUDA extra before manually adding
CUDA component packages:

```bash
python -m pip install --upgrade "frsutils[gpu-cuda12x]"
```

For an existing plain `cupy-cuda12x` installation that reports missing CUDA
headers, install runtime headers matching the local CUDA 12 toolkit minor
version. For example, a CUDA 12.0 environment can use:

```bash
python -m pip install "nvidia-cuda-runtime-cu12==12.0.*"
```

Then run `cupy.show_config()` again and verify a real CUDA operation. See
[backends and execution behavior](docs/user/backends.md) for the validated
environment and model-specific GPU claim boundaries.

### For developers and contributors

Clone the repository and create an isolated development environment. Either
Conda or `venv` can be used.

#### Conda development environment

```bash
git clone https://github.com/mehi64/frsutils.git
cd frsutils
conda create -n frsutils-dev python=3.12 -y
conda activate frsutils-dev
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,study]"
python -m pip check
```

#### `venv` development environment

Linux or macOS:

```bash
git clone https://github.com/mehi64/frsutils.git
cd frsutils
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,study]"
python -m pip check
```

On Windows, activate the environment with one of the following before running
the same `pip` commands:

```powershell
.\.venv\Scripts\Activate.ps1
```

or, in Git Bash:

```bash
source .venv/Scripts/activate
```

The editable install keeps the source checkout linked to the active environment,
so local code changes are immediately visible without reinstalling the package.

Available optional dependency groups are:

| Extra         | Purpose                                   |
| ------------- | ----------------------------------------- |
| `dev`         | Tests and development utilities           |
| `docs`        | MkDocs documentation build                |
| `study`       | Reproducible reference-study dependencies |
| `gpu-cuda12x` | Optional CuPy/CUDA 12 backend             |

For GPU development, install the CUDA extra together with the development
extras:

```bash
python -m pip install -e ".[dev,docs,study,gpu-cuda12x]"
```

For a narrower editable environment, install only the extras needed for the
current task, for example:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
python -m pip install -e ".[study]"
```

### Maintainer and release tooling

Release validation additionally uses `build` and `twine`:

```bash
python -m pip install build twine
rm -rf build dist frsutils.egg-info
python -m build
python -m twine check dist/*
```

Built wheel and source-distribution artifacts should be installation-smoke-tested
in fresh environments before release. For example, on Linux or macOS:

```bash
python -m venv /tmp/frsutils-wheel-smoke
/tmp/frsutils-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-wheel-smoke/bin/python -m pip install dist/frsutils-*.whl
/tmp/frsutils-wheel-smoke/bin/python -c "import frsutils; print(frsutils.__file__)"
/tmp/frsutils-wheel-smoke/bin/python -m pip check
```

Repeat the same check for the source distribution:

```bash
python -m venv /tmp/frsutils-sdist-smoke
/tmp/frsutils-sdist-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-sdist-smoke/bin/python -m pip install dist/frsutils-*.tar.gz
/tmp/frsutils-sdist-smoke/bin/python -c "import frsutils; print(frsutils.__file__)"
/tmp/frsutils-sdist-smoke/bin/python -m pip check
```

On Windows, use `Scripts/python.exe` instead of `bin/python` for these isolated
artifact checks. Detailed test and release procedures remain in
[`tests/test_procedures.md`](tests/test_procedures.md) and
[`docs/developer/release.md`](docs/developer/release.md).

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
print(result.signed_boundary)
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
    compute_signed_boundary,
)
```

Main user-facing entry points include:

- `compute_approximations`
- `compute_lower_approximation`
- `compute_upper_approximation`
- `compute_signed_boundary`
- `compute_boundary_region`
- `compute_positive_region`
- `build_similarity_matrix`
- `build_similarity_engine`
- `build_fuzzy_rough_model`
- `FuzzyRoughPositiveRegionScorer`

See the [public API guide](docs/user/public_api.md) for the full public boundary
and downstream-package contract. Component selectors, accepted aliases,
parameter prefixes, and model-specific options are documented in the
[public configuration contract](docs/user/configuration.md). Common terms are
defined in the [glossary](docs/user/glossary.md).

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
- [Fuzzy quantifiers](docs/concepts/fuzzy_quantifiers_info.md)

## Execution modes and backends

`frsutils` supports **dense** and exact **blockwise** execution through the public API. Dense execution processes the full data at once, while blockwise execution splits the computation into smaller exact chunks to reduce memory usage without changing the result. Dense NumPy is the stable reference path. Blockwise execution can reduce memory pressure by avoiding materialization of a full `n x n` similarity matrix.

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

When CuPy is available, selected blockwise computations can execute through CuPy on compatible hardware. Performance is workload-dependent, and acceleration is not guaranteed. CuPy is optional and limited to the documented backend-aware paths. The stable default backend is NumPy, and public result arrays remain NumPy arrays. See [backends and execution behavior](docs/user/backends.md) for the precise model-specific claim boundaries.

## Benchmarking

The repository includes a benchmark harness for dense NumPy, blockwise NumPy, and optional CuPy-backed blockwise execution. See the [performance benchmarking guide](docs/user/performance_benchmarking.md) for larger synthetic runs, paired NumPy/CuPy comparisons, and interpretation rules.

## Reproducible reference study

The repository includes a real-dataset research artifact that applies ITFRS,
VQRS, and OWAFRS through the stable package-root API, verifies exact
dense/blockwise agreement, records repeated runtimes and per-sample outputs,
and captures the execution environment:

After installing the developer environment with the `study` extra, run:

```bash
python studies/fuzzy_rough_reference_study/run_study.py
```

See the [reference-study documentation](studies/fuzzy_rough_reference_study/README.md)
and the created documents in the results folder.

## Project boundary

`frsutils` is the fuzzy-rough core library. This keeps `frsutils` focused on reusable fuzzy-rough computations that can be used by multiple research and application packages.

## Documentation

The published documentation is available at
[mehi64.github.io/frsutils](https://mehi64.github.io/frsutils/).

Repository sources:

- [Documentation index](docs/index.md)
- [Public API](docs/user/public_api.md)
- [Backends](docs/user/backends.md)
- [Glossary](docs/user/glossary.md)
- [Performance benchmarking](docs/user/performance_benchmarking.md)
- [Reproducible reference study](docs/user/reference_study.md)
- [Maintainer release process](docs/developer/release.md)
- [Software archiving and DOI metadata](docs/developer/archive_and_doi.md)

After installing the developer environment with the `docs` extra, build the
documentation locally with:

```bash
mkdocs build --strict
mkdocs serve
```

## Development and validation

After completing the developer installation above, run the main test suite from
the repository root:

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

Maintainers can use the following repository guides for repeatable releases and
software archiving:

- [Maintainer release process](docs/developer/release.md)
- [Software archiving and DOI metadata](docs/developer/archive_and_doi.md)

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
[`CITATION.cff`](CITATION.cff). When the citation metadata contains a
version-specific archive DOI, use that DOI to identify the executable release.

```bibtex
@software{Amiri_frsutils_2026,
  author = {Amiri, Mehran},
  title = {frsutils: Fuzzy-Rough Set Utilities for Python},
  url = {https://github.com/mehi64/frsutils},
  doi = {https://doi.org/10.5281/zenodo.21441122}
  version = {0.1.1},
  year = {2026}
}
```
