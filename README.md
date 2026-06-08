<img src="images/logo/logo4.png" alt="FRsutils Logo" width="250"/>

# FRsutils

FRsutils is a Python library for reusable fuzzy-rough set utilities. The package
focuses on fuzzy-rough core building blocks such as similarity matrices,
t-norms, implicators, fuzzy quantifiers, fuzzy-rough models, lower/upper
approximations, and positive-region computation.

Fuzzy-rough oversampling algorithms such as FRSMOTE now live in the standalone
`frsampling` package. That package depends on FRsutils through the
public `FRsutils.api` facade.

# For Developers

If you are a developer trying to extend FRsutils, please start here:
[Development Guidelines](lessons_learned_and_trainings/For_Developers.md).

# Installation

Install the fuzzy-rough core package:

```bash
pip install frsutils
```

For local development from this repository:

```bash
pip install -e .
```

## Core requirements

FRsutils core intentionally keeps the mandatory dependency set small, but the
public API includes an sklearn-style positive-region scorer, so scikit-learn is
part of the runtime contract:

- Python >= 3.10
- NumPy >= 1.21.0
- scikit-learn

## Optional development / dataset / GPU dependencies

Install these only when you need the related workflows:

- `pytest` for tests
- `pandas`, `openpyxl` for some dataset utilities
- `colorlog` for colored logging; FRsutils falls back to standard logging if it
  is not installed
- `matplotlib` for plotting examples/tests
- `cupy-cuda12x` or another CUDA-compatible CuPy wheel for explicit
  `backend="cupy"` experiments

## Oversampling package

Install the standalone oversampling package when you need FRSMOTE or future
fuzzy-rough oversamplers such as FRADASYN:

```bash
cd frsampling
pip install -e .
```

Use the new import path:

```python
from frsampling import FRSMOTE
```

No backward-compatibility wrapper is kept in FRsutils for the old FRSMOTE import paths. Existing scripts must be migrated explicitly. See the `frsampling` repository migration guide for FRSMOTE-specific migration details.

# Fuzzy-Rough set utilities [Under development]

A basic Python library needed for fuzzy rough set calculations that are used in
research, e.g.:

- lower approximation
- upper approximation
- positive region
- boundary region


## Public API quickstart

FRsutils exposes its stable user and downstream-package interface through
`FRsutils.api`. End users should prefer the task-oriented helpers in this
namespace. Downstream packages, including `frsampling`, should also depend only
on this facade instead of importing from `FRsutils.core` or `FRsutils.utils`.

The smallest end-user workflow is: prepare normalized numeric data, compute
fuzzy-rough approximations, and read the named fields from the result object.

```python
import numpy as np

from FRsutils.api import compute_approximations, compute_positive_region

# FRsutils expects numeric feature values on a comparable scale. In real
# experiments, normalize or scale your data before calling the fuzzy-rough API.
X = np.array(
    [
        [0.00, 0.10],
        [0.10, 0.20],
        [0.85, 0.80],
        [0.95, 0.90],
    ],
    dtype=float,
)
y = np.array([0, 0, 1, 1])

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
)

print("lower approximation:", result.lower)
print("upper approximation:", result.upper)
print("boundary region:", result.boundary)
print("positive region:", result.positive_region)

# Shortcut when only positive-region scores are needed.
scores = compute_positive_region(
    X,
    y,
    model="itfrs",
    similarity="linear",
)
print("positive-region scores:", scores)
```

For reusable fitted scoring workflows, use the sklearn-style positive-region
scorer:

```python
import numpy as np

from FRsutils.api import FuzzyRoughPositiveRegionScorer

X = np.array(
    [
        [0.00, 0.10],
        [0.10, 0.20],
        [0.85, 0.80],
        [0.95, 0.90],
    ],
    dtype=float,
)
y = np.array([0, 0, 1, 1])

scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="linear",
)

scores = scorer.fit_score(X, y)
result = scorer.as_result()

print(scores)
print(result.lower)
print(result.upper)
```

Downstream packages should use the builder-level public API when they need to
reuse a precomputed similarity matrix. This keeps external packages independent
from FRsutils internals while avoiding repeated similarity-matrix construction.

```python
import numpy as np

from FRsutils.api import build_similarity_matrix, compute_positive_region

X = np.array(
    [
        [0.00, 0.10],
        [0.10, 0.20],
        [0.85, 0.80],
        [0.95, 0.90],
    ],
    dtype=float,
)
y = np.array([0, 0, 1, 1])

similarity_matrix = build_similarity_matrix(
    X,
    similarity="linear",
)

scores = compute_positive_region(
    X=None,
    y=y,
    model="itfrs",
    similarity_matrix=similarity_matrix,
)

print(scores)
```

See [`docs/public_api_contract.md`](docs/public_api_contract.md) for the stable
public API boundary and downstream-package rules.

## Execution engines and backend status

FRsutils now exposes dense and exact blockwise execution through the public API.
Dense mode preserves the historical full-matrix behavior. Blockwise mode avoids
materializing the full `n x n` similarity matrix for approximation computation
and is available for ITFRS, VQRS, and OWAFRS.

```python
from FRsutils.api import compute_approximations

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

`backend="cupy"` is an optional experimental backend for GPU-accelerated
similarity-block computation. For `model="itfrs"` and `model="vqrs"` with
`engine="blockwise"`, approximation reductions/accumulators can also stay
CuPy-resident until final public NumPy output conversion. OWAFRS deliberately remains on the conservative NumPy row-buffer path after the
Phase 5 decision because exact OWA execution requires row-wise sorting and a
separate memory/sorting benchmark. Do not claim full GPU-native fuzzy-rough
execution yet. See
[`docs/backend_execution_status.md`](docs/backend_execution_status.md) and
[`docs/phase_5_owafrs_non_gpu_resident_decision.md`](docs/phase_5_owafrs_non_gpu_resident_decision.md).

The returned result records execution provenance so benchmark scripts and
downstream packages can verify which path was used:

```python
result.engine                      # "dense" or "blockwise"
result.backend                     # "numpy" or resolved optional backend
result.block_size                  # None for dense; integer for blockwise
result.used_blockwise              # bool
result.used_gpu_similarity_blocks          # bool
result.used_gpu_approximation_accumulators # bool, true for CuPy blockwise ITFRS/VQRS; false for OWAFRS
```

The sklearn-style `FuzzyRoughPositiveRegionScorer` accepts the same `engine`,
`backend`, and `block_size` parameters.


## Benchmark suite

Phase 6 adds a reproducible benchmark harness for the public approximation API:

```bash
python benchmarks/benchmark_fuzzy_rough_execution.py     --models itfrs,vqrs,owafrs     --sample-sizes 128,256,512     --n-features 8     --block-sizes 64,128     --scenarios dense_numpy,blockwise_numpy,blockwise_cupy     --repeats 3     --output-json benchmark_results.json     --output-csv benchmark_results.csv
```

The suite compares dense NumPy, exact blockwise NumPy, and optional CuPy-backed
blockwise execution. It records runtime, lightweight Python allocator peak
memory, dense-reference numerical-equivalence errors, and public execution
metadata. CuPy/CUDA-unavailable rows are reported as skipped. See
[`docs/phase_6_benchmark_suite.md`](docs/phase_6_benchmark_suite.md).


## Release-ready examples and paper claim boundary

Phase 7 adds two small release-ready examples:

```bash
python examples/phase7_public_api_quickstart.py
python examples/phase7_benchmark_smoke.py --output-dir phase7_benchmark_smoke_output
```

Use the wording in [`docs/paper_claims.md`](docs/paper_claims.md) when describing
FRsutils in a release note, software paper, or benchmark report. The safe claim
is that FRsutils provides dense and exact blockwise fuzzy-rough approximation
APIs, optional CuPy-accelerated similarity blocks, and experimental
CuPy-resident blockwise approximation accumulators for ITFRS/VQRS. Public
outputs remain NumPy arrays, and OWAFRS remains on the conservative exact
blockwise NumPy row-buffer path in this release.

Before tagging or submitting, use
[`docs/release_checklist.md`](docs/release_checklist.md).

## Algorithms and contents

- Similarities (See [fuzzy similarities](docs/similarities_info.md))
  - Linear
  - Gaussian
- Implicators (See [fuzzy implicators](docs/implicators_info.md))
  - Lukasiewicz
  - Goedel
  - Reichenbach
  - Kleene-Dienes
  - Goguen
  - Yager
  - Rescher
  - Weber
  - Fodor
- T-norms (See [fuzzy tnorms](docs/tnorms_info.md))
  - Min tnorm
  - Product tnorm
  - Lukasiewicz tnorm
  - Yager tnorm
  - DrasticProduct tnorm
  - EinsteinProduct tnorm
  - HamacherProduct tnorm
  - NilpotentMinimum tnorm
- OWA weights (Ordered Weighted Average) (See [OWA](docs/owa_weights_info.md))
  - Linear
  - Exponential
  - Harmonic
  - Logarithmic
- Fuzzy quantifiers
  - Linear
  - Quadratic
- FR Models
  - ITFRS (See [Implicator/T-norm Fuzzy-Rough Sets](docs/itfrs_info.md))
  - OWAFRS (See [Ordered Weighted Average Fuzzy-Rough Sets](docs/owafrs_info.md))
  - VQRS (See [Vaguely Quantified fuzzy-Rough Sets](docs/vqrs_info.md))

## Fuzzy-rough oversampling boundary

Fuzzy-rough oversampling algorithms are no longer part of FRsutils core. They
live in the standalone `frsampling` package, which depends on
FRsutils through the public `FRsutils.api` facade. FRsutils intentionally does
not provide old FRSMOTE compatibility wrappers.

```text
frsampling  --->  FRsutils.api
FRsutils    -X->  frsampling
```

FRsutils should be cited/used as the fuzzy-rough core engine: similarities,
t-norms, implicators, fuzzy quantifiers, approximation models, lower/upper
approximation, and positive region. Oversampling algorithms such as FRSMOTE and
future FRADASYN belong to the downstream oversampling package.

## Notes and assumptions

- All functions expect normalized scalar values or normalized NumPy arrays.
- Make sure the input dataset is normalized. This library expects numeric inputs
  used by fuzzy-rough computations to be in the range [0, 1].
- This library uses all features of data instances to calculate fuzzy-rough
  measures.
- Positive region, lower approximation, upper approximation, etc. are calculated
  based on the class of each instance.

## Docs

- We use Doxygen-style Python docstrings and documents are generated by Doxygen.
- To see online documentation, please visit
  [online documentation](https://mehi64.github.io/FRsutils/).

## How to run tests

From the repository root, the default test command excludes tests marked as
`slow` via `pyproject.toml`:

```bash
python -m pytest tests -q
```

Run the release/backend smoke set explicitly with:

```bash
python -m pytest tests/api tests/benchmarks tests/examples -q
```

Run exhaustive slow model-combination tests separately when needed:

```bash
python -m pytest tests/models_tests -m slow -q
```

For the standalone oversampling package, run from the `frsampling` repository
root after making FRsutils importable:

```bash
PYTHONPATH="$PWD/src:../FRsutils" python -m pytest tests -q
```

For more information on test procedures, please refer to
[test procedures](tests/test_procedures.md).

## Technical decisions justification

- Since data checking can slow down experiments, heavy numeric functions do not
  perform repeated input-range checks. Validation is preferred at construction or
  workflow boundaries.

## Maintenance notes

- Exhaustive model-combination tests are marked as `slow`; run them explicitly
  with `python -m pytest tests/models_tests -m slow -q`.
- VQRS is implemented and covered by the public API/blockwise/backend tests.
- New feature work should be deferred until the release/paper cleanup checklist is
  complete.

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](./LICENSE)
file for details.

## How to cite us in your research papers

If you use this library in your research, please cite it as follows:

**APA**:

> Mehran Amiri. (*2025*). *FRsutils* (Version 0.0.3) [Computer software]. https://github.com/mehi64/FRsutils

**BibTeX**:

```bibtex
@software{Mehran_Amiri_FRsutils_2025,
  author = {Amiri, Mehran},
  title = {FRsutils},
  url = {https://github.com/mehi64/FRsutils},
  version = {0.0.3},
  year = {2025}
}
```
