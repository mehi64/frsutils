# Fuzzy-Rough Oversampling

`fuzzy-rough-oversampling` is a standalone package for fuzzy-rough oversampling
algorithms built on top of `FRsutils`.

The package currently contains `FRSMOTE` and is structured so future
fuzzy-rough versions of oversampling algorithms, such as `FRADASYN`, can be
added under `fuzzy_rough_oversampling.algorithms`.

## Package boundary

```text
fuzzy_rough_oversampling  --->  FRsutils.api
FRsutils                  -X->  fuzzy_rough_oversampling
```

`FRsutils` owns the fuzzy-rough core:

- similarity matrix construction,
- t-norms and implicators,
- OWA weights and fuzzy quantifiers,
- fuzzy-rough models such as ITFRS, OWAFRS, and VQRS,
- lower/upper approximation and positive region.

`fuzzy_rough_oversampling` owns the imbalanced-learning algorithms:

- FRSMOTE,
- future FRADASYN,
- future fuzzy-rough oversamplers,
- `scikit-learn` / `imbalanced-learn` compatibility tests.

## Installation for local development

From the repository root, install FRsutils first:

```bash
pip install -e .
```

Then install the oversampling package:

```bash
cd fuzzy_rough_oversampling
pip install -e .
```

## Basic usage

```python
from fuzzy_rough_oversampling import FRSMOTE

sampler = FRSMOTE(random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

FRSMOTE expects numeric feature values compatible with the FRsutils fuzzy-rough
core assumptions. In the current design, feature values should be normalized to
`[0, 1]` before fuzzy-rough similarity/approximation computations are used.

## Registry-based usage

The package exposes a small registry/factory API so future algorithms can be
added without changing user code patterns.

```python
from fuzzy_rough_oversampling import build_oversampler, list_oversamplers

print(list_oversamplers())

sampler = build_oversampler("frsmote", random_state=42)
X_resampled, y_resampled = sampler.fit_resample(X, y)
```

FRSMOTE is registered under these aliases:

```text
frsmote
fr_smote
fuzzy_rough_smote
fuzzy-rough-smote
```

## imbalanced-learn Pipeline example

```python
from fuzzy_rough_oversampling import FRSMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline([
    ("frsmote", FRSMOTE(random_state=42)),
    ("classifier", SVC()),
])

pipe.fit(X_train, y_train)
```

## GridSearchCV example

```python
from fuzzy_rough_oversampling import FRSMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe = Pipeline([
    ("frsmote", FRSMOTE(random_state=42)),
    ("classifier", SVC()),
])

param_grid = {
    "frsmote__k_neighbors": [3, 5],
    "frsmote__similarity": ["gaussian"],
    "frsmote__similarity_sigma": [0.2, 0.5],
    "classifier__C": [0.1, 1.0],
}

search = GridSearchCV(pipe, param_grid=param_grid, cv=3)
search.fit(X_train, y_train)
```

## Migration from old FRsutils imports

Phase 7 uses a hard-break migration policy. FRsutils does not keep compatibility wrappers for old FRSMOTE import paths.

Old import:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
```

New import:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

Detailed migration guide:

- [Migration from FRsutils](docs/migration_from_frsutils.md)

Old imports are expected to fail after Phase 7. Update scripts explicitly to use `from fuzzy_rough_oversampling import FRSMOTE`.

## Running tests

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/fuzzy_rough_oversampling/src" \
python -m pytest fuzzy_rough_oversampling/tests -q
```

Expected Phase 7 baseline:

```text
16 passed
```

## Development rule for future algorithms

Future algorithms such as FRADASYN should:

1. live under `fuzzy_rough_oversampling.algorithms`,
2. inherit from the local base classes in `fuzzy_rough_oversampling.base` when useful,
3. import FRsutils functionality through `fuzzy_rough_oversampling._frsutils`,
4. register themselves using `register_oversampler`,
5. preserve flat sklearn/GridSearchCV-compatible parameters.
