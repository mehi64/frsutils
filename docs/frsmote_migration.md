# FRSMOTE Migration from FRsutils to fuzzy_rough_oversampling

## Purpose

FRSMOTE has been moved out of FRsutils core into the standalone
`fuzzy_rough_oversampling` package.

This keeps FRsutils focused on reusable fuzzy-rough primitives and models, while
oversampling algorithms live in a downstream package that depends on FRsutils
through the public `FRsutils.api` facade.

## Dependency direction

```text
fuzzy_rough_oversampling  --->  FRsutils.api
FRsutils                  -X->  fuzzy_rough_oversampling
```

## Install locally

From the repository root:

```bash
pip install -e .
cd fuzzy_rough_oversampling
pip install -e .
```

## Import change

Phase 7 uses a hard-break migration policy. No backward-compatibility wrapper is provided in `FRsutils`. Old imports must be replaced.

Old import:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
```

New import:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## Registry-based construction

```python
from fuzzy_rough_oversampling import build_oversampler

sampler = build_oversampler("frsmote", random_state=42)
```

## Pipeline migration

Old style:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline([
    ("frsmote", FRSMOTE(random_state=42)),
    ("classifier", SVC()),
])
```

New style:

```python
from fuzzy_rough_oversampling import FRSMOTE
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

pipe = Pipeline([
    ("frsmote", FRSMOTE(random_state=42)),
    ("classifier", SVC()),
])
```

The estimator name inside the pipeline can stay the same. Only the import path
changes. The old import path is not supported by a compatibility shim.

## GridSearchCV migration

GridSearchCV parameter names do not change as long as the pipeline step name is
still `"frsmote"`.

```python
param_grid = {
    "frsmote__k_neighbors": [3, 5],
    "frsmote__similarity": ["gaussian"],
    "frsmote__similarity_sigma": [0.2, 0.5],
}
```



## Backward compatibility policy

No backward-compatibility wrappers are kept for FRSMOTE in `FRsutils`.

These old paths should fail after Phase 7:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.preprocess.oversampling import FRSMOTE
from FRsutils.core.preprocess.FRSMOTE import FRSMOTE
```

Use only:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## What moved

Moved to `fuzzy_rough_oversampling`:

```text
FRSMOTE
FRSMOTE-specific base oversampler behavior
FRSMOTE tests
FRSMOTE sklearn/imbalanced-learn compatibility checks
```

Stays in FRsutils:

```text
similarity matrix construction
fuzzy-rough models such as ITFRS, OWAFRS, VQRS
t-norms, implicators, OWA weights, fuzzy quantifiers
lower approximation, upper approximation, positive region
flat-to-nested fuzzy-rough config normalization
```

## What to update in old scripts

Search for old imports:

```bash
grep -R "FRsutils.core.preprocess.*FRSMOTE\|FRsutils.core.preprocess.oversampling" .
```

Replace them with:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## Test command

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/fuzzy_rough_oversampling/src" \
python -m pytest fuzzy_rough_oversampling/tests -q
```
