# Phase 6 — Documentation and Migration Guide

## Purpose

Phase 6 makes the FRSMOTE split understandable for users, developers, and future
paper/release work. No algorithm behavior is changed in this phase.

The main goal is to document the new boundary:

```text
fuzzy_rough_oversampling  --->  FRsutils.api
FRsutils                  -X->  fuzzy_rough_oversampling
```

## Files updated

```text
README.md
fuzzy_rough_oversampling/README.md
docs/frsmote_migration.md
fuzzy_rough_oversampling/docs/migration_from_frsutils.md
fuzzy_rough_oversampling/docs/phase_6_docs_and_migration.md
```

## Documentation changes

- The root README now describes FRsutils as the fuzzy-rough core package.
- The root README no longer presents `imbalanced-learn` and `scikit-learn` as
  mandatory dependencies of FRsutils core.
- The standalone package README now includes:
  - installation steps,
  - direct FRSMOTE usage,
  - registry-based construction,
  - imbalanced-learn Pipeline example,
  - GridSearchCV example,
  - migration instructions from old FRsutils import paths.
- A migration guide documents old and new imports and what moved.

## Migration summary

Old import:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
```

New import:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## Future algorithm guidance

Future oversampling algorithms such as FRADASYN should live under:

```text
fuzzy_rough_oversampling/src/fuzzy_rough_oversampling/algorithms/
```

They should import FRsutils functionality through:

```text
fuzzy_rough_oversampling._frsutils
```

and register themselves using:

```python
from fuzzy_rough_oversampling.registry import register_oversampler
```
