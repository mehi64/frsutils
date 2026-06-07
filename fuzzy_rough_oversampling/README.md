# Fuzzy-Rough Oversampling

`fuzzy-rough-oversampling` is the planned standalone package for fuzzy-rough
oversampling algorithms built on top of `FRsutils`.

The package is intentionally separated from `FRsutils` so that:

- `FRsutils` can stay focused on fuzzy-rough primitives, models, similarity
  matrices, lower/upper approximations, and positive-region computation.
- this package can focus on imbalanced-learning algorithms such as `FRSMOTE`,
  `FRADASYN`, and future fuzzy-rough versions of oversampling methods.
- `scikit-learn` / `imbalanced-learn` dependencies stay in the oversampling
  package instead of becoming mandatory for the FRsutils core library.

## Intended dependency direction

```text
fuzzy_rough_oversampling  --->  FRsutils
FRsutils                  -X->  fuzzy_rough_oversampling
```

## Planned algorithms

- `FRSMOTE`: fuzzy-rough positive-region guided SMOTE.
- `FRADASYN`: fuzzy-rough version of ADASYN.
- future fuzzy-rough oversamplers can be added under
  `fuzzy_rough_oversampling.algorithms`.

## Current phase

Phase 3 has migrated FRSMOTE into this standalone package. FRADASYN and other
fuzzy-rough oversamplers can be added later under
`fuzzy_rough_oversampling.algorithms`.

## Local development

Install FRsutils first, then install this package in editable mode:

```bash
pip install -e ..
pip install -e .
```

## Usage

```python
from fuzzy_rough_oversampling import FRSMOTE, build_oversampler

sampler = FRSMOTE(random_state=42)
X_res, y_res = sampler.fit_resample(X, y)

sampler2 = build_oversampler("frsmote", random_state=42)
```
