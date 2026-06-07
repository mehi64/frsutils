# Phase 5 — Compatibility Test Coverage

## Purpose

Phase 5 adds practical compatibility coverage for the standalone `fuzzy_rough_oversampling` package after moving FRSMOTE out of `FRsutils`.

The goal is to verify that FRSMOTE is not only importable, but also usable in the workflows expected for an imbalanced-learn sampler:

- direct `fit_resample` usage,
- flat `get_params` / `set_params`,
- `sklearn.base.clone`,
- `imblearn.pipeline.Pipeline`,
- `GridSearchCV`,
- registry-based construction,
- multiple FRsutils fuzzy-rough model types: `itfrs`, `owafrs`, and `vqrs`.

## Files added or updated

```text
fuzzy_rough_oversampling/tests/test_frsmote_fit_resample.py
fuzzy_rough_oversampling/tests/test_frsmote_sklearn_compat.py
fuzzy_rough_oversampling/tests/test_frsmote_pipeline_gridsearch_compat.py
fuzzy_rough_oversampling/src/fuzzy_rough_oversampling/base.py
FRsutils/utils/logger/logger_util.py
```

## Important fix: sklearn clone compatibility

`BaseAllPurposeFuzzyRoughOversampler.get_params()` was adjusted so it returns constructor-compatible flat parameters rather than normalized runtime copies.

This matters because `sklearn.base.clone()` performs a strict identity check between parameters returned by `get_params(deep=False)` and the parameters stored by a newly constructed clone. Returning normalized strings or copied runtime values can make cloning fail even when the estimator is logically correct.

## Important fix: optional colorlog fallback

`FRsutils.utils.logger.logger_util` no longer hard-fails when `colorlog` is not installed. It now falls back to standard `logging.Formatter`.

This keeps downstream package tests importable in minimal environments while preserving colored logs when `colorlog` is available.

## Expected test command

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/fuzzy_rough_oversampling/src" \
python -m pytest fuzzy_rough_oversampling/tests -q
```

Expected result after Phase 5:

```text
13 passed
```

## Boundary note

These tests belong to the standalone oversampling package, not the root `FRsutils` test suite. `FRsutils` should test fuzzy-rough core primitives; `fuzzy_rough_oversampling` should test oversampling behavior and sklearn/imbalanced-learn compatibility.
