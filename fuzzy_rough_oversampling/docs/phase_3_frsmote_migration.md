# Phase 3: FRSMOTE Migration

Phase 3 moves the operational FRSMOTE implementation into the standalone
`fuzzy-rough-oversampling` package while keeping FRsutils as the fuzzy-rough core
engine.

## New public imports

```python
from fuzzy_rough_oversampling import FRSMOTE
from fuzzy_rough_oversampling import build_oversampler

sampler = FRSMOTE(random_state=42)
sampler2 = build_oversampler("frsmote", random_state=42)
```

## Dependency direction

```text
fuzzy_rough_oversampling  --->  FRsutils.api
FRsutils                  -X->  fuzzy_rough_oversampling
```

## What moved in this phase

- FRSMOTE implementation moved to
  `fuzzy_rough_oversampling.algorithms.frsmote`.
- The old FRsutils oversampler base logic was copied into the standalone package
  as package-local base classes.
- FRSMOTE now imports FRsutils functionality only through
  `fuzzy_rough_oversampling._frsutils`, which itself imports from `FRsutils.api`.
- Local validation and sampling helpers were added so FRSMOTE does not depend on
  FRsutils internal utility paths.

## What did not happen yet

The old FRSMOTE files inside `FRsutils/core/preprocess` are intentionally not
removed in this phase. Removing or replacing old imports belongs to Phase 4.
