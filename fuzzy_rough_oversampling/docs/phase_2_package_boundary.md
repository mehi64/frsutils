# Phase 2 Package Boundary

This package is the application layer for fuzzy-rough oversampling algorithms.
It should depend on the public FRsutils API added in Phase 1.

## Dependency direction

```text
fuzzy_rough_oversampling  --->  FRsutils.api
FRsutils                  -X->  fuzzy_rough_oversampling
```

## What belongs here

- FRSMOTE
- FRADASYN
- future fuzzy-rough oversampling algorithms
- imbalanced-learn / scikit-learn compatibility tests for oversamplers
- oversampling-specific benchmark scripts

## What stays in FRsutils

- similarity functions and similarity matrix construction
- t-norms, implicators, OWA weights, fuzzy quantifiers
- ITFRS, OWAFRS, VQRS, and other fuzzy-rough models
- lower approximation, upper approximation, and positive region computation
- config normalization needed by fuzzy-rough core consumers

## Phase 3 target

Move FRSMOTE implementation and required oversampler base classes into this
package. FRSMOTE should import fuzzy-rough functionality through
`fuzzy_rough_oversampling._frsutils`, not through deep FRsutils internal paths.
