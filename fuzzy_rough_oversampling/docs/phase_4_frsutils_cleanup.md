# Phase 4: FRsutils Cleanup

Phase 4 removes the old oversampling implementation from `FRsutils` so the
project boundary is clear:

```text
fuzzy_rough_oversampling  --->  FRsutils
FRsutils                 -X->  fuzzy_rough_oversampling
```

## What moved out of FRsutils

The following FRsutils-side files are no longer part of the core package:

```text
FRsutils/core/preprocess/oversampling/FRSMOTE.py
FRsutils/core/preprocess/oversampling/FR_ADASYN.py
FRsutils/core/preprocess/base_allpurpose_fuzzy_rough_oversampler.py
FRsutils/core/preprocess/base_solo_fuzzy_rough_oversampler.py
FRsutils/core/preprocess/base_fuzzy_rough_generative_oversampler.py
```

FRSMOTE now lives in:

```text
fuzzy_rough_oversampling/src/fuzzy_rough_oversampling/algorithms/frsmote.py
```

## Import migration

Old import:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
```

New import:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## Test migration

FRSMOTE-specific tests were removed from the FRsutils root test suite and moved
under the standalone package test suite.

## Boundary rule

FRsutils must not import the standalone oversampling package. Oversampling
algorithms may import FRsutils only through `FRsutils.api` or the local bridge
module `fuzzy_rough_oversampling._frsutils`.
