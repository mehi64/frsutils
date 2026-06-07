# Phase 7: No Backward Compatibility Wrappers

## Decision

Phase 7 finalizes the FRSMOTE split with a hard-break migration policy.
FRsutils does not keep compatibility wrappers for old FRSMOTE import paths.

The only supported import is:

```python
from fuzzy_rough_oversampling import FRSMOTE
```

## Unsupported old imports

These paths are intentionally unsupported:

```python
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.preprocess.oversampling import FRSMOTE
from FRsutils.core.preprocess.FRSMOTE import FRSMOTE
```

## Rationale

Keeping wrappers would preserve old scripts temporarily, but it would also keep
FRsutils coupled to the downstream oversampling package. The purpose of the
split is to keep FRsutils as the fuzzy-rough core engine and move algorithms
such as FRSMOTE, FRADASYN, and future fuzzy-rough oversamplers into the
standalone `fuzzy_rough_oversampling` package.

## Test coverage

The standalone package contains a hard-break import policy test:

```text
fuzzy_rough_oversampling/tests/test_frsmote_no_backward_compat.py
```

That test verifies that old FRsutils FRSMOTE import paths fail and the new
standalone import path remains the intended public API.
