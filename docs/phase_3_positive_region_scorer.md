# Phase 3: Positive-region scorer API

Phase 3 adds `FuzzyRoughPositiveRegionScorer` to the public `FRsutils.api` facade.
The scorer is intended for users and downstream libraries that want a reusable
object-oriented API for fuzzy-rough positive-region scores.

## Public contract

```python
from FRsutils.api import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(model="itfrs", similarity="linear")
scores = scorer.fit_score(X, y)
result = scorer.as_result()
```

The scorer depends only on public FRsutils task APIs internally. Downstream
packages should continue to import it from `FRsutils.api`, not from deep internal
paths.

## Current scope

The scorer returns fitted training-set positive-region scores. It does not yet
score unseen samples, because the current fuzzy-rough approximation models are
based on a fitted pairwise similarity matrix.
