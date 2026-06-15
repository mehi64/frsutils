# OWAFRS model

OWAFRS stands for Ordered Weighted Averaging Fuzzy-Rough Sets. It uses OWA
weighting strategies to aggregate sorted lower and upper approximation evidence.
This makes the model more flexible than strict infimum/supremum-style
aggregations and useful for noisy or uncertain data.

## Implementation contract

`frsutils.core.models.OWAFRS` is the dense NumPy reference implementation. It
expects a fully materialized similarity matrix, a one-dimensional label vector,
an upper T-norm, a lower implicator, and lower and upper OWA weighting
strategies.

The direct dense model requires at least two samples because OWAFRS excludes the
self-comparison on the diagonal and then applies OWA aggregation to the remaining
comparisons.

The public approximation API also provides exact blockwise OWAFRS execution:

```python
from frsutils import compute_approximations

result = compute_approximations(X, y, model="owafrs", engine="blockwise")
```

## Approximation outputs

frsutils reports the following public outputs:

```text
boundary_region = upper_approximation - lower_approximation
positive_region = lower_approximation
```

Public output arrays are NumPy arrays.

## Backend status

OWAFRS supports:

- dense NumPy execution,
- exact blockwise NumPy execution,
- optional CuPy-backed similarity blocks.

OWAFRS does not currently claim GPU-resident approximation accumulators. Exact
OWA aggregation requires row-wise sorting and weighted aggregation, so the
current blockwise implementation keeps this aggregation path NumPy-compatible.
