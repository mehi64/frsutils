# ITFRS model

ITFRS stands for Implicator/T-norm Fuzzy-Rough Sets. In FRsutils, ITFRS computes
lower and upper approximation degrees from a fuzzy similarity relation and class
labels using a lower implicator and an upper T-norm.

## Implementation contract

`FRsutils.core.models.ITFRS` is the dense NumPy reference implementation. It
expects a fully materialized similarity matrix, a one-dimensional label vector,
an upper T-norm, and a lower implicator.

The public approximation API also provides exact blockwise ITFRS execution:

```python
from FRsutils.api import compute_approximations

result = compute_approximations(X, y, model="itfrs", engine="blockwise")
```

## Approximation outputs

FRsutils reports the following public outputs:

```text
boundary_region = upper_approximation - lower_approximation
positive_region = lower_approximation
```

Public output arrays are NumPy arrays.

## Backend status

ITFRS supports:

- dense NumPy execution,
- exact blockwise NumPy execution,
- optional CuPy-backed similarity blocks,
- experimental GPU-resident blockwise approximation accumulators, with final
  public output converted back to NumPy arrays.
