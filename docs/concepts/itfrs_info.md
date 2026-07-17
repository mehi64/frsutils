# ITFRS model

ITFRS stands for Implicator/T-norm Fuzzy-Rough Sets. In frsutils, ITFRS computes lower and upper approximation degrees from a fuzzy similarity relation and class
labels using a lower implicator and an upper T-norm.

## Implementation contract

`frsutils.core.models.ITFRS` is the dense NumPy reference implementation. It
expects at least two samples, a fully materialized finite fuzzy-relation matrix
with values in `[0, 1]`, a one-dimensional label vector, an upper T-norm, and a
lower implicator. The low-level relation matrix may be asymmetric and need not
have a unit diagonal because self-comparisons are handled explicitly.

The public approximation API also provides exact blockwise ITFRS execution:

```python
from frsutils import compute_approximations

result = compute_approximations(X, y, model="itfrs", engine="blockwise")
```

## Approximation outputs

frsutils reports the following public outputs:

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
- experimental GPU-resident blockwise approximation accumulators, with final public output converted back to NumPy arrays.
