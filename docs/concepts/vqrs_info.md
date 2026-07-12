# VQRS model

VQRS stands for Vaguely Quantified Rough Sets. In frsutils, VQRS uses fuzzy
quantifiers to compute lower and upper approximation degrees from a fuzzy
similarity relation and class labels.

## Intuition

For each instance, VQRS compares the instance to the rest of the dataset through
the active similarity relation. It then measures how strongly similar instances
support membership in the target class. Fuzzy quantifiers control how strict the
lower and upper approximation interpretations are.

## Implementation contract

`frsutils.core.models.VQRS` is the dense NumPy reference implementation. It
expects a fully materialized similarity matrix, a one-dimensional label vector,
and lower and upper fuzzy quantifiers.

The public approximation API also provides exact blockwise VQRS execution:

```python
from frsutils import compute_approximations

result = compute_approximations(X, y, model="vqrs", engine="blockwise")
```

## Approximation outputs

frsutils reports the following public outputs:

```text
boundary_region = upper_approximation - lower_approximation
positive_region = lower_approximation
```

Public output arrays are NumPy arrays.

## Backend status

VQRS supports:

- dense NumPy execution,
- exact blockwise NumPy execution,
- optional CuPy-backed similarity blocks,
- experimental GPU-resident blockwise approximation accumulators, with final
  public output converted back to NumPy arrays.

For VQRS blockwise execution, public boundary and positive-region arrays are
derived from the returned NumPy lower and upper approximation arrays.
