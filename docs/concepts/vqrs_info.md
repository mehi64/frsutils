# VQRS model

VQRS stands for Vaguely Quantified Rough Sets. In frsutils, VQRS uses fuzzy
quantifiers to compute lower and upper approximation degrees from a fuzzy
similarity relation and class labels.

The cross-model rules for relation orientation, self-comparison removal, signed boundaries, and backend parity are summarized in [Core computation contracts](../user/computation_contracts.md).

## Intuition

For each instance, VQRS compares the instance to the rest of the dataset through
the active similarity relation. It then measures how strongly similar instances
support membership in the target class. Fuzzy quantifiers control how strict the
lower and upper approximation interpretations are.

The public defaults use a stricter quadratic lower quantifier
`Q(0.2, 1.0)` and a more permissive quadratic upper quantifier
`Q(0.0, 0.6)`. They represent `most` and `some` interpretations,
respectively, and keep the default lower and upper approximations distinct.

## Implementation contract

`frsutils.core.models.VQRS` is the dense NumPy reference implementation. It
expects at least two samples, a fully materialized finite fuzzy-relation matrix
with values in `[0, 1]`, a one-dimensional label vector, and lower and upper
fuzzy quantifiers. The low-level relation matrix may be asymmetric and need not
have a unit diagonal; VQRS removes the actual diagonal values from its
non-self evidence sums. FRsutils uses `relation[i, j] = R(x_i, x_j)`: row `i`
supplies the supporting and total evidence used for sample `x_i`. Transposing
an asymmetric relation can therefore change the quantified ratio.

The public approximation API also provides exact blockwise VQRS execution:

```python
from frsutils import compute_approximations

result = compute_approximations(X, y, model="vqrs", engine="blockwise")
```

## Approximation outputs

frsutils reports the following public outputs:

```text
signed_boundary = upper_approximation - lower_approximation
positive_region = lower_approximation
```

`boundary_region` remains available as a backward-compatible name for the same
signed difference. FRsutils does not clip this value, so custom component
choices can produce negative values when they do not guarantee
`lower_approximation <= upper_approximation`. Public output arrays are NumPy
arrays.

## Backend status

VQRS supports:

- dense NumPy execution,
- exact blockwise NumPy execution,
- optional CuPy-backed similarity blocks,
- experimental GPU-resident blockwise approximation accumulators, with final
  public output converted back to NumPy arrays.

For VQRS blockwise execution, public boundary and positive-region arrays are
derived from the returned NumPy lower and upper approximation arrays.
