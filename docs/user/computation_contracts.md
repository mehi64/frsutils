# Core computation contracts

This page collects the interpretation rules shared by the dense, exact
blockwise, and optional CuPy-backed fuzzy-rough execution paths. These contracts
apply to ITFRS, VQRS, and OWAFRS unless a model-specific section states
otherwise.

## Relation orientation

A dense or precomputed relation uses the orientation:

```text
similarity_matrix[i, j] = R(x_i, x_j)
```

Row `i` is the query row whose approximation value is returned for sample
`x_i`. Column `j` contributes evidence from sample `x_j`. Symmetric similarity
matrices are unaffected by this distinction; transposing an asymmetric relation
can change the result.

The core model attribute is:

```python
model.relation_orientation == "rows_are_queries"
```

## Self-comparisons

The current model implementations exclude diagonal self-comparisons from the
reported sample scores:

- ITFRS sets diagonal lower evidence to the neutral minimum value `1` and
  diagonal upper evidence to the neutral maximum value `0`.
- VQRS excludes the diagonal from both supporting mass and total similarity
  mass.
- OWAFRS removes one diagonal evidence value from each sorted row before OWA
  aggregation.

Dense and blockwise implementations use the same rule. This is a fixed model
contract in the current release, not a public configuration parameter.

## VQRS defaults

The default VQRS lower and upper quantifiers are intentionally different:

```text
lower: quadratic Q(alpha=0.2, beta=1.0)
upper: quadratic Q(alpha=0.0, beta=0.6)
```

The defaults represent a stricter lower quantifier and a more permissive upper
quantifier. Flat and nested configuration can replace either quantifier
explicitly.

## Boundary and positive-region outputs

FRsutils reports:

```text
signed_boundary = upper - lower
positive_region = lower
```

The boundary is not clipped. Custom component combinations can therefore
produce negative signed-boundary values when they do not guarantee
`lower <= upper`.

The public aliases `result.boundary`, `boundary_region()`, and
`compute_boundary_region()` are retained for backward compatibility and expose
the same signed difference. `signed_boundary` and `compute_signed_boundary()`
are the explicit names for new code.

The positive-region output retains the current lower-score contract. Its name
and meaning are unchanged in this release.

## Dense, blockwise, and CuPy equivalence

For the same model, data, and component configuration, exact blockwise NumPy
execution is expected to agree numerically with the dense NumPy reference path.
Optional CuPy-backed blockwise execution is tested against the same dense
reference within floating-point tolerance.

Backend residency differs by model:

| Model | CuPy similarity blocks | CuPy approximation accumulators |
| --- | --- | --- |
| ITFRS | supported | supported |
| VQRS | supported | supported |
| OWAFRS | supported | not claimed; OWA sorting and aggregation remain NumPy-resident |

All public result arrays are converted to NumPy arrays.

## Input validity

Feature matrices and precomputed relations must be finite numeric arrays. Public
and direct core paths reject `NaN`, positive infinity, and negative infinity.
Precomputed relations must be square, labels must be one-dimensional and
aligned, and fuzzy-rough approximation models require at least two samples.
