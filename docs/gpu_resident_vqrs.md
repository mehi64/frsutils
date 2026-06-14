# GPU-resident VQRS blockwise accumulator

This path extends the backend-resident approximation boundary from ITFRS to VQRS.
It keeps the public API unchanged while allowing the VQRS blockwise accumulator
path to stay on the selected backend when `backend="cupy"` is requested.

## Scope

Implemented for:

- `model="vqrs"`
- `engine="blockwise"`
- `backend="cupy"`

When these conditions hold, the VQRS blockwise path keeps the following on CuPy
until final public result conversion:

```text
similarity block
→ same-label mask
→ minimum T-norm values
→ numerator accumulator
→ denominator accumulator
→ interim VQRS ratio
→ lower/upper fuzzy-quantifier values
→ final conversion to NumPy public arrays
```

The full `n x n` similarity matrix is still not materialized for approximation
computation. Public result arrays remain NumPy arrays so downstream packages,
scikit-learn, imbalanced-learn, and `frsampling` keep their existing contracts.

## Public metadata

The existing metadata flag is reused:

```python
result.used_gpu_approximation_accumulators
```

For this path, this flag is `True` for blockwise VQRS with a CuPy backend. It is
also `True` for the blockwise ITFRS CuPy path. It remains `False` for
OWAFRS because exact OWAFRS still uses a conservative NumPy row-buffer path.

## Explicit non-scope

This VQRS path does not make OWAFRS approximation accumulators GPU-resident. OWAFRS
requires row-wise sorting and OWA-weighted aggregation, which has different
memory/performance tradeoffs from ITFRS min/max reductions and VQRS sums.

Current OWAFRS flow with `backend="cupy"` remains:

```text
CuPy similarity-block computation
→ conversion to NumPy block
→ NumPy lower/upper row buffers
→ NumPy row-wise sort and OWA aggregation
```

## Tests

The main contract tests are:

```text
tests/api/test_vqrs_blockwise_contract.py
tests/api/test_itfrs_gpu_resident_contract.py
```

The VQRS GPU-resident contract uses a NumPy-backed fake CuPy module so CPU-only
CI can verify the execution metadata and dense/blockwise numerical equivalence.
Optional real CuPy/CUDA tests still cover the installed-CuPy path when available.


## OWAFRS follow-up decision

The current release finalizes the non-scope decision for OWAFRS GPU-resident accumulators.
OWAFRS remains on the conservative NumPy row-buffer path for the current
release/paper cycle. Future GPU-resident OWAFRS work would require a separate
benchmark/spike focused on row-wise sorting, OWA-weighted aggregation, memory
pressure, and block-size behavior.
