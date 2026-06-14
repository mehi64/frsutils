# Phase 3 - GPU-resident ITFRS blockwise accumulator

Phase 3 implements the first approximation path that can keep fuzzy-rough work on
the selected array backend beyond similarity-block construction.

## Scope

Implemented for:

- `model="itfrs"`
- `engine="blockwise"`
- `backend="cupy"`

When these conditions hold, the ITFRS blockwise path now keeps the following on
CuPy until the final public result conversion:

```text
similarity block
→ lower implicator values
→ upper T-norm values
→ row-wise min/max reductions
→ lower/upper accumulators
→ final conversion to NumPy public arrays
```

The public result arrays remain NumPy arrays. This preserves compatibility with
scikit-learn, imbalanced-learn, `frsampling`, and existing user code.

## Public metadata

`FuzzyRoughApproximationResult` now includes:

```python
result.used_gpu_approximation_accumulators
```

For the Phase 3 path, this flag is `True` only for blockwise ITFRS with a CuPy
backend. `used_gpu_similarity_blocks` remains a separate flag and can be `True`
for other models because their similarity blocks may be computed through CuPy
while their approximation accumulators remain NumPy-resident.

## Explicit non-scope

Phase 3 did not make VQRS or OWAFRS approximation accumulators GPU-resident.
VQRS is handled by Phase 4. OWAFRS still uses the existing conservative path:

```text
CuPy similarity block computation
→ conversion to NumPy block
→ NumPy approximation accumulator
```

OWAFRS should remain separate because exact OWA execution requires row-wise
sorting and has different memory/performance tradeoffs.

## Engine boundary

`BlockwiseSimilarityEngine.iter_blocks()` remains the public NumPy block iterator.
A new backend-aware boundary, `iter_backend_blocks()`, allows internal approximation
engines to consume CuPy-resident block values without changing the public iterator
contract.

## Tests

The new contract test is:

```text
tests/api/test_itfrs_gpu_resident_phase3_contract.py
```

It uses a NumPy-backed fake CuPy module so normal CPU-only CI can verify the
public metadata and execution boundary. Real CuPy/CUDA numerical equivalence is
still covered by the optional CuPy tests when CUDA is available.
