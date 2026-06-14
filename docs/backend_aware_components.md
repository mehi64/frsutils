# Backend-aware components

This design moves backend-specific mathematical formulas into the core fuzzy-rough
components instead of keeping mirrored formulas inside `similarity_engine.py`.

## Scope

Implemented component-level backend hooks:

- `Similarity.compute_backend(diff, xp=...)`
- `TNorm.compute_backend(a, b, xp=...)`
- `TNorm.reduce_backend(arr, xp=...)`
- `Implicator.compute_backend(a, b, xp=..., validate_inputs=...)`
- `FuzzyQuantifier.compute_backend(x, xp=..., validate_inputs=...)`

The public/default behavior remains NumPy-compatible. Existing calls such as
`similarity(x, y)`, `tnorm(a, b)`, `implicator(a, b)`, and `quantifier(x)` still
return NumPy-compatible results.

## Why backend-aware components matter

Earlier implementations kept duplicated formulas in `similarity_engine.py` for
similarities and T-norms so CuPy similarity blocks could be computed. That was
useful for bootstrapping, but it created a maintenance risk: changing a formula
inside `similarities.py` or `tnorms.py` could silently diverge from the blockwise
engine implementation.

The engine now delegates formula execution back to the component:

```python
feature_sim = similarity_func.compute_backend(diff, xp=backend.xp)
sim_block = tnorm.compute_backend(sim_block, feature_sim, xp=backend.xp)
```

This keeps mathematical ownership in the component classes and makes later GPU
work safer.

## Current execution boundary

Backend-aware components support exact blockwise execution behind the public API.
ITFRS and VQRS may keep CuPy-backed similarity blocks and experimental
approximation accumulators on CuPy until final NumPy public output conversion.
OWAFRS may use CuPy-backed similarity blocks, but exact row-wise OWA sorting and
aggregation remain on the conservative NumPy row-buffer path.

## Supported backend-aware formulas

Similarities:

- `linear`
- `gaussian`

T-norms:

- `minimum`
- `product`
- `lukasiewicz`
- `drastic`
- `einstein`
- `hamacher`
- `nilpotent`
- `yager`

Implicators:

- `lukasiewicz`
- `goedel`
- `kleenedienes`
- `reichenbach`
- `goguen`
- `rescher`
- `yager`
- `weber`
- `fodor`

Fuzzy quantifiers:

- `linear`
- `quadratic`

## Tests

The new contract test is:

```text
tests/api/test_backend_aware_components_contract.py
```

It verifies that backend-aware formulas match the existing NumPy public behavior.
