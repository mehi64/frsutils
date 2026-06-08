# Backend-Aware Components — Phase 1

Phase 1 moves backend-specific mathematical formulas into the core fuzzy-rough
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

## Why this phase matters

Before this phase, `similarity_engine.py` contained duplicated formulas for
similarities and T-norms so CuPy similarity blocks could be computed. That was
useful for bootstrapping, but it created a maintenance risk: changing a formula
inside `similarities.py` or `tnorms.py` could silently diverge from the blockwise
engine implementation.

After this phase, the engine delegates formula execution back to the component:

```python
feature_sim = similarity_func.compute_backend(diff, xp=backend.xp)
sim_block = tnorm.compute_backend(sim_block, feature_sim, xp=backend.xp)
```

This keeps mathematical ownership in the component classes and makes later GPU
work safer.

## Current execution boundary

This phase itself did **not** make the full approximation path GPU-resident. The
next backend phase builds on these hooks. As of Phase 3, blockwise ITFRS can keep
its CuPy similarity blocks, implicator/T-norm values, and min/max accumulators
resident on CuPy until final NumPy public output conversion. VQRS and OWAFRS
still use the conservative NumPy accumulator path.

## Supported backend-aware formulas in this phase

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
tests/api/test_backend_aware_components_phase1_contract.py
```

It verifies that backend-aware formulas match the existing NumPy public behavior.
