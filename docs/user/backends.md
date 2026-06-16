# Backends and execution behavior

This page merges the previous CuPy, backend-status, backend-aware-component, and
GPU decision notes into one active backend document. Terms such as dense execution, blockwise execution, backend,
and public output are defined in the [glossary](glossary.md).

## Execution modes

FRsutils exposes execution modes through the canonical public API,
`frsutils.compute_approximations`.

| Execution mode | User-facing option | Intended use |
| --- | --- | --- |
| Dense NumPy | `engine="dense"` | Reference behavior and small datasets. |
| Exact blockwise NumPy | `engine="blockwise", backend="numpy"` | In-memory datasets where the full similarity matrix should not be materialized. |
| Optional CuPy blockwise | `engine="blockwise", backend="cupy"` | GPU-backed blockwise execution for supported paths. |

The public output type is stable across all modes: approximation arrays are
returned as NumPy arrays.

## CuPy usage

CuPy is optional and currently limited to selected backend-aware computation
paths. The stable backend is NumPy. Use CuPy only through explicit blockwise
execution:

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
    engine="blockwise",
    block_size=512,
    backend="cupy",
)
```

If CuPy or CUDA is not installed, CuPy-specific tests and benchmark cases should
skip cleanly instead of failing ordinary CPU-only validation.

## Model-specific backend status

| Model | Dense NumPy | Exact blockwise NumPy | CuPy-backed similarity blocks | GPU-resident approximation accumulators |
| --- | --- | --- | --- | --- |
| ITFRS | Yes | Yes | Yes | Yes, experimental |
| VQRS | Yes | Yes | Yes | Yes, experimental |
| OWAFRS | Yes | Yes | Yes | No |

## Public result contract

Regardless of backend, public approximation results expose NumPy arrays:

```python
result.lower
result.upper
result.boundary
result.positive_region
```

Execution metadata records what happened internally:

```python
result.engine
result.backend
result.block_size
result.used_blockwise
result.used_gpu_similarity_blocks
result.used_gpu_approximation_accumulators
```

For CuPy-backed blockwise execution, `used_gpu_similarity_blocks` indicates that
similarity blocks used the CuPy backend. For ITFRS and VQRS,
`used_gpu_approximation_accumulators` may also be true. For OWAFRS it should
remain false in the current implementation.

## ITFRS GPU-resident blockwise accumulator

For this path:

- `model="itfrs"`
- `engine="blockwise"`
- `backend="cupy"`

FRsutils can keep the following on CuPy until final public output conversion:

```text
similarity block
→ lower implicator values
→ upper T-norm values
→ row-wise min/max reductions
→ lower/upper accumulators
→ final conversion to NumPy public arrays
```

The public result arrays remain NumPy arrays. This preserves compatibility with
scientific Python, scikit-learn-style workflows, and downstream packages.

## VQRS GPU-resident blockwise accumulator

For this path:

- `model="vqrs"`
- `engine="blockwise"`
- `backend="cupy"`

FRsutils can keep the following on CuPy until final public output conversion:

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
computation.

## OWAFRS GPU boundary

OWAFRS supports dense NumPy and exact blockwise execution. With
`backend="cupy"`, similarity blocks can be computed using CuPy, but exact OWA
sorting and aggregation remain on a NumPy-compatible row-buffer path.

This is intentional for the current JOSS/release cycle. Exact OWAFRS requires
collecting values for each row, sorting them, and applying OWA weights. Making
that path fully GPU-resident would require a separate design and benchmark pass
for row-wise buffering, sorting, memory layout, and numerical equivalence.

Safe OWAFRS claim:

> OWAFRS supports dense NumPy and exact blockwise execution. With
> `backend="cupy"`, similarity blocks can be computed using the CuPy backend,
> but OWAFRS aggregation does not currently claim GPU-resident approximation
> accumulators. Public outputs remain NumPy arrays.

Do not claim:

- full GPU-native OWAFRS,
- GPU-resident OWAFRS OWA aggregation,
- guaranteed speedup for OWAFRS with CuPy.

## Backend-aware components

Backend-specific mathematical formulas should live in the core fuzzy-rough
components, not as mirrored formulas inside the similarity engine.

Implemented component-level backend hooks include:

- `Similarity.compute_backend(diff, xp=...)`
- `TNorm.compute_backend(a, b, xp=...)`
- `TNorm.reduce_backend(arr, xp=...)`
- `Implicator.compute_backend(a, b, xp=..., validate_inputs=...)`
- `FuzzyQuantifier.compute_backend(x, xp=..., validate_inputs=...)`

Existing ordinary calls such as `similarity(x, y)`, `tnorm(a, b)`,
`implicator(a, b)`, and `quantifier(x)` should remain NumPy-compatible.

The engine should delegate formula execution back to the component:

```python
feature_sim = similarity_func.compute_backend(diff, xp=backend.xp)
sim_block = tnorm.compute_backend(sim_block, feature_sim, xp=backend.xp)
```

This keeps mathematical ownership in component classes and reduces the risk of
formula divergence between dense and blockwise execution.

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

## Recommended wording

Safe wording for documentation, benchmark reports, and the JOSS paper:

> FRsutils provides dense NumPy and exact blockwise fuzzy-rough approximation
> APIs. Optional CuPy-backed blockwise execution is available for similarity
> blocks, with experimental GPU-resident approximation accumulators for ITFRS
> and VQRS. Public outputs remain NumPy arrays. OWAFRS uses GPU-backed
> similarity blocks only and does not currently claim GPU-resident OWA
> aggregation.

Avoid these claims unless future tests and benchmarks support them:

- “FRsutils is fully GPU-native.”
- “All fuzzy-rough models run fully on the GPU.”
- “CuPy always improves performance.”
- “OWAFRS aggregation is GPU-resident.”

## Backend tests

Core backend and GPU-boundary tests include:

```bash
python -m pytest tests/api/test_backend_aware_components_contract.py -q -rs
python -m pytest tests/api/test_cupy_backend_contract.py -q -rs
python -m pytest tests/api/test_itfrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_vqrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_owafrs_blockwise_cupy_contract.py -q -rs
```

The fake-CuPy contract tests should run in CPU-only CI. Optional real-CuPy/CUDA
numerical tests should run only when a compatible GPU environment is available.
