# CuPy and backend execution

CuPy support in FRsutils is optional and experimental. The stable backend is
NumPy, and the public API always returns NumPy arrays so results remain easy to
use with scientific Python, plotting tools, and scikit-learn-style workflows.

Use CuPy only through explicit blockwise execution:

```python
from FRsutils.api import compute_approximations

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

## Current model support

| Model | Dense execution | Blockwise NumPy | CuPy-backed similarity blocks | GPU-resident approximation accumulators |
| --- | --- | --- | --- | --- |
| ITFRS | Yes | Yes | Yes | Yes, experimental |
| VQRS | Yes | Yes | Yes | Yes, experimental |
| OWAFRS | Yes | Yes | Yes | No |

The OWAFRS distinction is intentional. Exact OWAFRS needs row-wise sorting and
OWA aggregation. In the current release cycle, similarity blocks may be computed
with CuPy, but OWAFRS aggregation remains on the conservative NumPy-compatible
row-buffer path. Do not describe OWAFRS as fully GPU-native.

## Public result contract

Regardless of backend, FRsutils public approximation results expose NumPy arrays:

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

## Recommended wording

Safe wording for documentation, benchmark reports, and the JOSS paper:

> FRsutils provides dense NumPy and exact blockwise fuzzy-rough approximation
> APIs. Optional CuPy-backed blockwise execution is available for similarity
> blocks, with experimental GPU-resident approximation accumulators for ITFRS
> and VQRS. Public outputs remain NumPy arrays. OWAFRS uses GPU-backed
> similarity blocks only and does not currently claim GPU-resident OWA
> aggregation.

## Testing CuPy support

Run the backend contract tests with:

```bash
python -m pytest tests/api/test_cupy_backend_contract.py -q -rs
```

If CuPy or CUDA is not installed, real-CuPy tests should be skipped. Fake-CuPy
contract tests still exercise the public metadata and array-conversion behavior
in regular CI.
