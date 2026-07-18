# Backends and execution behavior

This page describes the possible backends in frsutils. Terms such as dense execution, blockwise execution, backend, and public output are defined in the [glossary](glossary.md).

For the model-independent interpretation of relation rows, self-comparisons, boundary values, and dense/blockwise equivalence, see [Core computation contracts](computation_contracts.md).

## Execution modes

FRsutils exposes execution modes through the canonical public API,
`frsutils.compute_approximations`.

| Execution mode          | User-facing option                    | Intended use                                                                    |
| ----------------------- | ------------------------------------- | ------------------------------------------------------------------------------- |
| Dense NumPy             | `engine="dense"`                      | Reference behavior and small datasets.                                          |
| Exact blockwise NumPy   | `engine="blockwise", backend="numpy"` | In-memory datasets where the full similarity matrix should not be materialized. |
| Optional CuPy blockwise | `engine="blockwise", backend="cupy"`  | GPU-backed blockwise execution for supported paths.                             |

The public output type is stable across all modes: approximation arrays are
returned as NumPy arrays.

Dense approximation execution accepts `backend="numpy"` or `backend="auto"`
and resolves both to NumPy. A direct `backend="cupy"` request with
`engine="dense"` is rejected; CuPy is available only through blockwise
approximation execution. Unknown backend aliases are rejected in both modes.

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

### Host/device transfer boundary

For a CuPy-backed blockwise similarity engine, the validated feature matrix is
copied to the device lazily on the first block iteration and then reused for all
row and column slices. Similarity blocks therefore do not repeatedly transfer
the same feature rows from host memory. ITFRS and VQRS keep their supported
reductions and accumulators on the device until the final public arrays are
converted to NumPy. OWAFRS converts each completed similarity block to NumPy
because its exact sorting and OWA aggregation remain CPU-resident.

Low-level empty similarity blocks follow the same boundary: a backend-resident
empty block is returned when backend residency is requested, and conversion to
NumPy occurs only when the caller requests a public NumPy block.

## Installation

Install the optional CUDA 12 environment with:

```bash
python -m pip install "frsutils[gpu-cuda12x]"
```

For local development:

```bash
python -m pip install -e ".[gpu-cuda12x]"
```

The CUDA extra uses CuPy 14.x and includes CuPy's CUDA component dependencies.
The core package does not install CuPy or CUDA dependencies.

## Archiveable CUDA validation artifact

A release that mentions real CUDA execution should include a machine-readable
validation artifact generated on the release GPU:

```bash
python scripts/capture_cuda_validation.py \
  --require-cuda \
  --output-json cuda_validation_report.json
```

The script first proves that CuPy can discover a device, allocate arrays, execute
a kernel, synchronize, and return a value to NumPy. It then compares dense
NumPy against blockwise CuPy for two configurations of each public model and
multiple block sizes. The JSON records:

- operating system, Python, NumPy, FRsutils, and CuPy versions;
- CUDA runtime and driver versions;
- GPU name, compute capability, memory, and device count;
- captured `nvidia-smi` and `nvcc --version` outputs;
- maximum absolute differences for lower, upper, boundary, and positive-region
  arrays;
- public output dtypes and backend-residency metadata;
- explicit claim boundaries, including the absence of GPU-resident OWAFRS
  approximation accumulation.

A report with `status="unavailable"` or skipped real-CUDA tests is useful for
CPU-only diagnostics, but it is not evidence that GPU numerical execution was
validated. Archive only a `status="success"` report as release evidence.

## Validated CUDA environment

The real-CUDA numerical tests for the 0.1.0 release candidate were executed in
the following environment:

| Component                       | Validated value                       |
| ------------------------------- | ------------------------------------- |
| Operating system                | Ubuntu 24.04.4 LTS                    |
| Linux kernel                    | 6.8.0-111-generic                     |
| GPU                             | NVIDIA GeForce GTX 1050 Mobile, 4 GiB |
| GPU compute capability          | 6.1                                   |
| NVIDIA driver                   | 535.309.01                            |
| Driver-reported CUDA capability | CUDA 12.2                             |
| System CUDA Toolkit             | CUDA 12.0                             |
| Python                          | 3.11.6                                |
| NumPy                           | 2.3.5                                 |
| CuPy                            | 14.1.1                                |
| CUDA runtime header wheel       | `nvidia-cuda-runtime-cu12` 12.0.146   |
| CUDA path discovery             | `cuda-pathfinder` 1.5.6               |

This table records one successfully tested configuration. It is not an
exhaustive compatibility matrix, and these exact versions are not mandatory for
all installations.

The stable CPU backend remains NumPy. GPU performance depends on the model,
dataset size, block size, available GPU memory, driver, and CUDA environment.
FRsutils does not guarantee that CuPy execution is faster for every workload.

## CUDA installation troubleshooting

Inspect the detected GPU and CuPy environment first:

```bash
nvidia-smi
python -c "import cupy as cp; cp.show_config()"
```

If CuPy detects the GPU but the first numerical operation fails with:

```text
RuntimeError: Failed to find CUDA headers
```

prefer reinstalling the FRsutils CUDA extra, which includes CuPy's CUDA
component dependencies:

```bash
python -m pip install --upgrade "frsutils[gpu-cuda12x]"
```

For an existing plain `cupy-cuda12x` installation, install CUDA runtime headers
matching the local CUDA 12 toolkit minor version. For example, a CUDA 12.0
environment can use:

```bash
python -m pip install "nvidia-cuda-runtime-cu12==12.0.*"
```

After installation, `cupy.show_config()` should report a non-empty
`CUDA Extra Include Dirs` entry. Then verify a real CUDA operation:

```bash
python -c "import cupy as cp; x = cp.arange(10, dtype=cp.float32); print(cp.asnumpy(x * x))"
```

## Model-specific backend status

| Model  | Dense NumPy | Exact blockwise NumPy | CuPy-backed similarity blocks | GPU-resident approximation accumulators |
| ------ | ----------- | --------------------- | ----------------------------- | --------------------------------------- |
| ITFRS  | Yes         | Yes                   | Yes                           | Yes, experimental                       |
| VQRS   | Yes         | Yes                   | Yes                           | Yes, experimental                       |
| OWAFRS | Yes         | Yes                   | Yes                           | No                                      |

## Dense/blockwise parity validation

The release test suite separates formula coverage from cross-layer execution
coverage:

- direct component and model tests verify every registered formula;
- the direct OWAFRS blockwise matrix checks all 648 canonical combinations of
  upper T-norm, lower implicator, upper OWA method, and lower OWA method at four
  block sizes;
- the public core execution matrix performs 1,360 deterministic NumPy
  dense/blockwise comparisons across ITFRS, VQRS, and OWAFRS while exercising
  every canonical similarity and similarity T-norm.

These tests compare `lower`, `upper`, `boundary`, and `positive_region` with
strict numerical tolerances. They establish exact NumPy dense/blockwise parity
for the tested canonical configuration space. CuPy correctness is validated by
separate fake-backend contract tests and real-CUDA tests because a CPU-only run
cannot prove CUDA execution.

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

This is intentional for the current release cycle.

> OWAFRS supports dense NumPy and exact blockwise execution. With
> `backend="cupy"`, similarity blocks can be computed using the CuPy backend,
> but OWAFRS aggregation is not currently GPU-resident. Public outputs remain
> NumPy arrays. OWAFRS therefore uses CuPy only for similarity-block generation
> and does not guarantee a speedup.

## Backend-aware components

Backend-specific mathematical formulas live in the core fuzzy-rough
components. Implemented component-level backend hooks include:

- `Similarity.compute_backend(diff, xp=...)`
- `TNorm.compute_backend(a, b, xp=...)`
- `TNorm.reduce_backend(arr, xp=...)`
- `Implicator.compute_backend(a, b, xp=..., validate_inputs=...)`
- `FuzzyQuantifier.compute_backend(x, xp=..., validate_inputs=...)`

Existing ordinary calls such as `similarity(x, y)`, `tnorm(a, b)`,
`implicator(a, b)`, and `quantifier(x)` remain NumPy-compatible.

The engine delegates formula execution back to the component:

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

#
