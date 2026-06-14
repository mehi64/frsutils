# Backend execution status

This page summarizes the execution paths currently supported by FRsutils through
the canonical public API, `FRsutils.api`.

## Execution modes

| Execution mode | User-facing option | Intended use |
| --- | --- | --- |
| Dense NumPy | `engine="dense"` | Reference behavior and small datasets |
| Exact blockwise NumPy | `engine="blockwise", backend="numpy"` | Larger datasets without materializing a full similarity matrix |
| Optional CuPy blockwise | `engine="blockwise", backend="cupy"` | Experimental GPU-backed blockwise execution |

The public output type is stable across all modes: approximation arrays are
returned as NumPy arrays.

## Model-specific backend status

| Model | Dense reference | Exact blockwise | CuPy similarity blocks | CuPy approximation accumulators |
| --- | --- | --- | --- | --- |
| ITFRS | NumPy | Yes | Yes | Yes, experimental |
| VQRS | NumPy | Yes | Yes | Yes, experimental |
| OWAFRS | NumPy | Yes | Yes | No |

## Notes for users

- Use `FRsutils.api.compute_approximations` as the primary public entry point.
- Use `engine="dense"` when you want the simplest reference path.
- Use `engine="blockwise"` when full similarity-matrix materialization is too
  expensive or when testing optional backend behavior.
- Treat `backend="cupy"` as optional and experimental. It should not be required
  for ordinary installation or for reproducing baseline results.

## Notes for maintainers

The direct model classes in `FRsutils.core.models` are dense NumPy reference
implementations. Backend-aware execution belongs in the similarity and
approximation engines behind the public API. This separation keeps the scientific
reference implementations simple while allowing scalable execution paths to
advance independently.
