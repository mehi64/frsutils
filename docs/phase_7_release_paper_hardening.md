# Phase 7 - Release and Paper Hardening

Phase 7 turns the backend/blockwise implementation work into release-ready and
paper-ready project material. It does not add a new runtime execution path.
Instead, it freezes examples, claims, benchmark usage, and release checks.

## Added artifacts

- `examples/phase7_public_api_quickstart.py`
- `examples/phase7_benchmark_smoke.py`
- `docs/paper_claims.md`
- `docs/release_checklist.md`
- `tests/examples/test_phase7_examples_contract.py`

## Purpose

The earlier phases introduced the public API, execution metadata, backend-aware
components, blockwise execution, GPU-resident ITFRS/VQRS accumulators, the
OWAFRS non-GPU-resident decision, and the benchmark suite. Phase 7 makes those
features presentable without overstating them.

## Final recommended claim

Use this release/paper claim:

```text
FRsutils provides dense and exact blockwise fuzzy-rough approximation APIs with
optional CuPy-accelerated similarity blocks and experimental CuPy-resident
ITFRS/VQRS blockwise approximation accumulators. Public outputs remain NumPy
arrays, and OWAFRS remains on the conservative exact blockwise NumPy row-buffer
path in the current release.
```

## What Phase 7 intentionally does not do

Phase 7 does not:

- add GPU-resident OWAFRS,
- add full GPU-native public outputs,
- move FRSMOTE back into FRsutils,
- generate final paper benchmark numbers,
- change the runtime API contract.

## Validation

The Phase 7 examples are covered by lightweight tests in:

```text
tests/examples/test_phase7_examples_contract.py
```

The intended release smoke command is:

```bash
python -m pytest tests/api tests/benchmarks tests/examples -q
```
