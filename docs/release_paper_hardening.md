# Release and paper hardening

This release-hardening work turns the backend/blockwise implementation work into release-ready and
paper-ready project material. It does not add a new runtime execution path.
Instead, it freezes examples, claims, benchmark usage, and release checks.

## Added artifacts

- `paper.md`
- `paper.bib`
- `examples/public_api_quickstart.py`
- `examples/benchmark_smoke.py`
- `docs/paper_claims.md`
- `docs/release_checklist.md`
- `tests/examples/test_examples_contract.py`

## Purpose

The project now has a public API, execution metadata, backend-aware components,
blockwise execution, experimental GPU-resident ITFRS/VQRS accumulators, the
OWAFRS non-GPU-resident decision, and a benchmark suite. This document keeps
those features presentable without overstating them.

## Final recommended claim

Use this release/paper claim:

```text
frsutils provides dense and exact blockwise fuzzy-rough approximation APIs with
optional CuPy-accelerated similarity blocks and experimental CuPy-resident
ITFRS/VQRS blockwise approximation accumulators. Public outputs remain NumPy
arrays, and OWAFRS remains on the conservative exact blockwise NumPy row-buffer
path in the current release.
```

## What this release hardening intentionally does not do

This release hardening does not:

- add GPU-resident OWAFRS,
- add full GPU-native public outputs,
- move FRSMOTE back into frsutils,
- generate final paper benchmark numbers,
- change the runtime API contract.

## Validation

The release examples are covered by lightweight tests in:

```text
tests/examples/test_examples_contract.py
```

The intended release smoke command is:

```bash
python -m pytest tests/api tests/benchmarks tests/examples -q
```


## JOSS paper draft boundary

The JOSS paper draft is intentionally conservative. It describes frsutils as the
fuzzy-rough core library, keeps oversampling algorithms in the downstream
`frsampling` boundary, and mirrors the model-specific CuPy wording in
`docs/paper_claims.md`. Before submission, confirm the author affiliation and
acknowledgements.
