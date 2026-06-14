# Documentation smoke check

Use this checklist after public API or documentation changes and before a
release tag or JOSS submission. It verifies that the documented user-facing
workflow still matches executable code.

## Scope

This smoke check covers the documentation that a new user or reviewer is most
likely to read first:

- `README.md`
- `docs/public_api.md`
- `docs/cupy_info.md`
- `docs/backend_execution_status.md`
- `docs/paper_claims.md`
- `examples/public_api_quickstart.py`

The canonical user-facing import path is `FRsutils.api`. Examples and user
documentation should avoid importing from internal `FRsutils.core` modules unless
explicitly explaining implementation details.

## Fast documentation smoke commands

Run these commands from the repository root after applying documentation and
example updates.

```bash
python examples/public_api_quickstart.py
```

```bash
python -m pytest tests/api/test_public_api_examples_smoke.py -q -rs
```

```bash
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
```

## Model/API consistency commands

Run the fast model tests together with the public API tests when documentation
mentions model behavior, backend behavior, or approximation outputs.

```bash
python -m pytest \
  tests/models_tests/test_itfrs_fast.py \
  tests/models_tests/test_vqrs_fast.py \
  tests/models_tests/test_owafrs_fast.py \
  tests/api \
  tests/core_tests/test_approximation_engines.py \
  -q -rs
```

## Slow release validation

Run slow tests before release tagging or JOSS submission. The default project
configuration may exclude tests marked as `slow`, so override `addopts`.

```bash
python -m pytest -o addopts="" -q -rs
```

If the full suite is too slow for every iteration, run it at least before the
release tag and before JOSS submission.

## Optional real-CuPy validation

Fake-CuPy tests run in normal CI and verify the public API contract. Real CuPy
validation requires a compatible local CuPy/CUDA installation.

```bash
python -m pytest tests/api/test_cupy_backend_phase6_contract.py -q -rs
```

```bash
python -m pytest tests/api/test_itfrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_vqrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_owafrs_blockwise_cupy_contract.py -q -rs
```

Real-CuPy tests should either pass on a configured GPU machine or skip cleanly
when CuPy/CUDA is unavailable.

## Manual documentation checks

Before publishing, check the following manually:

- README quickstart imports from `FRsutils.api`.
- `docs/public_api.md` matches the current public API names.
- Examples do not rely on internal `FRsutils.core` imports.
- Backend wording does not claim full GPU-native execution.
- ITFRS and VQRS may claim CuPy-backed blockwise approximation accumulators.
- OWAFRS only claims CuPy-backed similarity blocks, not GPU-resident OWAFRS
  approximation accumulators.
- Public outputs are documented as NumPy arrays regardless of backend.
- `docs/paper_claims.md` and `paper.md`, if present, use the same model/backend
  claims.

## Expected outcome

After this smoke check passes, documentation, examples, public API tests, and
core approximation engine tests should tell the same story:

- `FRsutils.api` is the stable public facade.
- Dense NumPy execution is the stable baseline.
- Blockwise execution is exact and available through the public API.
- CuPy support is optional and model-specific.
- Public results remain NumPy-compatible.
