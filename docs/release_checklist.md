# FRsutils release and JOSS readiness checklist

Use this checklist before tagging a release or submitting FRsutils for JOSS
review.

## 1. Public API checks

- [ ] The canonical import path is documented as `FRsutils.api`.
- [ ] User-facing examples avoid importing from internal `FRsutils.core` modules.
- [ ] `FRsutils.api.__all__` exposes the intended public objects only.
- [ ] Top-level `FRsutils` remains compact and does not accidentally expose the
      full public facade.
- [ ] `examples/public_api_quickstart.py` runs successfully.

Recommended command:

```bash
python -m pytest tests/api -q -rs
```

## 2. Core model checks

Run the focused fast and public-contract tests for all fuzzy-rough models:

```bash
python -m pytest \
  tests/models_tests/test_itfrs_fast.py \
  tests/models_tests/test_vqrs_fast.py \
  tests/models_tests/test_owafrs_fast.py \
  tests/api \
  tests/core_tests/test_approximation_engines.py \
  -q -rs
```

Run slow exhaustive model-combination tests before a release tag:

```bash
python -m pytest tests/models_tests -m slow -o addopts="" -q -rs
```

## 3. Backend and CuPy checks

- [ ] NumPy backend tests pass.
- [ ] Fake-CuPy contract tests pass in normal CI.
- [ ] Real CuPy/CUDA tests skip cleanly when CuPy is unavailable.
- [ ] If claiming benchmark results for CuPy, run the tests on a machine with
      compatible CuPy/CUDA installed.

Recommended commands:

```bash
python -m pytest tests/api/test_cupy_backend_phase6_contract.py -q -rs
python -m pytest tests/api/test_itfrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_vqrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_owafrs_blockwise_cupy_contract.py -q -rs
```

## 4. Documentation checks

- [ ] `README.md` contains a working quickstart using `FRsutils.api`.
- [ ] `docs/public_api.md` matches the current public API.
- [ ] `examples/public_api_quickstart.py` runs successfully.
- [ ] `tests/api/test_public_api_examples_smoke.py` passes.
- [ ] `docs/documentation_smoke_check.md` has been followed after public API or
      documentation changes.
- [ ] `docs/cupy_info.md` and `docs/backend_execution_status.md` use conservative
      backend wording.
- [ ] `docs/paper_claims.md` reflects the claims made in the JOSS paper.
- [ ] No documentation claims full GPU-native execution.
- [ ] OWAFRS documentation does not claim GPU-resident approximation
      accumulators.

Recommended documentation smoke commands:

```bash
python examples/public_api_quickstart.py
python -m pytest tests/api/test_public_api_examples_smoke.py -q -rs
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

## 5. Metadata checks

See `docs/joss_metadata_check.md` for a detailed metadata checklist. At minimum:

- [ ] `LICENSE` exists and matches SPDX headers.
- [ ] `pyproject.toml` package metadata is current.
- [ ] `CITATION.cff` exists or the README citation instructions are current.
- [ ] `paper.md` exists and uses the same model/backend claims as the docs.
- [ ] The package version is consistent across metadata and citation examples.

## 6. Final validation commands

Focused release smoke:

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
```

Full repository validation, including slow tests:

```bash
python -m pytest -o addopts="" -q -rs
```

If the full test suite is too slow for every local iteration, run it before
release tagging and before JOSS submission.
