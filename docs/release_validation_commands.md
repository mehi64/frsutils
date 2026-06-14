# Release validation commands

This document lists the command groups used to validate the public API and core
fuzzy-rough approximation paths before release or JOSS submission.


## Documentation smoke validation

Run the documented quickstart and public API example smoke test after changing
README, `docs/public_api.md`, backend documentation, or examples.

```bash
python examples/public_api_quickstart.py
```

```bash
python -m pytest tests/api/test_public_api_examples_smoke.py -q -rs
```

```bash
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

For the full documentation smoke checklist, see
`docs/documentation_smoke_check.md`.

## Public API validation

```bash
python -m pytest tests/api -q -rs
```

## Core approximation engine validation

```bash
python -m pytest tests/core_tests/test_approximation_engines.py -q -rs
```

## Fast model validation

```bash
python -m pytest \
  tests/models_tests/test_itfrs_fast.py \
  tests/models_tests/test_vqrs_fast.py \
  tests/models_tests/test_owafrs_fast.py \
  -q -rs
```

## Model-specific full validation

ITFRS, including slow tests:

```bash
python -m pytest \
  tests/models_tests/test_itfrs.py \
  tests/models_tests/test_itfrs_fast.py \
  tests/api/test_itfrs_execution_contract.py \
  tests/api/test_itfrs_blockwise_phase2_contract.py \
  tests/api/test_itfrs_gpu_resident_phase3_contract.py \
  tests/api/test_itfrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_phase6_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```

VQRS, including slow tests:

```bash
python -m pytest \
  tests/models_tests/test_vqrs.py \
  tests/models_tests/test_vqrs_fast.py \
  tests/api/test_vqrs_execution_contract.py \
  tests/api/test_vqrs_blockwise_phase4_contract.py \
  tests/api/test_vqrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_phase6_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```

OWAFRS, including slow tests:

```bash
python -m pytest \
  tests/models_tests/test_owafrs.py \
  tests/models_tests/test_owafrs_fast.py \
  tests/api/test_owafrs_execution_contract.py \
  tests/api/test_owafrs_blockwise_phase5_contract.py \
  tests/api/test_owafrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_phase6_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```

## Full repository validation

```bash
python -m pytest -o addopts="" -q -rs
```

This command includes tests marked as `slow` and may take substantially longer
than the public API smoke set.
