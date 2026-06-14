# Release validation commands

This document lists the command groups used to validate the public API and core
fuzzy-rough approximation paths before release or JOSS submission.


## Documentation smoke validation

Run the documented quickstart and public API example smoke test after changing
README, `docs/public_api.md`, backend documentation, or examples.

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
```

```bash
python -m pytest tests/api/test_public_api_examples_smoke.py tests/examples/test_examples_contract.py -q -rs
```

```bash
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

For the full documentation smoke checklist, see
`docs/documentation_smoke_check.md`.


## Documentation link sanity check

After moving or renaming documentation files, check README and docs relative
links manually or with a Markdown link checker. The release-critical links are
listed in `docs/joss_metadata_check.md` and `docs/documentation_smoke_check.md`.


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
  tests/api/test_itfrs_blockwise_contract.py \
  tests/api/test_itfrs_gpu_resident_contract.py \
  tests/api/test_itfrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```

VQRS, including slow tests:

```bash
python -m pytest \
  tests/models_tests/test_vqrs.py \
  tests/models_tests/test_vqrs_fast.py \
  tests/api/test_vqrs_execution_contract.py \
  tests/api/test_vqrs_blockwise_contract.py \
  tests/api/test_vqrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```

OWAFRS, including slow tests:

```bash
python -m pytest \
  tests/models_tests/test_owafrs.py \
  tests/models_tests/test_owafrs_fast.py \
  tests/api/test_owafrs_execution_contract.py \
  tests/api/test_owafrs_blockwise_contract.py \
  tests/api/test_owafrs_blockwise_cupy_contract.py \
  tests/api/test_public_api_approximations.py \
  tests/api/test_cupy_backend_contract.py \
  tests/core_tests/test_approximation_engines.py \
  -o addopts="" -q -rs
```


## Final submit-readiness validation

Use `docs/submit_readiness_report.md` to record the final release-candidate
validation evidence. The minimum final sequence is:

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
python -m pytest -o addopts="" -ra --tb=short --durations=50
```

For a machine-readable full-suite report, run:

```bash
mkdir -p test_reports
python -m pytest -o addopts="" -ra --tb=short --durations=50 \
  --junitxml=test_reports/pytest_full_slow_final.xml
```

Generated outputs under `test_reports/` and `benchmark_smoke_output/` should be
kept out of commits.

## Full repository validation

```bash
python -m pytest -o addopts="" -q -rs
```

This command includes tests marked as `slow` and may take substantially longer
than the public API smoke set.
