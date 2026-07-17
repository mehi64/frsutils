# Test procedures

FRsutils uses `pytest` for scientific, behavioral, regression, backend, and
repository-level tests. The commands below are intended to be run from the
repository root.

## Install the development environment

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip check
```

CuPy tests are optional and are skipped when a compatible CuPy installation is
not available. For CUDA 12.x environments, install the project GPU extra:

```bash
python -m pip install -e ".[dev,gpu-cuda12x]"
```

## Default test suite

The default configuration in `pytest.ini` excludes tests marked `slow`:

```bash
python -m pytest -ra
```

This is the suite used for routine development and the Python-version matrix in
continuous integration.

## Public API branch-coverage gate

The stable public API has a branch-aware coverage floor. Run the same command as
continuous integration with:

```bash
python -m pytest tests/api \
  --cov=frsutils.api \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=95 \
  -ra
```

The gate is intentionally below 100 percent: it prevents material regressions
without encouraging tests that merely execute defensive lines and do not verify
scientific or public behavior.

## Core branch-coverage gate

The mathematical core has a separate branch-aware coverage floor. Run the same
command as continuous integration with:

```bash
python -m pytest tests/core_tests tests/models_tests tests/api \
  --cov=frsutils.core \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=95 \
  -ra
```

The core gate combines direct component/model tests with public integration
tests. This prevents coverage from depending only on one test layer while still
requiring user-visible numerical behavior to be checked.

## Scientific reference-data contract

Run the integrity and schema checks whenever a JSON reference value, manifest
entry, loader, or accessor changes:

```bash
python -m pytest tests/reference_data_tests -ra
```

These tests verify that every JSON file is registered in the manifest, its
SHA-256 digest matches, strict JSON parsing succeeds, encoded NumPy arrays keep
their declared dtype and shape, and loaded arrays are read-only. Also run the
operator or model test module that consumes the changed dataset.

## Slow and exhaustive tests

Run only tests marked `slow`:

```bash
python -m pytest -o addopts="" -m slow -ra
```

Run the complete suite, including both routine and slow tests:

```bash
python -m pytest -o addopts="" -ra
```

The `-o addopts=""` override is required because `pytest.ini` excludes slow
tests by default.

For exhaustive suites that become slower in one long process, run the same
slow-test collection as deterministic contiguous shards. Shard indexes are
zero-based:

```bash
python scripts/run_slow_test_shard.py --shard-index 0 --shard-count 4
python scripts/run_slow_test_shard.py --shard-index 1 --shard-count 4
python scripts/run_slow_test_shard.py --shard-index 2 --shard-count 4
python scripts/run_slow_test_shard.py --shard-index 3 --shard-count 4
```

Every collected slow-test node ID belongs to exactly one shard. All shards must
pass before a release.

The slow suite includes three complementary validation layers:

- exhaustive dense OWAFRS reference combinations;
- all canonical OWAFRS component combinations against the direct blockwise
  engine at multiple block sizes;
- a public cross-layer execution matrix covering every canonical similarity and
  similarity T-norm with all ITFRS component pairs, all VQRS quantifier pairs,
  and a balanced OWAFRS component matrix.

The public execution matrix performs 1,360 dense/blockwise result comparisons:
1,152 for ITFRS, 64 for VQRS, and 144 for OWAFRS. CuPy residency and transfer
contracts remain separate because real CUDA tests are environment-dependent.

## CuPy backend contracts

CPU-only development runs exercise strict fake-CuPy residency and transfer
contracts as part of the default suite. On a machine with a working CUDA device,
install the GPU extra and run the real-CUDA model contracts explicitly:

```bash
python -m pytest \
  tests/api/test_cupy_backend_contract.py \
  tests/api/test_itfrs_blockwise_cupy_contract.py \
  tests/api/test_vqrs_blockwise_cupy_contract.py \
  tests/api/test_owafrs_blockwise_cupy_contract.py \
  -o addopts="" -vv -rs
```

These tests verify real device allocation, backend-resident similarity blocks,
dense/blockwise numerical parity, public NumPy outputs, output dtypes, and the
model-specific accumulator claims. A skipped real-CUDA test is not evidence of
GPU correctness; record at least one complete run on the release environment.

## Targeted test modules

Examples:

```bash
python -m pytest tests/core_tests/test_tnorms.py -ra
python -m pytest tests/models_tests/test_itfrs.py -ra
python -m pytest tests/models_tests/test_vqrs.py -ra
python -m pytest tests/models_tests/test_owafrs.py -ra
```

Use targeted runs while developing, but run the default suite before opening a
pull request.

## Build and validate distributions

Install the release tools and build from a clean output directory:

```bash
python -m pip install build twine
rm -rf build dist frsutils.egg-info
python -m build
python -m twine check dist/*
```

The wheel contains the installable `frsutils` package. The source distribution
also contains tests, documentation, examples, studies, and JSON scientific
reference data. Inspect the source distribution when changing packaging rules:

```bash
tar -tzf dist/frsutils-*.tar.gz | grep "tests/reference_data"
```

Test installation from the built wheel in an isolated environment:

```bash
python -m venv /tmp/frsutils-wheel-smoke
/tmp/frsutils-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-wheel-smoke/bin/python -m pip install dist/frsutils-*.whl
cd /tmp
/tmp/frsutils-wheel-smoke/bin/python -c "import frsutils; print(frsutils.__file__)"
/tmp/frsutils-wheel-smoke/bin/python -m pip check
```

Repeat the same procedure with the source distribution before a release:

```bash
python -m venv /tmp/frsutils-sdist-smoke
/tmp/frsutils-sdist-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-sdist-smoke/bin/python -m pip install dist/frsutils-*.tar.gz
cd /tmp
/tmp/frsutils-sdist-smoke/bin/python -c "import frsutils; print(frsutils.__file__)"
/tmp/frsutils-sdist-smoke/bin/python -m pip check
```

On Windows, replace `bin/python` with `Scripts/python.exe`.

## Continuous integration

`.github/workflows/ci.yml` performs the following checks:

1. installs the editable package and development dependencies on supported
   Python versions;
2. runs the reference-data contract and default test suites;
3. enforces at least 95 percent branch-aware coverage separately for
   `frsutils.api` and `frsutils.core` on Python 3.12;
4. builds the wheel and source distribution;
5. checks package metadata and required distribution contents;
6. installs both built artifacts in isolated environments and performs import
   and dependency smoke tests.

`.github/workflows/extended-tests.yml` executes the exhaustive slow tests as
four deterministic contiguous shards in separate jobs. It can be started
manually before a release and also runs monthly so exhaustive model and
execution-matrix combinations do not silently regress. CPU-only CI is expected
to skip real-CuPy
tests cleanly; real CUDA validation is recorded separately according to
`docs/developer/release.md`.

A pull request should not be merged until all required checks pass. Changes to
`tests/reference_data/`, its loader, accessors, or contract tests additionally
require code-owner approval on the protected default branch.
