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
3. builds the wheel and source distribution;
4. checks package metadata and required distribution contents;
5. installs both built artifacts in isolated environments and performs import
   and dependency smoke tests.

A pull request should not be merged until all required checks pass. Changes to
`tests/reference_data/`, its loader, accessors, or contract tests additionally
require code-owner approval on the protected default branch.
