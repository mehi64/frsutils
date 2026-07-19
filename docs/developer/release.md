# Maintainer release process

This page defines the repeatable release process for `frsutils`. It is intended
for maintainers and is version-independent. Run commands from the repository
root, replace `<version>` with the version being published, and keep generated
reports outside the source tree unless they are intentional project artifacts.

## Release contract

Every release must preserve these public commitments:

- the canonical package and import name is `frsutils`;
- the license is BSD-3-Clause;
- the supported public API is exposed from the package root;
- public approximation outputs are NumPy arrays;
- ITFRS, VQRS, and OWAFRS dense and blockwise computations follow the
  documented contracts;
- optional CuPy behavior remains model-specific and limited to documented
  execution paths.

The precise backend and GPU claim boundaries are documented in
[Backends and execution](../user/backends.md).

## 1. Select and record the release version

Choose the next semantic version and update the version-specific metadata in:

- `pyproject.toml`;
- `CITATION.cff`;
- `CHANGELOG.md`;
- the release-notes file for that version;
- user-facing examples or citations that explicitly show a package version.

The release date in `CITATION.cff` and the changelog must match the actual
release date.

## 2. Start from a clean repository

```bash
git switch main
git pull --ff-only
git status --short
```

The final command must produce no output. Generated documentation, build
artifacts, caches, local logs, editor settings, virtual environments, and test
reports must not be tracked.

## 3. Create the validation environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs,study]"
python -m pip install build twine cffconvert
python -m pip check
```

On Windows, activate `.venv\Scripts\Activate.ps1` and use the same Python
commands.

## 4. Validate metadata and documentation

```bash
cffconvert --validate
mkdocs build --strict
```

Inspect the generated documentation with `mkdocs serve`. The generated `site/`
directory is disposable and must not be committed.

## 5. Run the default validation suite

```bash
python -m pytest tests/reference_data_tests -ra
python -m pytest tests -ra
python examples/public_api_quickstart.py
python benchmarks/benchmark_smoke.py --output-dir benchmark_smoke_output
```

Remove `benchmark_smoke_output/` after inspection.

## 6. Run exhaustive slow tests

The recommended four-shard execution is:

```bash
for i in 0 1 2 3; do
    python scripts/run_slow_test_shard.py \
        --shard-index "$i" \
        --shard-count 4 || break
done
```

The complete slow suite can also be run in one process:

```bash
python -m pytest -o addopts="" -m slow -ra --durations=50
```

## 7. Validate public API and coverage contracts

```bash
mkdir -p test_reports
python scripts/validate_installed_public_api.py \
  --output-json test_reports/source_public_api_validation.json

python -m pytest tests/api \
  --cov=frsutils/api \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=95 \
  -ra

python -m pytest tests/core_tests tests/models_tests tests/api \
  --cov=frsutils/core \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=95 \
  -ra
```

Do not commit `test_reports/`, `.coverage`, or `htmlcov/`.

## 8. Refresh the reference-study provenance

The committed reference-study snapshot must identify the code that produced it.
Commit all source, documentation, and metadata changes, verify a clean worktree,
and regenerate the study:

```bash
git add --all
git commit -m "Release frsutils <version>"
git status --short
python studies/fuzzy_rough_reference_study/run_study.py
python -m pytest tests/studies -ra
```

Confirm in
`studies/fuzzy_rough_reference_study/results/environment.json` that:

- `frsutils_version` matches the selected release version;
- `git_commit` identifies the committed source state;
- `git_worktree_dirty` is `false`.

Every row in `dense_blockwise_equivalence.csv` must pass, and
`benchmark_results.json` must contain no failed case. Commit regenerated study
artifacts when they differ from the repository snapshot.

## 9. Run optional real-CUDA validation

Run real-CUDA validation only when an appropriate NVIDIA/CUDA environment is
available and the release includes GPU-related changes:

```bash
python scripts/capture_cuda_validation.py \
  --require-cuda \
  --output-json cuda_validation_report.json

python -m pytest \
  tests/api/test_cupy_backend_contract.py \
  tests/api/test_itfrs_blockwise_cupy_contract.py \
  tests/api/test_vqrs_blockwise_cupy_contract.py \
  tests/api/test_owafrs_blockwise_cupy_contract.py \
  tests/api/test_real_cupy_parity_matrix.py \
  -o addopts="" -vv -rs
```

Store the generated CUDA report outside the repository unless it is deliberately
published as a release artifact.

## 10. Build and validate distributions

```bash
rm -rf build dist frsutils.egg-info
python -m build
python -m twine check dist/*
```

Install and exercise both artifacts in clean environments:

```bash
python -m venv /tmp/frsutils-wheel-smoke
/tmp/frsutils-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-wheel-smoke/bin/python -m pip install dist/frsutils-<version>-*.whl
/tmp/frsutils-wheel-smoke/bin/python -m pip check

python -m venv /tmp/frsutils-sdist-smoke
/tmp/frsutils-sdist-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-sdist-smoke/bin/python -m pip install dist/frsutils-<version>.tar.gz
/tmp/frsutils-sdist-smoke/bin/python -m pip check
```

GitHub Actions performs additional read-only installed-package validation for
both artifacts.

## 11. Tag and publish

After all required workflows pass on the release commit:

```bash
git tag -a v<version> -m "frsutils <version>"
git push origin main
git push origin v<version>
```

Create the GitHub release from the corresponding release-notes file. Publish the
wheel and source distribution to PyPI only after checking that `dist/` contains
artifacts for the selected version and no older files.

## 12. Verify the published package

Install the published version without using the local cache:

```bash
python -m venv /tmp/frsutils-published-smoke
/tmp/frsutils-published-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-published-smoke/bin/python -m pip install --no-cache-dir \
  "frsutils==<version>"
/tmp/frsutils-published-smoke/bin/python -m pip check
/tmp/frsutils-published-smoke/bin/python -c \
  "import frsutils; from importlib.metadata import version; print(version('frsutils'))"
```

Record an immutable software archive and DOI according to the
[software archiving guide](archive_and_doi.md) when a citable archive is desired.
