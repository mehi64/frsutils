# Release and JOSS validation

This page defines the reproducible release procedure for `frsutils` 0.1.1. Run
all commands from the repository root and archive command outputs outside the
source tree when they are useful as release evidence.

## Release identity

- Package: `frsutils`
- Target version: `0.1.1`
- License: BSD-3-Clause
- Supported Python versions: 3.10, 3.11, and 3.12
- Public API boundary: package-root imports from `frsutils`
- JOSS sources: `paper.md` and `paper.bib`

## Safe software claims

`frsutils` provides dense NumPy and exact blockwise fuzzy-rough approximation
APIs for ITFRS, VQRS, and OWAFRS. Public outputs are NumPy arrays. Optional CuPy
execution is model-specific and is available only through documented blockwise
paths.

Do not claim:

- fully GPU-native execution;
- GPU-resident OWAFRS aggregation;
- guaranteed CuPy speedup;
- universal runtime or memory improvements;
- FRSMOTE as part of the stable `frsutils` public API.

See [Backends and execution](../user/backends.md) for the precise boundaries.

## 1. Start from a clean repository

```bash
git switch main
git pull --ff-only
git status --short
```

The final command must produce no output. Generated files, caches, local logs,
IDE settings, and virtual environments must not be committed.

## 2. Install the release environment

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

## 3. Validate metadata and documentation

```bash
cffconvert --validate
python scripts/validate_joss_submission.py
mkdocs build --strict
```

Inspect the generated site locally with `mkdocs serve`. Do not commit `site/`.

Verify that version `0.1.1` is consistent in:

- `pyproject.toml`;
- `CITATION.cff`;
- `README.md`;
- `CHANGELOG.md`;
- `RELEASE_NOTES_v0.1.1.md`;
- the release and archive documentation.

## 4. Run the fast and default tests

```bash
python -m pytest tests/reference_data_tests -ra
python -m pytest tests -ra
python examples/public_api_quickstart.py
python benchmarks/benchmark_smoke.py --output-dir benchmark_smoke_output
```

Remove `benchmark_smoke_output/` after inspection.

## 5. Run the exhaustive slow tests

The recommended four-shard execution is:

```bash
for i in 0 1 2 3; do
    python scripts/run_slow_test_shard.py \
        --shard-index "$i" \
        --shard-count 4 || break
done
```

Alternatively, run the complete suite in one process:

```bash
python -m pytest -o addopts="" -m slow -ra --durations=50
```

Retain any JUnit or terminal reports outside the repository.

## 6. Validate the public API and coverage

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

## 7. Regenerate the reference study from the release commit

The provenance record must identify the actual commit that will be released.
Therefore, first commit all source, documentation, and metadata changes:

```bash
git add --all
git commit -m "Prepare frsutils 0.1.1 for JOSS submission"
git status --short
```

The worktree must be clean before regeneration:

```bash
python studies/fuzzy_rough_reference_study/run_study.py
python -m pytest tests/studies -ra
```

Confirm in `studies/fuzzy_rough_reference_study/results/environment.json`:

- `frsutils_version` is `0.1.1`;
- `git_commit` is the release-preparation commit;
- `git_worktree_dirty` is `false`.

Also confirm that every row in `dense_blockwise_equivalence.csv` passes and that
`benchmark_results.json` contains no failed cases. Commit the regenerated result
snapshot, then rerun `git status --short`.

## 8. Optional real-CUDA validation

Real-CUDA validation is required only when the release makes real-GPU numerical
claims:

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

Keep the generated CUDA report as external release evidence unless it is
intentionally archived with the release.

## 9. Build and validate distributions

```bash
rm -rf build dist frsutils.egg-info
python -m build
python -m twine check dist/*
```

Install and exercise both artifacts in clean environments:

```bash
python -m venv /tmp/frsutils-wheel-smoke
/tmp/frsutils-wheel-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-wheel-smoke/bin/python -m pip install dist/frsutils-0.1.1-*.whl
/tmp/frsutils-wheel-smoke/bin/python -m pip check

python -m venv /tmp/frsutils-sdist-smoke
/tmp/frsutils-sdist-smoke/bin/python -m pip install --upgrade pip
/tmp/frsutils-sdist-smoke/bin/python -m pip install dist/frsutils-0.1.1.tar.gz
/tmp/frsutils-sdist-smoke/bin/python -m pip check
```

GitHub Actions performs stronger read-only installed-package validation for both
artifacts.

## 10. Final repository and paper checks

Before tagging, confirm:

- the repository is public and Issues are enabled;
- CI, documentation, extended-test, and paper workflows are green;
- README links and MkDocs navigation resolve;
- `paper.md` contains the required JOSS sections and accurate claims;
- the paper date and author metadata are final;
- funding and AI-use disclosures are accurate;
- no cache, generated site, local log, editor setting, or temporary report is
  tracked;
- `git status --short` is empty.

## 11. Tag, release, and archive

```bash
git tag -a v0.1.1 -m "frsutils 0.1.1"
git push origin main
git push origin v0.1.1
```

Create the GitHub release from `RELEASE_NOTES_v0.1.1.md`, archive it through
Zenodo, and follow the [software archive and DOI guide](archive_and_doi.md).
