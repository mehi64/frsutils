# Release and JOSS validation

This page merges the previous release checklist, release validation commands,
documentation smoke checks, JOSS metadata notes, submit-readiness report, paper
claim boundaries, and release-hardening notes into one active maintainer file.

Use it before tagging a release candidate or submitting FRsutils for JOSS
review.

## Release candidate identity

- Package name: `frsutils`
- Current package version in the inspected metadata: `0.0.5`
- License: `BSD-3-Clause`
- Public API boundary: `frsutils`
- JOSS paper files: `paper.md`, `paper.bib`

Update this section if package metadata changes.

## Safe project and paper claims

FRsutils is a scientific Python library for reusable fuzzy-rough set
computations. It provides backend-aware public APIs for similarity construction,
fuzzy-rough lower and upper approximations, boundary regions, and positive-region
scores. The documented model aliases are `itfrs`, `vqrs`, and `owafrs`.

Safe short claim:

> FRsutils provides dense and exact blockwise fuzzy-rough approximation APIs for
> ITFRS, VQRS, and OWAFRS, with a stable NumPy output contract and optional CuPy
> support in explicit blockwise execution paths.

Safe backend claim:

> FRsutils provides dense NumPy and exact blockwise fuzzy-rough approximation
> APIs with optional CuPy-accelerated similarity blocks and experimental
> CuPy-resident ITFRS/VQRS blockwise approximation accumulators. Public outputs
> remain NumPy arrays, and OWAFRS remains on the conservative exact blockwise
> NumPy row-buffer path in the current release.

Compact JOSS-facing paragraph:

> FRsutils is a Python library for fuzzy-rough set computations, including
> similarity construction, lower and upper approximations, boundary regions, and
> positive-region scores. It provides a compact public API through `frsutils`,
> supports ITFRS, VQRS, and OWAFRS model aliases, and offers dense as well as
> exact blockwise execution. Optional CuPy-backed blockwise execution is
> available for selected internal steps while preserving NumPy arrays as the
> public output contract.

Do not claim:

- full GPU-native execution,
- GPU-resident OWAFRS aggregation,
- guaranteed CuPy speedup,
- FRSMOTE as part of the stable FRsutils core API unless the project boundary is
  intentionally changed.

For backend details, see [Backends](../user/backends.md).

## Public API release checks

- [ ] The canonical import path is documented as `frsutils`.
- [ ] User-facing examples avoid importing from internal `frsutils.core` modules.
- [ ] `frsutils.__all__` exposes the intended public objects only.
- [ ] Top-level `frsutils` does not accidentally expose internal implementation
      modules.
- [ ] `examples/public_api_quickstart.py` runs successfully.
- [ ] Downstream-style code can use only `frsutils` without importing internals.

Recommended command:

```bash
python -m pytest tests/api -q -rs
```

## Core model release checks

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

## Backend and CuPy checks

- [ ] NumPy backend tests pass.
- [ ] Fake-CuPy contract tests pass in normal CI.
- [ ] Real CuPy/CUDA tests skip cleanly when CuPy is unavailable.
- [ ] Any benchmark-based CuPy claim was generated on a machine with compatible
      CuPy/CUDA installed.
- [ ] No documentation claims full GPU-native execution.
- [ ] OWAFRS documentation does not claim GPU-resident approximation
      accumulators.

Recommended commands:

```bash
python -m pytest tests/api/test_backend_aware_components_contract.py -q -rs
python -m pytest tests/api/test_cupy_backend_contract.py -q -rs
python -m pytest tests/api/test_itfrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_vqrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_owafrs_blockwise_cupy_contract.py -q -rs
```

## Documentation smoke checks

Install the documentation dependencies and validate the published MkDocs site:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

Use `mkdocs serve` for a local browser preview. The GitHub Pages workflow builds
all documentation pull requests and deploys `main` after Pages is configured to
use **GitHub Actions** as its publishing source.

Run these after changing README, public API docs, backend docs, examples, or the
JOSS paper:

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
python -m pytest tests/api/test_public_api_examples_smoke.py tests/examples/test_examples_contract.py -q -rs
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

Manual documentation checks:

- [ ] `README.md` contains a working quickstart using `frsutils`.
- [ ] `docs/user/public_api.md` matches the current public API.
- [ ] `docs/user/backends.md` uses conservative backend wording.
- [ ] `paper.md` uses the same model/backend claims as the docs.
- [ ] User-facing documentation has no temporary milestone-based file names.
- [ ] README and docs relative links resolve to existing files.

Documentation link sanity check:

```bash
python - <<'CHECK_LINKS'
from pathlib import Path
import re
files = [Path("README.md"), Path("paper.md"), *Path("docs").rglob("*.md")]
missing = []
for file_path in files:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
        target = match.group(1).split("#", 1)[0]
        if not target or target.startswith("#") or re.match(r"https?://|mailto:", target):
            continue
        if not (file_path.parent / target).resolve().exists():
            missing.append((str(file_path), target))
for item in missing:
    print(item)
raise SystemExit(1 if missing else 0)
CHECK_LINKS
```

## Metadata checks

Minimum checks before release/JOSS submission:

- [ ] `LICENSE` exists and matches SPDX headers.
- [ ] `pyproject.toml` package metadata is current.
- [ ] `CITATION.cff` exists or README citation instructions are current.
- [ ] `paper.md` exists and uses the same model/backend claims as the docs.
- [ ] `paper.bib` contains every citation key used in `paper.md`.
- [ ] Package version is consistent across metadata and citation examples.
- [ ] Documentation links from README and paper point to existing files.
- [ ] Test evidence is recorded for the release candidate.

Citation-key sanity check:

```bash
python - <<'CHECK_CITATIONS'
from pathlib import Path
import re
paper = Path("paper.md").read_text(encoding="utf-8")
bib = Path("paper.bib").read_text(encoding="utf-8")
used = set(re.findall(r"@([A-Za-z0-9:_-]+)", paper))
defined = set(re.findall(r"@\w+\{([^,]+),", bib))
missing = sorted(used - defined)
print("used:", sorted(used))
print("missing:", missing)
raise SystemExit(1 if missing else 0)
CHECK_CITATIONS
```

## Final validation commands

Focused release smoke:

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
```

Full repository validation, including slow tests:

```bash
python -m pytest -o addopts="" -ra --tb=short --durations=50
```

For a machine-readable full-suite report:

```bash
mkdir -p test_reports
python -m pytest -o addopts="" -ra --tb=short --durations=50 \
  --junitxml=test_reports/pytest_full_slow_final.xml
```

Generated outputs under `test_reports/` and `benchmark_smoke_output/` should be
kept out of commits.

## Open-source repository checks

- [ ] GitHub Issues are enabled in repository settings.
- [ ] The issue forms and pull request template render correctly.
- [ ] `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, and
      `CHANGELOG.md` are linked from the README.
- [ ] The CI workflow passes for Python 3.10, 3.11, and 3.12.
- [ ] The package job builds and validates both wheel and source distribution.
- [ ] GitHub Pages uses **GitHub Actions** as its publishing source.
- [ ] The documentation workflow deploys the current `main` branch successfully.

## Repository hygiene before release

- [ ] Run `git status --short`.
- [ ] Stage only intended source, tests, examples, docs, metadata, and paper
      files.
- [ ] Do not stage generated reports, benchmark outputs, cache directories, IDE
      files, local environments, or local logs.
- [ ] Confirm no temporary milestone file names remain in user-facing docs or
      examples.
- [ ] Confirm all changed Python files keep the short BSD-3-Clause SPDX header
      style.

Files that should not be committed:

- `test_reports/`
- `benchmark_smoke_output/`
- `.pytest_cache/`
- `__pycache__/`
- `htmlcov/`
- `.coverage`
- local virtual environments or IDE caches

## Software DOI and citation strategy

The final JOSS article DOI is assigned only after acceptance, so do not cite a
future JOSS DOI in another paper before acceptance. Use a software release DOI
for FRsutils instead.

Recommended path:

- [ ] Create a release tag for FRsutils, for example `v0.1.0` or the version
      chosen for submission.
- [ ] Archive that release with Zenodo or another research-software archive.
- [ ] Use the software archive DOI in downstream papers that need to cite
      FRsutils before JOSS acceptance.
- [ ] If the JOSS submission is already open, optionally mention that FRsutils
      has been submitted to JOSS and is under open review.
- [ ] After JOSS acceptance, update downstream citations if timing allows.

## JOSS submission steps

- [ ] Confirm the repository is public or will be public for review.
- [ ] Confirm `LICENSE`, `CITATION.cff`, `pyproject.toml`, `README.md`,
      `paper.md`, and `paper.bib` are included.
- [ ] Confirm the software archive DOI/version are available if requested during
      JOSS review.
- [ ] Submit the JOSS paper through the JOSS submission process.
- [ ] After the JOSS review issue is opened, respond to reviewer/editor comments
      in the issue.
- [ ] At acceptance time, follow editor instructions for final release, archive
      DOI, and version metadata.

## Submit-ready result record

Fill this section after the final local run.

- Final validation date:
- Python version:
- Operating system:
- CuPy/CUDA environment:
- Full test result:
- Expected skips:
- JOSS paper citation check:
- Documentation link check:
- Release tag:
- Software archive DOI:
- JOSS submission/review URL:
- Final JOSS DOI after acceptance:

## Submit-ready criteria

The repository is submit-ready when all of the following are true:

- [ ] Full test suite passes, including slow tests.
- [ ] Example smoke commands pass.
- [ ] `paper.md` and `paper.bib` exist and citation keys are complete.
- [ ] README and docs links resolve.
- [ ] License metadata is consistent across `LICENSE`, `pyproject.toml`,
      `CITATION.cff`, README, and source SPDX headers.
- [ ] User-facing files do not contain temporary milestone file names.
- [ ] Optional CuPy/GPU support is described conservatively.
- [ ] Repository has no generated reports or local caches staged for commit.
