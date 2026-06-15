# frsutils release and JOSS readiness checklist

Use this checklist before tagging a release or submitting frsutils for JOSS
review.

## 1. Public API checks

- [ ] The canonical import path is documented as `frsutils`.
- [ ] User-facing examples avoid importing from internal `frsutils.core` modules.
- [ ] `frsutils.__all__` exposes the intended public objects only.
- [ ] Top-level `frsutils` exposes the intended public objects only and does
      not accidentally expose internal implementation modules.
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
python -m pytest tests/api/test_cupy_backend_contract.py -q -rs
python -m pytest tests/api/test_itfrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_vqrs_blockwise_cupy_contract.py -q -rs
python -m pytest tests/api/test_owafrs_blockwise_cupy_contract.py -q -rs
```

## 4. Documentation checks

- [ ] `README.md` contains a working quickstart using `frsutils`.
- [ ] `docs/public_api.md` matches the current public API.
- [ ] `examples/public_api_quickstart.py` runs successfully.
- [ ] `tests/api/test_public_api_examples_smoke.py` passes.
- [ ] `docs/documentation_smoke_check.md` has been followed after public API or
      documentation changes.
- [ ] `docs/cupy_info.md` and `docs/backend_execution_status.md` use conservative
      backend wording.
- [ ] `docs/paper_claims.md` reflects the claims made in the JOSS paper.
- [ ] README and docs relative links resolve to existing files.
- [ ] User-facing documentation has no temporary milestone-based file names.
- [ ] No documentation claims full GPU-native execution.
- [ ] OWAFRS documentation does not claim GPU-resident approximation
      accumulators.

Recommended documentation smoke commands:

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
python -m pytest tests/api/test_public_api_examples_smoke.py tests/examples/test_examples_contract.py -q -rs
python -m pytest tests/api/test_public_api_downstream_contract.py -q -rs
```

## 5. Metadata checks

See `docs/joss_metadata_check.md` for a detailed metadata checklist. At minimum:

- [ ] `LICENSE` exists and matches SPDX headers.
- [ ] `pyproject.toml` package metadata is current.
- [ ] `CITATION.cff` exists or the README citation instructions are current.
- [ ] `paper.md` exists and uses the same model/backend claims as the docs.
- [ ] The package version is consistent across metadata and citation examples.


## 6. JOSS paper checks

- [ ] `paper.md` builds with the JOSS template.
- [ ] `paper.bib` contains every citation key used in `paper.md`.
- [ ] Author affiliation is correct.
- [ ] The AI usage disclosure is accurate for the submitted version.
- [ ] The paper does not overstate CuPy/GPU support.
- [ ] The paper describes oversampling algorithms as downstream usage rather than
      part of the frsutils core package.

Recommended local citation-key sanity check:

```bash
python - <<'PY'
from pathlib import Path
import re
paper = Path('paper.md').read_text(encoding='utf-8')
bib = Path('paper.bib').read_text(encoding='utf-8')
used = set(re.findall(r'@([A-Za-z0-9:_-]+)', paper))
defined = set(re.findall(r'@\w+\{([^,]+),', bib))
missing = sorted(used - defined)
print('used:', sorted(used))
print('missing:', missing)
raise SystemExit(1 if missing else 0)
PY
```

## 7. Final validation commands

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


## 8. Submit package checks

Before submitting to JOSS or tagging a release candidate:

- [ ] Complete `docs/submit_readiness_report.md` with the final validation
      environment and test outcome.
- [ ] Confirm `git status --short` contains only intended source, docs, metadata,
      example, and test changes.
- [ ] Confirm no generated reports, local benchmark outputs, caches, or IDE
      files are staged.
- [ ] Confirm `paper.md`, `paper.bib`, `CITATION.cff`, `LICENSE`,
      `pyproject.toml`, and `README.md` are included in the release candidate.
- [ ] Confirm optional CuPy/GPU test skips, if any, are documented as optional
      environment skips rather than failures.

Recommended final status command:

```bash
git status --short
```

## 9. External submission and citation steps

Before submitting or coordinating the FRSMOTE paper citation, follow
`docs/joss_final_submission_checklist.md`. It records the author-only steps that
cannot be completed by automated tests, including affiliation confirmation,
release tagging, software DOI creation, JOSS submission, and the citation plan
for papers that need to reference frsutils before the JOSS article is accepted.
