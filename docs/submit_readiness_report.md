# Submit readiness report

Use this file to record the final validation evidence before submitting FRsutils
for JOSS review or tagging a release candidate.

## Release candidate identity

- Package name: `FRsutils`
- Package version: `0.0.3`
- License: `BSD-3-Clause`
- Public API boundary: `FRsutils.api`
- JOSS paper files: `paper.md`, `paper.bib`

## Required final commands

Run the commands from the repository root. Record the date, environment, and
outcome in the release notes or pull request before submission.

### 1. Package import smoke

```bash
python - <<'PY'
import FRsutils
import FRsutils.api as api
print("FRsutils import: OK")
print("public API objects:", sorted(api.__all__))
PY
```

### 2. Example smoke

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
```

### 3. Public API and core smoke

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
```

### 4. Full repository validation including slow tests

```bash
python -m pytest -o addopts="" -ra --tb=short --durations=50
```

For an archived machine-readable report:

```bash
mkdir -p test_reports
python -m pytest -o addopts="" -ra --tb=short --durations=50 \
  --junitxml=test_reports/pytest_full_slow_final.xml
```

### 5. JOSS paper citation-key sanity check

```bash
python - <<'PY'
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
PY
```

### 6. Documentation link sanity check

```bash
python - <<'PY'
from pathlib import Path
import re
files = [Path("README.md"), Path("paper.md"), *Path("docs").glob("*.md")]
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
PY
```

## Result record

Fill this section after the final local run.

- Final validation date:
- Python version:
- Operating system:
- CuPy/CUDA environment:
- Full test result:
- Expected skips:
- JOSS paper citation check:
- Documentation link check:

## Files that should not be committed

- `test_reports/`
- `benchmark_smoke_output/`
- `.pytest_cache/`
- `__pycache__/`
- `htmlcov/`
- `.coverage`
- local virtual environments or IDE caches

## Submit-ready criteria

The repository is submit-ready when all of the following are true. After these
checks pass, complete the external author-only checklist in
`docs/joss_final_submission_checklist.md`.

- [ ] Full test suite passes, including slow tests.
- [ ] Example smoke commands pass.
- [ ] `paper.md` and `paper.bib` exist and citation keys are complete.
- [ ] README and docs links resolve.
- [ ] License metadata is consistent across `LICENSE`, `pyproject.toml`,
      `CITATION.cff`, README, and source SPDX headers.
- [ ] User-facing files do not contain temporary milestone file names.
- [ ] Optional CuPy/GPU support is described conservatively.
- [ ] The repository has no generated reports or local caches staged for commit.
