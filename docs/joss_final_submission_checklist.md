# FRsutils final JOSS submission checklist

Use this checklist for the final human steps after the repository files in this
release candidate have been copied into the working branch. It records the tasks
that require author judgment, external services, credentials, or local hardware.

## 1. Author and paper checks

- [ ] Confirm the author name in `paper.md` and `CITATION.cff` is correct.
- [ ] Confirm the `paper.md` affiliation is correct. Current placeholder:
      `Independent researcher`.
- [ ] Confirm the `AI usage disclosure` accurately describes how generative AI
      was used for software, documentation, and paper preparation.
- [ ] Confirm the paper claims match the package scope: FRsutils is the
      fuzzy-rough core library; FRSMOTE belongs to the downstream `frsampling`
      package.
- [ ] Confirm the CuPy/GPU wording is conservative: ITFRS and VQRS may use
      GPU-backed similarity blocks and experimental GPU-resident approximation
      accumulators; OWAFRS claims GPU-backed similarity blocks only.
- [ ] Build or preview the JOSS paper with the JOSS toolchain if available.

## 2. Local validation commands

Run these commands from the repository root after copying all final files.

```bash
python examples/public_api_quickstart.py
python examples/benchmark_smoke.py --output-dir benchmark_smoke_output
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
python -m pytest -o addopts="" -ra --tb=short --durations=50
```

Optional machine-readable final report:

```bash
mkdir -p test_reports
python -m pytest -o addopts="" -ra --tb=short --durations=50 \
  --junitxml=test_reports/pytest_full_slow_final.xml
```

Do not commit `test_reports/` or `benchmark_smoke_output/`.

## 3. Paper and documentation sanity checks

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

## 4. Repository hygiene before release

- [ ] Run `git status --short`.
- [ ] Stage only source, tests, examples, docs, metadata, and paper files that
      are intended for the release candidate.
- [ ] Do not stage generated reports, benchmark outputs, cache directories,
      IDE files, local environments, or local logs.
- [ ] Confirm no temporary milestone file names remain in user-facing docs or
      examples.
- [ ] Confirm all Python files changed for this release keep the short
      BSD-3-Clause SPDX header style.

## 5. Software DOI and citation strategy for the FRSMOTE paper

The final JOSS article DOI is assigned only after acceptance, so do not cite a
future JOSS DOI in the FRSMOTE paper before acceptance. Use a software release
DOI for FRsutils instead.

Recommended path:

- [ ] Create a release tag for FRsutils, for example `v0.1.0` or the version you
      decide to submit.
- [ ] Archive that release with Zenodo or another research-software archive.
- [ ] Use the software archive DOI in the FRSMOTE paper citation.
- [ ] If the JOSS submission is already open, optionally add a note in the
      FRSMOTE paper such as: `FRsutils has been submitted to JOSS and is under
      open review.`
- [ ] After JOSS acceptance, update the FRSMOTE paper citation if timing allows
      so it cites the published JOSS DOI as well as, or instead of, the software
      archive DOI.

Important options:

- Zenodo GitHub integration normally creates a DOI after a GitHub release.
- Zenodo manual upload can be used when you need to reserve a DOI before final
  publication of the upload.
- A JOSS review issue URL can document open review status, but it is not a
  substitute for the final JOSS article DOI.

## 6. JOSS submission steps

- [ ] Confirm the repository is public or will be public for review.
- [ ] Confirm `LICENSE`, `CITATION.cff`, `pyproject.toml`, `README.md`,
      `paper.md`, and `paper.bib` are included.
- [ ] Confirm the software archive DOI/version are available if requested during
      JOSS review.
- [ ] Submit the JOSS paper through the JOSS submission process.
- [ ] After the JOSS review issue is opened, respond to reviewer/editor comments
      in the issue.
- [ ] At acceptance time, follow the editor instructions for final release,
      archive DOI, and version metadata.

## 7. Final record to fill manually

- Final validation date:
- Python version:
- Operating system:
- Full test result:
- Expected skips:
- Release tag:
- Software archive DOI:
- JOSS submission/review URL:
- Final JOSS DOI after acceptance:
