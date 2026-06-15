# JOSS-facing metadata check

This checklist keeps the repository metadata aligned with the public API and
JOSS-facing documentation.

## Package metadata

Check `pyproject.toml`:

- [ ] Package name is correct.
- [ ] Version is current.
- [ ] Python version requirement is current.
- [ ] Runtime dependencies include only required packages.
- [ ] Optional dependencies are documented as optional.
- [ ] Project URLs point to the correct repository and documentation pages.

## License consistency

Check `LICENSE`, source headers, and package metadata:

- [ ] `LICENSE` exists at repository root.
- [ ] `pyproject.toml` license field matches the chosen project license.
- [ ] Python file headers use the same SPDX identifier as the project license.
- [ ] README license section matches the root `LICENSE` file.

Do not mix license identifiers in release files. If the project license changes,
update all SPDX headers and documentation together.

## Citation metadata

Check citation files and README citation text:

- [ ] `CITATION.cff` exists or the README clearly explains how to cite the
      package.
- [ ] Citation title is correct and consistent with README/package metadata.
- [ ] Author metadata is correct.
- [ ] Version matches `pyproject.toml` for a tagged release.
- [ ] Repository URL is correct.
- [ ] DOI is added after archival, if applicable.
- [ ] JOSS DOI is added after acceptance, if applicable.


## Current metadata alignment status

For the current release-candidate metadata, the project uses `BSD-3-Clause`
consistently across the root `LICENSE` file, `pyproject.toml`, `CITATION.cff`,
README license text, and Python SPDX headers. Keep this section current if the
license or package version changes before submission.

## JOSS paper metadata

Check `paper.md` and bibliography files:

- [x] `paper.md` exists at the repository root.
- [ ] Author affiliations are correct.
- [x] The statement of need focuses on fuzzy-rough set utilities.
- [x] The paper describes `frsutils` as the canonical user-facing API.
- [x] The paper does not describe internal modules as stable public API.
- [x] Backend/CuPy claims match `docs/paper_claims.md`.
- [x] OWAFRS is not described as having GPU-resident approximation accumulators.
- [x] Oversampling algorithms are described as downstream usage, not as part of
      the frsutils core package, unless the project scope changes.

## Documentation links

Check links from README and docs:

- [ ] `docs/public_api.md`
- [ ] `docs/cupy_info.md`
- [ ] `docs/backend_execution_status.md`
- [ ] `docs/paper_claims.md`
- [ ] `docs/release_checklist.md`
- [ ] `examples/public_api_quickstart.py`

## Test evidence

Before submission, record the commands and outcome for the release candidate. Record the final outcome in `docs/submit_readiness_report.md`.

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
python -m pytest -o addopts="" -q -rs
```

If real CuPy/CUDA tests are skipped, note that optional GPU tests were skipped
because CuPy/CUDA was unavailable in the tested environment.


## Paper draft status

The initial JOSS paper draft now exists as `paper.md`, with references in
`paper.bib`. Before submission, manually confirm the author affiliation and any
funding acknowledgement. If a DOI archive is created before review, add it to the
README, `CITATION.cff`, and the JOSS submission form.
