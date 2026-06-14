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
- [ ] Citation title is `FRsutils`.
- [ ] Author metadata is correct.
- [ ] Version matches `pyproject.toml` for a tagged release.
- [ ] Repository URL is correct.
- [ ] DOI is added after archival, if applicable.
- [ ] JOSS DOI is added after acceptance, if applicable.

## JOSS paper metadata

Check `paper.md` and bibliography files:

- [ ] `paper.md` exists at the expected JOSS path.
- [ ] Author affiliations are correct.
- [ ] The statement of need focuses on fuzzy-rough set utilities.
- [ ] The paper describes `FRsutils.api` as the canonical user-facing API.
- [ ] The paper does not describe internal modules as stable public API.
- [ ] Backend/CuPy claims match `docs/paper_claims.md`.
- [ ] OWAFRS is not described as having GPU-resident approximation accumulators.
- [ ] Oversampling algorithms are described as downstream usage, not as part of
      the FRsutils core package, unless the project scope changes.

## Documentation links

Check links from README and docs:

- [ ] `docs/public_api.md`
- [ ] `docs/cupy_info.md`
- [ ] `docs/backend_execution_status.md`
- [ ] `docs/paper_claims.md`
- [ ] `docs/release_checklist.md`
- [ ] `examples/public_api_quickstart.py`

## Test evidence

Before submission, record the commands and outcome for:

```bash
python -m pytest tests/api tests/core_tests/test_approximation_engines.py -q -rs
python -m pytest -o addopts="" -q -rs
```

If real CuPy/CUDA tests are skipped, note that optional GPU tests were skipped
because CuPy/CUDA was unavailable in the tested environment.
