# frsutils 0.1.0

`frsutils` 0.1.0 is the first JOSS-oriented release of the lowercase Python
package and its stable package-root API.

## Highlights

- ITFRS, VQRS, and OWAFRS through a shared approximation result contract.
- Package-root functions for similarity matrices, lower and upper
  approximations, signed boundary regions, and positive-region scores.
- Dense NumPy reference execution and exact blockwise execution.
- Optional, model-specific CuPy-backed execution paths with conservative claims.
- Reusable registries and builders for similarities, t-norms, implicators,
  fuzzy quantifiers, and OWA weights.
- A reproducible real-dataset reference study with resolved configuration,
  environment metadata, machine-readable outputs, figures, and checksums.
- Continuous integration, package-build validation, documentation deployment,
  and JOSS paper validation workflows.

## Compatibility

- Python 3.10, 3.11, and 3.12 are the supported release targets.
- Core dependencies: NumPy and scikit-learn.
- CuPy is optional and is not installed with the core package.

## Validation

Before publishing the release, run:

```bash
python -m pytest tests -ra
python scripts/validate_joss_submission.py
python -m build
python -m twine check dist/*
mkdocs build --strict
```

The exhaustive slow suite and the reference-study regeneration commands are
listed in `docs/developer/release.md`.

## Archival

After the `v0.1.0` GitHub release is published and archived by Zenodo, record the
version-specific archive DOI in `CITATION.cff` and the repository citation
section. Do not insert a placeholder DOI into release metadata.
