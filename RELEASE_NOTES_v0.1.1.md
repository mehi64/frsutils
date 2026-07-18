# frsutils 0.1.1

`frsutils` 0.1.1 is the JOSS-submission release of the fuzzy-rough computation
library. It keeps the package-root API introduced in 0.1.0 and concentrates on
scientific documentation, reproducibility evidence, release validation, and
repository hygiene.

## Highlights

- ITFRS, VQRS, and OWAFRS through a shared approximation result contract.
- Package-root functions for similarity matrices, lower and upper
  approximations, signed boundary regions, and positive-region scores.
- Dense NumPy reference execution and exact blockwise execution.
- Optional, model-specific CuPy-backed execution paths with conservative claims.
- Reusable registries and builders for similarities, T-norms, implicators,
  fuzzy quantifiers, and OWA weights.
- A reproducible real-dataset reference study with resolved configuration,
  environment metadata, machine-readable outputs, figures, and checksums.
- Continuous integration, package-build validation, documentation deployment,
  and JOSS paper validation workflows.
- Machine-readable installed-package validation for wheel and source
  distributions from read-only package trees and working directories.
- Reviewable JSON scientific reference data protected by provenance metadata,
  SHA-256 manifests, schema checks, and read-only NumPy reconstruction.
- Corrected scientific concept pages with traceable primary references.

## Compatibility

- Python 3.10, 3.11, and 3.12 are the supported release targets.
- Core dependencies are NumPy and scikit-learn.
- NumPy remains the stable default backend.
- CuPy is optional and is not installed with the core package.
- The CUDA 12 optional environment is constrained to CuPy 14.x.
- The recorded GPU environment is one validated configuration, not an
  exhaustive compatibility guarantee.

## Validation

Before publishing the release, run:

```bash
python -m pytest tests -ra
python scripts/validate_installed_public_api.py \
  --output-json test_reports/source_public_api_validation.json
python scripts/validate_joss_submission.py
python -m build
python -m twine check dist/*
mkdocs build --strict
```

Run the exhaustive slow suite and regenerate the reference study from the clean
release commit as described in `docs/developer/release.md`.

## Archival

After the `v0.1.1` GitHub release is published and archived by Zenodo, record the
version-specific archive DOI in `CITATION.cff` and the repository citation
section. Do not insert an invented or placeholder DOI into active release
metadata.
