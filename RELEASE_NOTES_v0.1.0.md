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
- Machine-readable installed-package validation for wheel and source
  distributions from read-only package trees and working directories.
- Reviewable JSON scientific reference data protected by provenance metadata,
  SHA-256 manifests, schema checks, and read-only NumPy reconstruction.

## Compatibility

- Python 3.10, 3.11, and 3.12 are the supported release targets.
- Core dependencies are NumPy and scikit-learn.
- NumPy remains the stable default backend.
- CuPy is optional and is not installed with the core package.
- The CUDA 12 optional environment is constrained to CuPy 14.x.
- Real-CUDA numerical validation for this release candidate was completed on an
  NVIDIA GeForce GTX 1050 Mobile with driver 535.309.01, CUDA Toolkit 12.0,
  Python 3.11.6, NumPy 2.3.5, and CuPy 14.1.1.
- The recorded GPU environment is a validated configuration, not an exhaustive
  compatibility guarantee.

## Validation

The real-CUDA smoke checks completed successfully for device discovery,
element-wise kernel execution, and matrix multiplication. The previously
reported non-CUDA logger-encoding regression has been corrected and its
targeted test now passes. After completion of the JSON reference-data
migration and final API hardening, the default suite reported 3006 passed
and 167 optional-backend skips, with 6633 slow tests deselected. The public API
suite reported 544 passed and 25 optional-backend skips. All 6633 slow tests
completed in four deterministic shards, including exhaustive OWAFRS
combinations and a public three-model dense/blockwise execution matrix.
Branch-aware coverage was 97 percent for `frsutils.api` and at least 95 percent
for `frsutils.core`, with both protected by 95 percent CI floors. Built wheel
and source distributions also passed real dense/blockwise public-API
computations from read-only package trees and read-only working directories.

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

The exhaustive slow suite and the reference-study regeneration commands are
listed in `docs/developer/release.md`.

## Archival

After the `v0.1.0` GitHub release is published and archived by Zenodo, record the
version-specific archive DOI in `CITATION.cff` and the repository citation
section. Do not insert a placeholder DOI into release metadata.
