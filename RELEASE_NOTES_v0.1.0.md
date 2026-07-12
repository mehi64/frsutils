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
element-wise kernel execution, and matrix multiplication. The ordinary suite
then executed with real CuPy enabled and reported 2802 passed tests and one
non-CUDA logger-encoding failure. A final clean test record must be captured
after that UTF-8 logger fix before publishing the release.

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
