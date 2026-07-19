# frsutils 0.1.1

`frsutils` 0.1.1 strengthens the scientific documentation, reproducibility
artifacts, release validation, and repository hygiene of the fuzzy-rough
computation library while preserving the package-root API introduced in 0.1.0.

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
  and manuscript-validation workflows.
- Machine-readable installed-package validation for wheel and source
  distributions from read-only package trees and working directories.
- Reviewable JSON scientific reference data protected by provenance metadata,
  SHA-256 manifests, schema checks, and read-only NumPy reconstruction.
- Corrected scientific concept pages with traceable primary references.

## Compatibility

- Python 3.10 or newer is required; continuous integration covers Python 3.10,
  3.11, and 3.12.
- Core dependencies are NumPy and scikit-learn.
- NumPy remains the stable default backend.
- CuPy is optional and is not installed with the core package.
- The CUDA 12 optional environment is constrained to CuPy 14.x.
- Recorded GPU evidence represents a validated configuration, not an exhaustive
  compatibility or performance guarantee.

## Reproducibility and validation

The repository includes automated checks for default and exhaustive tests,
branch-aware API and core coverage, citation metadata, documentation builds,
wheel and source-distribution contents, isolated installed-package behavior, and
dense/blockwise numerical equivalence. The reference-study snapshot records its
configuration, software environment, source revision, outputs, and checksums.

## Citation

Machine-readable software citation metadata is distributed in `CITATION.cff`.
An immutable archive DOI, when present in that file, identifies the corresponding
executable software release.
