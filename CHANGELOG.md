# Changelog

All notable user-visible changes to `frsutils` are documented in this file.
The format follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use semantic versioning.

## [Unreleased]

### Added

- A fully rewritten JOSS paper with substantive statement-of-need, state-of-the-field, software-design, research-impact, and AI-disclosure sections.
- Automated JOSS metadata and paper validation, repository-level readiness tests, and a draft-paper GitHub Actions workflow.
- A final JOSS submission checklist, GitHub–Zenodo archival guide, and release notes for version 0.1.0.
- Reproducible real-dataset reference study for ITFRS, VQRS, and OWAFRS.
- Dense/blockwise equivalence tables, repeated runtime observations, per-sample
  approximation outputs, environment metadata, figures, and artifact checksums.
- A study-specific optional dependency group and documentation page.
- A documented CUDA 12 installation path, validated real-GPU environment, and
  maintainer protocol for real-CUDA release checks.
- A deterministic public core execution matrix covering 1,360
  dense/blockwise comparisons across all canonical similarity configurations
  and representative or exhaustive model component combinations.
- A separate 95 percent branch-aware CI coverage gate for `frsutils.core`.
- A machine-readable installed public-API validator covering dense/blockwise
  parity, scorer fitted-state contracts, default logging silence, and
  read-only package execution.

## [0.1.0] - 2026-07-11

This is the first JOSS-oriented public release of the lowercase `frsutils`
package and stable public API.

### Added

- Stable package-root API for similarity construction, fuzzy-rough lower and
  upper approximations, boundary regions, and positive-region scores.
- Public model aliases for ITFRS, VQRS, and OWAFRS.
- Dense NumPy and exact blockwise execution paths with explicit execution
  metadata.
- Optional CuPy-backed blockwise execution for the documented model-specific
  paths while preserving NumPy public outputs.
- Reusable registries and builders for similarities, t-norms, implicators, OWA
  weights, and fuzzy quantifiers.
- Benchmark scripts, runnable public-API examples, and extensive contract,
  numerical, serialization, and backend tests.
- Contribution, code-of-conduct, support, issue, and pull-request policies.
- Continuous integration for Python 3.10, 3.11, and 3.12.
- Automated wheel and source-distribution validation with isolated
  installed-package computations from read-only package trees and working
  directories.
- MkDocs documentation and GitHub Pages deployment workflow.
- Human-readable JSON scientific reference data with SHA-256 integrity
  manifests, read-only loading, provenance metadata, and contract tests.

### Tested

- Confirmed CuPy device discovery, element-wise CUDA kernel execution, and
  matrix multiplication on an NVIDIA GeForce GTX 1050 Mobile.
- Completed the final default test suite with 3006 passed, 167
  optional-backend skips, and 6633 slow tests deselected.
- Completed the public API suite with 544 passed and 25 optional-backend skips;
  branch-aware API coverage remained 97 percent.
- Confirmed that the scientific reference-data contract suite passes and that
  the former CSV logger UTF-8 regression now passes.
- Completed all 6633 slow tests in four deterministic shards, including the
  exhaustive OWAFRS combinations and the public three-model execution matrix.
- Confirmed at least 95 percent branch-aware coverage for `frsutils.core` and
  retained the 95 percent CI floor for both the core and public API.
- Confirmed that wheel and source-distribution installations execute all three
  public models through dense and blockwise paths without requiring writable
  package or current-working-directory locations.

### Changed

- Constrained the optional CUDA 12 backend to CuPy 14.x and included CuPy's CUDA
  component dependencies in the optional installation extra.
- Documented CUDA runtime-header troubleshooting and the real-GPU validation
  environment used for the 0.1.0 release candidate.
- Canonical package and import name standardized as lowercase `frsutils`.
- Project licensing and package metadata standardized on BSD-3-Clause.
- Documentation navigation uses `docs/index.md` as the published site home.
- Project metadata and README documentation links point to the GitHub Pages
  site.
- Source distributions include project documentation, examples, benchmarks,
  tests, citation metadata, and JOSS paper sources.

## Earlier versions

For changes before version 0.1.0, consult the Git history, Git tags, and prior
GitHub releases. Historical entries are intentionally not reconstructed without
verifiable release artifacts.
