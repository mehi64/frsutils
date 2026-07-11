# Reproducible studies

This directory contains research artifacts that exercise the stable public
`frsutils` API. Study code may use repository benchmark helpers, but scientific
approximation computations must import from `frsutils` rather than internal
`frsutils.core` modules.

Current study:

- [`fuzzy_rough_reference_study`](fuzzy_rough_reference_study/README.md): a
  real-dataset analysis of ITFRS, VQRS, and OWAFRS with dense/blockwise
  equivalence checks, execution benchmarks, environment capture, and
  machine-readable results.
