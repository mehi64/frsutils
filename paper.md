---
title: 'frsutils: Fuzzy-Rough Set Utilities for Python'
tags:
  - Python
  - fuzzy-rough sets
  - rough sets
  - fuzzy sets
  - positive region
  - similarity matrices
authors:
  - name: Mehran Amiri
    affiliation: 1
    corresponding: true
affiliations:
  - name: Independent Researcher, Germany
    index: 1
date: 18 July 2026
bibliography: paper.bib
---

# Summary

`frsutils` is a scientific Python library for computing core fuzzy-rough set
approximations. Fuzzy-rough models describe whether an observation belongs
certainly, possibly, or ambiguously to a concept when observations are related
by graded similarity rather than exact equality. The library provides a compact
public API for similarity matrices, lower and upper approximations, boundary
regions, and positive-region scores. It exposes three model families through a
shared interface: implicator/t-norm fuzzy-rough sets (ITFRS)
[@radzikowska2002comparative], vaguely quantified rough sets (VQRS)
[@cornelis2007vqrs], and ordered weighted averaging fuzzy-rough sets (OWAFRS)
[@yager1988owa; @cornelis2010owafrs]. Dense NumPy execution is provided as a
reference implementation, together with exact blockwise execution and optional,
model-specific CuPy acceleration [@harris2020numpy; @okuta2017cupy].

# Statement of need

Rough sets [@pawlak1982rough] and fuzzy-rough sets
[@dubois1990rough; @radzikowska2002comparative] are used in research on feature
selection, classification, prototype and instance selection, data complexity,
and learning from uncertain or overlapping concepts. These applications depend
on a recurring computational layer: construct a graded relation between
samples, evaluate lower and upper approximations, and derive quantities such as
the boundary or positive region. In practice, this layer is often reimplemented
inside individual experiments. Small differences in orientation, aggregation,
self-comparison, class handling, or parameter defaults can then make results
hard to compare and implementations difficult to reuse.

`frsutils` addresses this reproducibility and reuse problem for researchers and
research-software developers working in Python. It supplies tested mathematical
components, model implementations, and task-level functions behind one stable
package-root API. The same interface can therefore be used to compare model
families, validate a new method against a dense reference implementation, or
serve as the approximation dependency of downstream sample-selection and
imbalanced-learning software. The blockwise engine also addresses a practical
limitation of pairwise fuzzy-rough computations: a complete similarity matrix
requires quadratic storage in the number of samples. `frsutils` processes exact
similarity blocks instead of requiring that full matrix to be materialized,
while retaining explicit execution metadata and the same scientific outputs.

# State of the field

Existing software covers related but different levels of the fuzzy-rough
research stack. The R package `RoughSets` implements a broad collection of
rough-set and fuzzy-rough concepts and applications, including approximation,
feature and instance selection, rule induction, and classification
[@riza2014roughsets]. `fuzzy-rough-learn` provides Python machine-learning
algorithms and data descriptors built with fuzzy-rough ideas, including
classifiers, regressors, feature selection, and prototype selection
[@lenz2020fuzzyroughlearn]. Its public API is organized primarily around those
learning algorithms and descriptors.

`frsutils` deliberately occupies a narrower layer. Its primary public objects
are the approximation operations and their components, rather than complete
predictive estimators. It places ITFRS, VQRS, and OWAFRS behind a common result
contract; exposes per-sample lower, upper, boundary, and positive-region values;
and provides both dense reference and exact blockwise execution with recorded
backend and engine metadata. These requirements did not map cleanly onto an
extension of an estimator-oriented package: downstream research code needed a
small dependency whose public boundary was the mathematical approximation
itself, whose model semantics were selectable, and whose execution strategy
could change without changing returned scientific values. Building `frsutils`
therefore adds a reusable approximation-engine layer rather than duplicating
the broader learning functionality of `RoughSets` or `fuzzy-rough-learn`.
NumPy and scikit-learn remain foundational dependencies and conventions rather
than competing alternatives [@harris2020numpy; @pedregosa2011scikit].

# Software design

The central design choice is to separate the stable research contract from
model construction and execution details. Users call package-root functions
such as `compute_approximations` and receive a
`FuzzyRoughApproximationResult` containing NumPy arrays and execution metadata.
A single result type makes model comparisons and downstream integration
predictable, while registries and builders keep similarities, t-norms,
implicators, fuzzy quantifiers, and OWA weights independently configurable.
This balances a small user-facing API against the extensibility needed for
methodological research.

A second trade-off concerns transparency, memory, and speed. Dense model classes
materialize or consume a complete similarity matrix. They are concise,
inspectable reference implementations and are valuable for hand-computed tests
and regression checks, but their pairwise storage limits scale. The blockwise
engine computes the same approximations from successive similarity blocks. It
reduces similarity-matrix materialization without introducing an approximate
algorithm, at the cost of additional control flow and sensitivity of runtime to
the selected block size. Dense and blockwise results are continuously compared
in tests and in the distributed reference study.

A third choice is to keep backend-specific arrays internal. NumPy arrays are the
public output contract even when CuPy is used for supported blockwise
operations. This avoids making downstream packages conditional on GPU array
types and keeps serialization and tests consistent. The trade-off is that output
conversion may prevent a fully GPU-resident end-to-end pipeline. Claims are
therefore model-specific: ITFRS and VQRS support documented experimental
GPU-resident accumulation paths, while OWAFRS currently uses GPU-backed
similarity blocks without claiming GPU-resident approximation accumulation.
GPU support is optional so that the core installation remains portable.

These choices matter scientifically because they permit two forms of
verification. Individual formulas can be checked in small dense examples, and
larger workflows can verify numerical equivalence across execution engines.
Tests also cover configuration, serialization, package-root API contracts,
backend metadata, and optional backend behavior. The resulting architecture is
intended to let new fuzzy-rough methods reuse and test their approximation layer
instead of embedding another private implementation.

# Research impact statement

The repository includes a reproducible reference study under
`studies/fuzzy_rough_reference_study/`. It uses only the stable package-root API
to apply ITFRS, VQRS, and OWAFRS to three binary tasks derived from public
datasets distributed with scikit-learn. For every model and task, it records
per-sample lower and upper approximations, signed boundary values,
positive-region scores, runtime observations, resolved configuration, software
and system versions, and an artifact checksum manifest. No external dataset
download or unpublished code is required.

The stored reference run contains nine real-data dense/blockwise comparisons;
all satisfy the declared absolute tolerance of $10^{-12}$, with a maximum
recorded discrepancy of approximately $6.7\times10^{-16}$. A separate fixed
benchmark sweep records 27 successful model, engine, and problem-size cases in
machine-readable CSV and JSON files. These results are software-validation
artifacts, not a claim that one fuzzy-rough model or backend is universally
superior. They make model semantics, execution choices, and
numerical-equivalence evidence directly reproducible.

Beyond the bundled study, `frsutils` is used as the fuzzy-rough computation
layer in the author's ongoing FRSMOTE research workflow, where positive-region
scores guide minority-sample selection for synthetic oversampling. This use
motivated the package-root approximation contract, component configuration,
and exact dense/blockwise equivalence requirements. The downstream method is
not required to reproduce the public reference study, but it provides a concrete
research use case in which the library replaces experiment-specific
reimplementation of fuzzy-rough approximations.

# AI usage disclosure

OpenAI ChatGPT, primarily the GPT-5.5 Thinking and GPT-5.6 Thinking models, was
used during development. Assistance covered code generation and refactoring,
test-scaffolding suggestions, documentation drafting and editing, review of API
wording, preparation of the reproducible reference-study workflow, release
documentation and repository-quality review, and drafting and copy-editing
portions of this
manuscript. The author made the problem-framing, mathematical, architectural design,
public-API, licensing, and release decisions. The author reviewed and modified
all AI-assisted material, executed the software and examples, inspected the
produced artifacts, and validated code behavior with automated tests and
numerical comparisons. The author accepts full responsibility for the accuracy,
originality, licensing compliance, and scientific claims of the submitted
software and paper.

# Acknowledgements

The author thanks the developers and maintainers of NumPy, scikit-learn, CuPy,
`RoughSets`, and `fuzzy-rough-learn`, and the broader fuzzy-rough research
community whose theoretical and software contributions made this work possible.
This work received no external funding.
