---
title: 'FRsutils: Fuzzy-Rough Set Utilities for Python'
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
affiliations:
  - name: Independent researcher
    index: 1
date: 14 June 2026
bibliography: paper.bib
---

# Summary

FRsutils is a scientific Python library for reusable fuzzy-rough set
computations. It provides a compact public API for constructing similarity
matrices, computing lower and upper approximations, deriving boundary regions,
and evaluating positive-region scores. The current release exposes three
fuzzy-rough model families through stable aliases: implicator/t-norm fuzzy rough
sets (`itfrs`), vaguely quantified rough sets (`vqrs`), and ordered weighted
averaging fuzzy rough sets (`owafrs`). FRsutils is built on NumPy [@harris2020numpy]
and follows a small, task-oriented API surface inspired by scientific Python and
scikit-learn conventions [@pedregosa2011scikit].

# Statement of need

Rough sets [@pawlak1982rough] and fuzzy rough sets
[@dubois1990rough; @radzikowska2002comparative] provide mathematical tools for
representing approximate concepts when samples are related by indiscernibility,
similarity, or gradual membership. These models are used in feature selection,
classification, prototype or instance selection, and fuzzy-rough data analysis.
However, research code for fuzzy-rough approximations is often written as
project-specific scripts, making it difficult to compare variants, reuse
components, or validate implementations across studies.

FRsutils addresses this gap by factoring common fuzzy-rough building blocks into
a reusable Python package. It is intended for researchers and research-software
developers who need tested implementations of similarities, t-norms,
implicators, fuzzy quantifiers, OWA weights, approximation models, and
positive-region computations. The package is also intended to support downstream
method development, where new fuzzy-rough sample-selection, scoring, or
oversampling methods need a dependable approximation layer rather than another
copy of ad hoc matrix code.

# State of the field

Several mature Python libraries provide general numerical computing, machine
learning, or GPU array functionality, including NumPy [@harris2020numpy],
scikit-learn [@pedregosa2011scikit], and CuPy [@okuta2017cupy]. These libraries
are essential infrastructure, but they do not provide a focused fuzzy-rough
approximation API. Related software such as fuzzy-rough-learn focuses on
machine-learning algorithms built from fuzzy-rough ideas
[@lenz2020fuzzyroughlearn]. FRsutils is complementary: it provides a lightweight
core approximation layer, reusable component registries, and a stable
`FRsutils.api` facade that downstream packages can depend on without importing
internal modules.

The library therefore occupies a narrow but useful layer between mathematical
research code and complete machine-learning estimators. It gives researchers a
common implementation surface for comparing ITFRS, VQRS, and OWAFRS variants,
while keeping execution metadata and backend behavior explicit enough for
reproducible experiments.

# Software design and functionality

The public entry point is `FRsutils.api`. Users can compute all approximation
outputs with `compute_approximations`, or call focused wrappers such as
`compute_lower_approximation`, `compute_upper_approximation`,
`compute_boundary_region`, and `compute_positive_region`. The returned
`FuzzyRoughApproximationResult` stores NumPy arrays for the lower approximation,
upper approximation, boundary region, and positive-region score, together with
execution metadata such as the selected model, engine, backend, and block size.

FRsutils supports dense reference execution and exact blockwise execution. Dense
execution builds or consumes a full pairwise similarity matrix and is useful for
small datasets and reference checks. Blockwise execution computes exact
approximations by processing similarity blocks, reducing the need to materialize
the full matrix. Optional CuPy-backed blockwise paths [@okuta2017cupy] are
available for selected internal operations while preserving NumPy arrays as the
public output contract. The GPU-related claims are intentionally model-specific:
ITFRS and VQRS may use GPU-backed similarity blocks and experimental
GPU-resident approximation accumulators, whereas OWAFRS currently claims only
GPU-backed similarity blocks.

The package also provides reusable component registries and builders for
similarities, t-norms, implicators, OWA weights, and fuzzy quantifiers. This
separation allows researchers to combine components in a controlled way while
keeping user-facing examples and downstream code on the public API.

The implementation distinguishes dense model classes from blockwise public
execution engines. Dense model classes act as small NumPy reference
implementations that are easy to inspect and test. The public blockwise path is
used when callers need the same approximation results without constructing the
entire pairwise matrix at once. Automated tests cover exact hand-computed
examples, model serialization, public API contracts, backend metadata, optional
CuPy behavior, and example scripts. This testing strategy is intended to make
FRsutils suitable as a dependency for later fuzzy-rough research software rather
than only as a collection of standalone scripts.

# Research impact

FRsutils makes fuzzy-rough approximation experiments easier to reproduce by
separating mathematical components, model construction, execution mode, and
result metadata. The package supports classical and noise-tolerant fuzzy-rough
families, including VQRS [@cornelis2007vqrs] and OWAFRS
[@cornelis2010owafrs], whose aggregation behavior is based on ordered weighted
averaging [@yager1988owa]. By providing tested approximation routines and a
stable import surface, FRsutils can serve as a foundation for downstream sample
selection, scoring, benchmarking, and educational material in fuzzy-rough
research. The stable public API also makes it easier to cite a single software
artifact from future method papers, including papers that propose new
fuzzy-rough oversampling or selection algorithms on top of the FRsutils core.

# AI usage disclosure

Generative AI tools were used to assist with drafting documentation, reviewing
API wording, and preparing release-checklist material. The author reviewed the
generated material, edited it for project-specific accuracy, and validated code
and examples with automated tests before release.

# Acknowledgements

The author thanks the developers and maintainers of NumPy, scikit-learn, CuPy,
and the broader fuzzy-rough research community whose work made this package
possible.
