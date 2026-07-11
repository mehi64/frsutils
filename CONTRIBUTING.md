# Contributing to frsutils

Thank you for helping improve `frsutils`. The project aims to provide a small,
reliable, and well-documented scientific Python library for fuzzy-rough set
research. Contributions should preserve mathematical correctness, reproducible
behavior, and a stable public API for researchers and downstream packages.

## Before opening a contribution

- Read the [support policy](SUPPORT.md) to choose the right channel.
- Search existing issues and pull requests before creating a new one.
- For a substantial API, model, or mathematical change, open an issue first so
  the scientific contract and project scope can be agreed before implementation.
- Keep the core package focused on reusable fuzzy-rough computations. Application-
  specific algorithms should normally live in downstream packages unless there is
  a clear reusable-core justification.

## Development setup

Use Python 3.10 or newer. From a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Run the default test suite:

```bash
python -m pytest tests -ra
```

The default configuration excludes tests marked `slow`. Run the exhaustive model
combinations before a release or when changing model construction:

```bash
python -m pytest tests/models_tests -m slow -o addopts="" -ra
```

Build the documentation locally:

```bash
mkdocs build --strict
mkdocs serve
```

Validate package artifacts:

```bash
python -m pip install build twine
rm -rf build dist *.egg-info
python -m build
python -m twine check dist/*
```

## Scientific and API requirements

Changes to mathematical components, approximation models, or execution engines
should include:

1. A clear definition of the intended behavior and any mathematical convention.
2. Tests with independently derived or manually verifiable expected values.
3. Edge-case tests for shapes, labels, empty or invalid inputs, and numerical
   boundaries when relevant.
4. Dense-versus-blockwise equivalence tests when both execution modes apply.
5. Backend-contract tests when NumPy or CuPy behavior changes.
6. Documentation updates for user-visible behavior, limitations, and claim
   boundaries.

Do not make performance, GPU-residency, numerical-equivalence, or scalability
claims that are not covered by tests or reproducible benchmarks.

## Code and documentation style

- Every non-empty Python module must begin with
  `# SPDX-License-Identifier: BSD-3-Clause` followed by a short module docstring.
- Public classes, methods, and functions use compact NumPy-style docstrings.
- Document input and output shapes for scientific array operations when useful.
- Keep private-helper documentation short; comments should explain non-obvious
  reasoning rather than restate code.
- Add pytest tests for every behavior change or bug fix.
- Prefer imports from the public `frsutils` package boundary in examples and user
  documentation.
- Keep generated files, datasets, logs, benchmark outputs, local environments,
  and test reports out of commits.

## Documentation changes

Documentation lives under `docs/` and is published with MkDocs. Keep:

- user-facing usage and contracts in `docs/user/`,
- scientific definitions in `docs/concepts/`,
- maintainer and release procedures in `docs/developer/`.

Run `mkdocs build --strict` before opening a pull request that changes
Markdown, navigation, examples, or links.

## Pull requests

Create a focused branch and keep each pull request limited to one coherent
change. A pull request should:

- explain the problem and the chosen approach,
- identify scientific or public-API effects,
- include tests and documentation where applicable,
- update `CHANGELOG.md` for user-visible changes,
- pass the GitHub Actions checks,
- avoid unrelated formatting or refactoring.

AI-assisted contributions are acceptable, but contributors remain responsible
for every submitted line. Review, run, and understand generated code; verify
scientific claims independently; and disclose material AI assistance in the pull
request when it affected implementation, tests, or documentation.

## Licensing

By submitting a contribution, you agree that it may be distributed under the
project's BSD-3-Clause license. Do not submit code, data, text, or figures that
you do not have permission to redistribute.
