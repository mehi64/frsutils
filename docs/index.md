# frsutils documentation

`frsutils` provides reusable fuzzy-rough set components and task-oriented APIs
for similarity construction, lower and upper approximations, boundary regions,
and positive-region scores.

Use the navigation to find public API contracts, execution and backend behavior,
benchmark guidance, and the scientific definitions implemented by the library.

## User documentation

- [Public API](user/public_api.md): canonical imports, task-level APIs, result
  objects, scorer usage, and downstream-package boundaries.
- [Backends](user/backends.md): dense, blockwise, NumPy/CuPy behavior, backend
  metadata, and model-specific GPU claim boundaries.
- [Benchmarks](user/benchmarks.md): benchmark commands, recorded fields, paired
  NumPy/CuPy comparison, and safe interpretation rules.
- [Reproducible reference study](user/reference_study.md): real-dataset analysis,
  dense/blockwise equivalence evidence, environment capture, and committed results.
- [Glossary](user/glossary.md): short canonical definitions for execution and
  fuzzy-rough terminology.

## Scientific concepts

The concept notes are intentionally kept as separate files so each fuzzy-rough
component can be reviewed independently:

- [ITFRS](concepts/itfrs_info.md)
- [VQRS](concepts/vqrs_info.md)
- [OWAFRS](concepts/owafrs_info.md)
- [Similarities](concepts/similarities_info.md)
- [T-norms](concepts/tnorms_info.md)
- [Implicators](concepts/implicators_info.md)
- [OWA weights](concepts/owa_weights_info.md)

## Developer and release documentation

- [Release and JOSS validation](developer/release.md): release commands, test
  evidence, metadata checks, and safe software claims.
- [Final JOSS submission checklist](developer/joss_submission_checklist.md):
  automated and manual submission gates.
- [Software archive and DOI guide](developer/archive_and_doi.md): GitHub–Zenodo
  release archival and citation updates.

## Documentation policy

Keep new documentation in one of these places unless there is a strong reason to
split it:

- user-facing API, usage notes, or glossary terms: `docs/user/`
- scientific concept notes: `docs/concepts/`
- release, testing, and maintainer process: `docs/developer/`

Avoid adding short-lived status reports or duplicate checklist files to the
active docs tree. If temporary notes are needed, keep them outside active docs or
remove them before release.

## Community

Project governance and collaboration resources are maintained in the repository:

- [Contributing guide](https://github.com/mehi64/frsutils/blob/main/CONTRIBUTING.md)
- [Code of Conduct](https://github.com/mehi64/frsutils/blob/main/CODE_OF_CONDUCT.md)
- [Support policy](https://github.com/mehi64/frsutils/blob/main/SUPPORT.md)
- [Changelog](https://github.com/mehi64/frsutils/blob/main/CHANGELOG.md)
- [Issue tracker](https://github.com/mehi64/frsutils/issues)

## Local documentation build

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
mkdocs serve
```
