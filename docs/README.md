# FRsutils documentation

This folder contains the active project documentation for FRsutils. It is kept
small on purpose: user-facing usage, scientific concepts, backend behavior, and
release/JOSS validation are separated, while duplicated checklist and decision
notes have been merged into fewer files.

## User documentation

- [Public API](user/public_api.md): canonical imports, task-level APIs, result
  objects, scorer usage, and downstream-package boundaries.
- [Backends](user/backends.md): dense, blockwise, NumPy/CuPy behavior, backend
  metadata, and model-specific GPU claim boundaries.
- [Benchmarks](user/benchmarks.md): benchmark commands, recorded fields, paired
  NumPy/CuPy comparison, and safe interpretation rules.
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

- [Release and JOSS validation](developer/release.md): release checklist,
  validation commands, metadata checks, JOSS submission steps, and safe paper
  wording.

## Documentation policy

Keep new documentation in one of these places unless there is a strong reason to
split it:

- user-facing API, usage notes, or glossary terms: `docs/user/`
- scientific concept notes: `docs/concepts/`
- release, testing, and maintainer process: `docs/developer/`

Avoid adding short-lived status reports or duplicate checklist files to the
active docs tree. If temporary notes are needed, keep them outside active docs or
remove them before release.
