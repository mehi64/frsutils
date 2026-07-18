# Reproducible reference study

`frsutils` includes a real-dataset research artifact under
`studies/fuzzy_rough_reference_study/`. It provides inspectable evidence that the
package-root API can support a downstream scientific workflow.

The study applies ITFRS, VQRS, and OWAFRS to three binary tasks derived from
public scikit-learn datasets. It records per-sample approximation values,
verifies dense/blockwise numerical equivalence, measures repeated runtimes,
runs the repository benchmark profile, captures the software and hardware
environment, and writes checksums for generated outputs.

## Run the study

```bash
python studies/fuzzy_rough_reference_study/run_study.py
```

The canonical configuration is:

```text
studies/fuzzy_rough_reference_study/study_config.json
```

Detailed methods and the committed result snapshot are available in:

```text
studies/fuzzy_rough_reference_study/README.md
studies/fuzzy_rough_reference_study/results/
```

## Scope of the evidence

The artifact supports the following claims:

- one stable package-root API executes all three documented model families;
- exact blockwise NumPy execution reproduces dense NumPy outputs within the
  configured tolerance;
- configurations, per-sample outputs, runtimes, package versions, platform
  metadata, and checksums are machine-readable;
- no external dataset download is required.

It does not claim that one fuzzy-rough model is universally superior, that
runtime values generalize to other hardware, or that Python-level memory
measurements capture every native allocator.

For a release, regenerate the snapshot after committing the release changes and
confirm that `environment.json` records the release version, the release commit,
and `git_worktree_dirty: false`.
