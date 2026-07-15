# Reproducible reference study

FRsutils includes a real-dataset research artifact under
`studies/fuzzy_rough_reference_study/`. It provides concrete, inspectable
evidence that the public API is usable in a downstream scientific workflow.

The study applies ITFRS, VQRS, and OWAFRS to three binary tasks derived from
public scikit-learn datasets. It records per-sample approximation values,
verifies dense/blockwise numerical equivalence, measures repeated runtimes,
runs the repository's synthetic execution benchmark, captures the software and
hardware environment, and writes checksums for generated outputs.

## Run it

```bash
python studies/fuzzy_rough_reference_study/run_study.py
```

The canonical configuration is:

```text
studies/fuzzy_rough_reference_study/study_config.json
```

The study documentation and committed result snapshot are available at:

```text
studies/fuzzy_rough_reference_study/README.md
studies/fuzzy_rough_reference_study/results/
```

## Scope of the evidence

The artifact supports the following claims:

- one stable package-root API can execute all three documented model families,
- exact blockwise NumPy execution reproduces dense NumPy outputs within a fixed
  tolerance for the configured tasks,
- configurations, per-sample outputs, runtimes, package versions, platform
  metadata, and artifact checksums are machine-readable,
- the study can be regenerated without downloading external datasets.

It does not claim that one fuzzy-rough model is universally superior, that
runtime values generalize to other hardware, or that the benchmark fully
measures native allocator memory.

# 
