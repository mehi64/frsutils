# Release and Paper Hardening Checklist

Use this checklist before tagging a release, preparing a software paper, or
sharing benchmark artifacts.

## 1. Scope and package boundary

- [ ] FRsutils contains fuzzy-rough core computation only.
- [ ] FRSMOTE and future sampling algorithms remain in `frsampling`.
- [ ] Documentation states that downstream packages depend on `FRsutils.api`.
- [ ] No new examples import from private `FRsutils.core` or `FRsutils.utils`
      paths unless the example is explicitly an internal developer example.

## 2. Public API examples

- [ ] `examples/phase7_public_api_quickstart.py` runs from a fresh checkout.
- [ ] README examples use `FRsutils.api`.
- [ ] Result objects are accessed by named fields, not tuple order.
- [ ] Execution metadata is shown for dense/blockwise/backend paths.

## 3. Benchmark artifacts

- [ ] `examples/phase7_benchmark_smoke.py` runs in CPU-only mode.
- [ ] `benchmarks/benchmark_fuzzy_rough_execution.py` writes JSON and CSV.
- [ ] CuPy/CUDA-unavailable rows are skipped, not treated as failures.
- [ ] Final paper numbers are generated on a fixed target machine, not inferred
      from CI or a small smoke test.
- [ ] Benchmark command line and environment metadata are preserved.

## 4. GPU and backend claims

- [ ] The release does not claim full GPU-native fuzzy-rough execution.
- [ ] The release does not claim GPU-native FRSMOTE.
- [ ] ITFRS and VQRS GPU-resident accumulator support is described as
      experimental.
- [ ] OWAFRS is explicitly documented as non-GPU-resident for the approximation
      accumulator in the current release.
- [ ] Public outputs are documented as NumPy arrays.

## 5. Tests

Run at minimum:

```bash
python -m pytest tests/api tests/benchmarks tests/examples -q
```

For `frsampling`, run from its repository root after installing or exposing
FRsutils on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD/src:../FRsutils" python -m pytest tests -q
```

## 6. Documentation consistency

- [ ] README claim matches `docs/paper_claims.md`.
- [ ] `docs/backend_execution_status.md` matches the actual metadata behavior.
- [ ] `docs/public_api_contract.md` describes the stable public boundary.
- [ ] `docs/phase_5_owafrs_non_gpu_resident_decision.md` remains linked from
      backend docs.
- [ ] `docs/phase_6_benchmark_suite.md` clearly says it provides a benchmark
      harness, not final benchmark numbers.
