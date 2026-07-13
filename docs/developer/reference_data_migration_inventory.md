# Inline Reference Data Migration Inventory

## Status

This is the phase-one inventory for migrating scattered scientific test oracles to
JSON. No test, loader, manifest entry, or canonical reference-data file is changed
by this phase. The machine-readable snapshot is stored in
`reference_data_migration_inventory.json` beside this document.

The inventory was generated from 8 source
files and contains 15 candidate reference
cases. Every helper-derived output included in the snapshot was independently
recomputed from an explicit mathematical formula. Captured arrays and duplicate
fixtures were checked exactly for value, dtype, and shape; formula checks used a
strict absolute tolerance of `1e-15` where decimal floating-point evaluation can
differ by a final machine-representation bit.

## Source files locked for this inventory

| Source file | SHA-256 |
|---|---|
| `tests/api/test_dense_approximation_baseline_contract.py` | `5f360d82fdc74bf40d1d48472a3fce15798a08e331ba9b13a4b41d41aaaf9b60` |
| `tests/api/test_dense_similarity_baseline_contract.py` | `26c27820d6326460bb405442c21cb9a0c9b2f2f9eae4370e1c3ff7f547601dc7` |
| `tests/models_tests/test_itfrs_fast.py` | `078476e8229510f738b785bdf3baa17896ba47303d26af1969fc61a17730f7aa` |
| `tests/models_tests/test_vqrs_fast.py` | `205b236c55f2e41581cc53bf767db2aedbb4abb887c1f4e9053bfa4a33c49a49` |
| `tests/models_tests/test_owafrs_fast.py` | `41e30a0edf608f0cc5a521aac26cd9dd48269d546cca1cb2d007094d60b894d5` |
| `tests/core_tests/test_implicators.py` | `7e1de566bfeb8c1f58b3b6b1f9ea6e5bace87f67d0d7ef9c97fb30d466655595` |
| `tests/core_tests/test_fuzzy_quantifiers.py` | `08b3d15852cc5881f9fe190ed68d512a32cba9e7876222353af0e0faf51a078c` |
| `tests/core_tests/test_approximation_engines.py` | `23dcf33a849bd5fe41f04582e2a0538ee3691f016731e3ac36dee7386716b743` |

If any source digest changes, regenerate and review this inventory before starting
the next migration phase.

## Canonical target map

| Candidate case | Planned JSON | Provenance | Current test tolerance |
|---|---|---|---|
| `dense_approximation_four_sample_linear_similarity` | `approximation_baselines.json` | literal_expected_values | atol=1e-12 |
| `dense_linear_one_feature` | `similarities.json` | literal_expected_values | atol=1e-12 |
| `dense_linear_two_features_minimum` | `similarities.json` | literal_expected_values | atol=1e-12 |
| `dense_gaussian_one_feature_sigma_0_5` | `similarities.json` | literal_expected_values | atol=1e-12 |
| `itfrs_dense_fast_minimum_lukasiewicz` | `itfrs.json` | literal_hand_computed_expected_values | atol=1e-12 |
| `vqrs_dense_fast_asymmetric_quantifiers` | `vqrs.json` | helper_output_exactly_verified_against_independent_formula | atol=1e-12 |
| `owafrs_dense_fast_linear_owa` | `owafrs.json` | helper_output_exactly_verified_against_independent_formula | np.testing.assert_allclose defaults: rtol=1e-7, atol=0 |
| `implicator_boundary_scalar_cases` | `implicator_scalar.json` | literal_formula_and_boundary_expected_values | atol=1e-12 |
| `implicator_vector_branch_edge_cases` | `implicator_scalar.json` | literal_branch_expected_values | atol=1e-12 |
| `fuzzy_quantifier_linear_piecewise_1d` | `fuzzy_quantifiers.json` | literal_expected_values_verified_against_independent_formula | atol=1e-12 |
| `fuzzy_quantifier_quadratic_piecewise_1d` | `fuzzy_quantifiers.json` | literal_expected_values_verified_against_independent_formula | atol=1e-12 |
| `fuzzy_quantifier_linear_piecewise_2d` | `fuzzy_quantifiers.json` | literal_expected_values_verified_against_independent_formula | atol=1e-12 |
| `fuzzy_quantifier_quadratic_piecewise_2d` | `fuzzy_quantifiers.json` | literal_expected_values_verified_against_independent_formula | atol=1e-12 |
| `itfrs_blockwise_default_components` | `itfrs.json` | helper_output_exactly_verified_against_independent_formula | atol=1e-12 |
| `vqrs_blockwise_default_quantifiers` | `vqrs.json` | helper_output_exactly_verified_against_independent_formula | atol=1e-12 |

Planned new canonical files:

- `tests/reference_data/approximation_baselines.json`
- `tests/reference_data/fuzzy_quantifiers.json`

Planned existing files to extend:

- `tests/reference_data/implicator_scalar.json`
- `tests/reference_data/itfrs.json`
- `tests/reference_data/owafrs.json`
- `tests/reference_data/similarities.json`
- `tests/reference_data/vqrs.json`

## Duplicate and consolidation decisions

### `dense_one_feature_input`

- `tests/api/test_dense_approximation_baseline_contract.py:X_BASELINE`
- `tests/api/test_dense_similarity_baseline_contract.py:X_ONE_FEATURE`

**Decision:** Keep one canonical fixture if the final schemas support shared fixture references; otherwise duplicate only when domain files must remain self-contained.

### `vqrs_four_sample_similarity_and_labels`

- `tests/models_tests/test_vqrs_fast.py:VQRS_SIMILARITY_MATRIX/VQRS_LABELS`
- `tests/core_tests/test_approximation_engines.py:VQRS_SIMILARITY_MATRIX/VQRS_LABELS`

**Decision:** Store the fixture once in vqrs.json with multiple component configurations and expected-output records.

### `fuzzy_quantifier_coarse_cases`

- `test_quantifier_known_outputs`
- `piecewise formula tests`
- `boundary and midpoint property tests`

**Decision:** Use the full one-dimensional linear and quadratic piecewise cases as canonical data; keep property-oriented test inputs local where they test API shape or boundaries rather than a separate oracle.

## Important migration constraints

1. Assertion tolerances remain in Python tests. They must not be moved to JSON,
   because changing both an expected value and its tolerance could silently weaken
   a scientific regression test.
2. `tests/reference_data_loader.py` currently supports encoded `float64` and
   `int64` arrays only. The selected VQRS, OWAFRS, and blockwise fixtures contain
   string or object labels. The low-risk next step is to store those labels as plain
   JSON lists with explicit label metadata, unless loader support is deliberately
   expanded and contract-tested.
3. The dense approximation baseline currently includes negative OWAFRS boundary
   values. Phase two and phase three must copy those values verbatim rather than
   normalizing, clipping, or recomputing them from current production code.
4. The VQRS and OWAFRS fast tests currently build expected results through helpers
   that call production quantifier or OWA components. This inventory independently
   reproduced those formulas and verified exact equality before freezing the
   candidate values.
5. The main blockwise ITFRS and VQRS outputs were also independently reproduced
   from their defining formulas. Configuration-routing cases that use alternate
   components remain local contract tests unless a separate scientific oracle is
   justified.
6. Inputs for shape, validation, invalid values, empty datasets, singleton datasets,
   backend routing, aliases, and serialization remain in Python. They are software
   fixtures, not stable scientific reference data.
7. Later phases must preserve the number of tests. Only the location from which
   expected values are loaded should change.

## Phase-one completion criteria

- All eight agreed source files are represented.
- Candidate values, dtypes, and shapes are machine-readable.
- Duplicate fixtures have explicit consolidation decisions.
- Helper-derived outputs have an independent formula check.
- Source assertion tolerances are documented but excluded from candidate data.
- No existing project file was changed by phase one.
