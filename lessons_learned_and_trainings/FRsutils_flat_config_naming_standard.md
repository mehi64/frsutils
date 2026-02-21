# FRsutils — Flat Parameter Naming Standard (Sklearn-friendly, Nested-internal)

This document defines the **external (flat)** parameter naming conventions that remain fully compatible with
**scikit-learn / imbalanced-learn Pipeline + GridSearchCV**, while allowing the library to internally convert
flat configs into **nested dictionaries** for clean component isolation.

> **Important:** In scikit-learn, the double-underscore `__` is reserved for *routing parameters* to sub-estimators
> (e.g., `pipeline_step__param`). Therefore, **this standard does not use `__` inside FRSMOTE parameter names**.
> Use `__` only at the pipeline level: `frsmote__<param>`.

---

## A) Core rules

| Rule | Description | Example |
|---|---|---|
| R1 | Use flat keys externally; convert to nested internally. | external: `similarity_sigma=0.5` → internal: `similarity.params.sigma=0.5` |
| R2 | Component selection uses a single key that stores the component **name**. | `similarity="gaussian"` |
| R3 | Component parameters use the pattern: `<component_prefix>_<param_name>` | `similarity_sigma=0.5` |
| R4 | Never put `__` inside estimator parameter names. | ✅ `ub_tnorm_p` / ❌ `ub_tnorm__p` |
| R5 | Keys must be stable and clone-friendly (avoid passing complex objects via sklearn params). | Prefer primitives (str/int/float/bool/None) |
| R6 | Backward-compatibility aliases are optional; if supported, they must map to the new standard. | `gaussian_similarity_sigma` → `similarity_sigma` |

---

## B) Pipeline + GridSearch usage pattern

| Where | Syntax | Example |
|---|---|---|
| Pipeline step routing | `<step_name>__<param>` | `frsmote__k_neighbors=5` |
| Estimator param naming (this standard) | `<component_prefix>_<param>` | `frsmote__similarity_sigma=0.5` |

---

## C) Parameter naming table (external / flat)

### C1) Oversampler-level parameters (FRSMOTE / BaseOverSampler)

| Group | Flat key | Type | Notes |
|---|---|---:|---|
| oversampler | `sampling_strategy` | str / dict | imbalanced-learn sampling strategy |
| oversampler | `instance_ranking_strategy` | str / dict | e.g., `"pos"` or dict per class |
| oversampler | `sampling_ratio` | None / float / dict | your project-specific per-class ratio |
| oversampler | `k_neighbors` | int | K for NN selection |
| oversampler | `random_state` | None / int | sklearn-compatible |
| oversampler | `bias_interpolation` | bool | enables biased interpolation |

---

### C2) Similarity matrix construction

| Component | Flat selector key | Flat parameter keys (pattern) | Internal target |
|---|---|---|---|
| Similarity | `similarity` | `similarity_<param>` | `nested["similarity"] = {"name": ..., "params": {...}}` |
| Similarity T-norm | `similarity_tnorm` | `similarity_tnorm_<param>` | `nested["similarity_tnorm"] = {"name": ..., "params": {...}}` |

**Common examples (not exhaustive):**

| Similarity name | Flat keys commonly used |
|---|---|
| `gaussian` | `similarity_sigma` |
| `linear` | (usually no extra params) |
| `cosine` | (usually no extra params) |

| T-norm name | Flat keys commonly used |
|---|---|
| `minimum` | (none) |
| `product` | (none) |
| `yager` | `similarity_tnorm_p` |
| `lambda` | `similarity_tnorm_lambda` |

---

### C3) Fuzzy-Rough model selection

| Purpose | Flat key | Type | Notes |
|---|---|---:|---|
| model type | `type` | str | one of: `itfrs`, `owafrs`, `vqrs` |

---

### C4) ITFRS components

| Component | Flat selector key | Flat parameter keys (pattern) | Internal target |
|---|---|---|---|
| Upper T-norm | `ub_tnorm_name` | `ub_tnorm_<param>` | `nested["fr_model"]["ub_tnorm"]` |
| Lower implicator | `lb_implicator_name` | `lb_implicator_<param>` | `nested["fr_model"]["lb_implicator"]` |

---

### C5) OWAFRS additional components

| Component | Flat selector key | Flat parameter keys (pattern) | Internal target |
|---|---|---|---|
| Upper OWA | `ub_owa_method_name` | `ub_owa_method_<param>` | `nested["fr_model"]["ub_owa_method"]` |
| Lower OWA | `lb_owa_method_name` | `lb_owa_method_<param>` | `nested["fr_model"]["lb_owa_method"]` |

---

### C6) VQRS components

| Component | Flat selector key | Flat parameter keys (pattern) | Internal target |
|---|---|---|---|
| Lower fuzzy quantifier | `lb_fuzzy_quantifier_name` | `lb_fuzzy_quantifier_<param>` | `nested["fr_model"]["lb_fuzzy_quantifier"]` |
| Upper fuzzy quantifier | `ub_fuzzy_quantifier_name` | `ub_fuzzy_quantifier_<param>` | `nested["fr_model"]["ub_fuzzy_quantifier"]` |

---

## D) Recommended internal nested structure (target)

The normalizer should convert flat inputs into a nested structure similar to:

```python
{
  "oversampler": {...},
  "similarity": {"name": "...", "params": {...}},
  "similarity_tnorm": {"name": "...", "params": {...}},
  "fr_model": {
    "type": "...",
    "ub_tnorm": {"name": "...", "params": {...}},
    "lb_implicator": {"name": "...", "params": {...}},
    "ub_owa_method": {"name": "...", "params": {...}},
    "lb_owa_method": {"name": "...", "params": {...}},
    "lb_fuzzy_quantifier": {"name": "...", "params": {...}},
    "ub_fuzzy_quantifier": {"name": "...", "params": {...}},
  }
}
```

Only the entries required by the selected `type` must be present.

---

## E) Full example (external config for sklearn)

```python
config = {
  "type": "owafrs",

  "similarity": "gaussian",
  "similarity_sigma": 0.5,
  "similarity_tnorm": "minimum",

  "lb_implicator_name": "lukasiewicz",
  "ub_tnorm_name": "product",
  "ub_owa_method_name": "linear",
  "lb_owa_method_name": "linear",

  "k_neighbors": 3,
  "random_state": None,
  "sampling_strategy": "auto",
  "instance_ranking_strategy": "pos",
  "sampling_ratio": {"1": 0.6, "2": 0.8},
  "bias_interpolation": False,
}
```

In a pipeline grid-search:

```python
param_grid = {
  "frsmote__similarity": ["gaussian"],
  "frsmote__similarity_sigma": [0.2, 0.5, 1.0],
  "frsmote__ub_tnorm_name": ["minimum", "product"],
  "frsmote__k_neighbors": [3, 5],
}
```

---

## F) Optional backward-compatibility aliases

If you decide to support legacy keys, normalize them early:

| Legacy key | New key |
|---|---|
| `similarity_name` | `similarity` |
| `similarity_tnorm_name` | `similarity_tnorm` |
| `gaussian_similarity_sigma` | `similarity_sigma` |

---

## G) Notes on sklearn compatibility

1. Keep **all tunable knobs** as flat keys returned by `get_params()`.
2. Nested dictionaries should be **internal only** and rebuilt from flat params during `set_params()` / `configure()` / pre-build.
3. Avoid exposing mutable nested structures as sklearn params unless you implement robust flatten/unflatten in `get_params/set_params`.
