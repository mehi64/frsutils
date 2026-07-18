# Public configuration contract

`frsutils` uses a flat, prefix-based configuration contract so researchers can
select fuzzy-rough components and pass their parameters without constructing
internal objects.

This page is the user-facing source of truth for:

- component selector names,
- parameter prefixes,
- registered aliases,
- component-specific parameters,
- model-specific configuration,
- flat-only public configuration and validation rules.

## The naming rule

A configurable component has a **selector** and, when needed, one or more
**component parameters**.

For model components, the selector follows:

```text
<component>_name = <registered alias>
```

Component parameters follow:

```text
<component>_<parameter> = <value>
```

For example, Yager is selected as the upper T-norm and its constructor parameter
`p` is routed with the same component prefix:

```python
result = compute_approximations(
    X,
    y,
    model="itfrs",
    ub_tnorm_name="yager",
    ub_tnorm_p=2.5,
    lb_implicator_name="lukasiewicz",
)
```

The OWA equivalent is:

```python
ub_owa_method_name="exponential"
ub_owa_method_base=2.5
```

The fuzzy quantifier equivalent is:

```python
lb_fuzzy_quantifier_name="linear"
lb_fuzzy_quantifier_alpha=0.1
lb_fuzzy_quantifier_beta=0.6
```

Similarity selectors are the two short public names `similarity` and
`similarity_tnorm`. Their parameters still use the same prefix rule:

```python
similarity="gaussian"
similarity_sigma=0.5

similarity_tnorm="yager"
similarity_tnorm_p=2.5
```

## Selector and prefix table

| Component role                | Selector                   | Parameter prefix       |
| ----------------------------- | -------------------------- | ---------------------- |
| Similarity                    | `similarity`               | `similarity_`          |
| Similarity aggregation T-norm | `similarity_tnorm`         | `similarity_tnorm_`    |
| Upper T-norm                  | `ub_tnorm_name`            | `ub_tnorm_`            |
| Lower implicator              | `lb_implicator_name`       | `lb_implicator_`       |
| Upper OWA weighting strategy  | `ub_owa_method_name`       | `ub_owa_method_`       |
| Lower OWA weighting strategy  | `lb_owa_method_name`       | `lb_owa_method_`       |
| Lower fuzzy quantifier        | `lb_fuzzy_quantifier_name` | `lb_fuzzy_quantifier_` |
| Upper fuzzy quantifier        | `ub_fuzzy_quantifier_name` | `ub_fuzzy_quantifier_` |

The parameter suffix is the corresponding registered component constructor
parameter. For example, the Yager T-norm constructor parameter is `p`, so the
public parameters are `ub_tnorm_p` or `similarity_tnorm_p`, depending on the
role of that T-norm.

## Which components each model uses

| Model    | Configurable model components                        | Public defaults                                 |
| -------- | ---------------------------------------------------- | ----------------------------------------------- |
| `itfrs`  | upper T-norm, lower implicator                       | `minimum`, `lukasiewicz`                        |
| `owafrs` | upper T-norm, lower implicator, upper OWA, lower OWA | `minimum`, `lukasiewicz`, `linear`, `linear`    |
| `vqrs`   | lower fuzzy quantifier, upper fuzzy quantifier       | quadratic lower `Q(0.2, 1.0)` and upper `Q(0.0, 0.6)` |

Similarity configuration is shared by all three models. The public API defaults
to `similarity="linear"` and `similarity_tnorm="minimum"`.

Parameters for components that the selected model does not use are rejected.

## Similarity methods

| Primary alias | Accepted aliases    | Component parameters | Constraint / default       |
| ------------- | ------------------- | -------------------- | -------------------------- |
| `linear`      | `linear`            | none                 | —                          |
| `gaussian`    | `gaussian`, `gauss` | `sigma`              | `sigma > 0`, default `0.1` |

Public parameter example:

```python
similarity="gaussian"
similarity_sigma=0.5
```

## T-norm methods

The same T-norm registry is used by `similarity_tnorm` and `ub_tnorm_name`.
Only the public prefix changes according to the component role.

| Primary alias | Accepted aliases                                     | Component parameters | Constraint / default   |
| ------------- | ---------------------------------------------------- | -------------------- | ---------------------- |
| `minimum`     | `minimum`, `min`, `goedel`, `standardintersection`   | none                 | —                      |
| `product`     | `product`, `prod`, `algebraic`                       | none                 | —                      |
| `lukasiewicz` | `lukasiewicz`, `luk`, `bounded`, `boundeddifference` | none                 | —                      |
| `drastic`     | `drastic`, `drasticproduct`                          | none                 | —                      |
| `einstein`    | `einstein`, `einsteinproduct`                        | none                 | —                      |
| `hamacher`    | `hamacher`, `hamacherproduct`                        | none                 | —                      |
| `nilpotent`   | `nilpotent`, `nilpotentminimum`                      | none                 | —                      |
| `yager`       | `yager`, `yg`                                        | `p`                  | `p > 0`, default `2.0` |

Examples:

```python
similarity_tnorm="yager"
similarity_tnorm_p=2.5
```

```python
ub_tnorm_name="yager"
ub_tnorm_p=2.5
```

A parameter belongs to the selected alias. Therefore this is invalid because
`minimum` has no `p` parameter:

```python
ub_tnorm_name="minimum"
ub_tnorm_p=2.5
```

## Implicator methods

Current implicators have no component-specific constructor parameters.

| Primary alias  | Accepted aliases               |
| -------------- | ------------------------------ |
| `lukasiewicz`  | `lukasiewicz`, `luk`           |
| `goedel`       | `goedel`                       |
| `kleenedienes` | `kleenedienes`, `kleene`, `kd` |
| `reichenbach`  | `reichenbach`                  |
| `goguen`       | `goguen`, `product`            |
| `rescher`      | `rescher`                      |
| `yager`        | `yager`                        |
| `weber`        | `weber`                        |
| `fodor`        | `fodor`                        |

Example:

```python
lb_implicator_name="goguen"
```

## OWA weighting methods

| Primary alias | Accepted aliases              | Component parameters | Constraint / default      |
| ------------- | ----------------------------- | -------------------- | ------------------------- |
| `linear`      | `linear`, `additive`          | none                 | —                         |
| `exponential` | `exponential`, `exp`, `gp`    | `base`               | `base > 1`, default `2.0` |
| `harmonic`    | `harmonic`, `harm`, `inv_add` | none                 | —                         |

Examples:

```python
ub_owa_method_name="exponential"
ub_owa_method_base=2.5

lb_owa_method_name="harmonic"
```

## Fuzzy quantifier methods

| Primary alias | Accepted aliases    | Component parameters               | Constraint / default                             |
| ------------- | ------------------- | ---------------------------------- | ------------------------------------------------ |
| `linear`      | `linear`            | `alpha`, `beta`, `validate_inputs` | `0 <= alpha < beta <= 1`; `validate_inputs=True` |
| `quadratic`   | `quadratic`, `quad` | `alpha`, `beta`, `validate_inputs` | `0 <= alpha < beta <= 1`; `validate_inputs=True` |

Examples:

```python
lb_fuzzy_quantifier_name="linear"
lb_fuzzy_quantifier_alpha=0.1
lb_fuzzy_quantifier_beta=0.6

ub_fuzzy_quantifier_name="quadratic"
ub_fuzzy_quantifier_alpha=0.1
ub_fuzzy_quantifier_beta=0.8
```

The VQRS defaults intentionally use different quantifiers:

```python
lb_fuzzy_quantifier_name="quadratic"
lb_fuzzy_quantifier_alpha=0.2
lb_fuzzy_quantifier_beta=1.0

ub_fuzzy_quantifier_name="quadratic"
ub_fuzzy_quantifier_alpha=0.0
ub_fuzzy_quantifier_beta=0.6
```

The lower quantifier represents a stricter `most` interpretation and the
upper quantifier represents a more permissive `some` interpretation. This
keeps the default lower and upper approximations semantically distinct.

The optional validation flags use the same routing rule:

```python
lb_fuzzy_quantifier_validate_inputs=False
ub_fuzzy_quantifier_validate_inputs=False
```

Disabling component input validation is intended for controlled advanced
workflows. The default is `True`.

## Configuring ITFRS

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="gaussian",
    similarity_sigma=0.5,
    similarity_tnorm="yager",
    similarity_tnorm_p=2.5,
    ub_tnorm_name="yager",
    ub_tnorm_p=2.0,
    lb_implicator_name="lukasiewicz",
)
```

ITFRS accepts upper T-norm and lower implicator configuration. OWA and fuzzy
quantifier parameters are not part of the ITFRS contract.

## Configuring OWAFRS

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="owafrs",
    similarity="linear",
    ub_tnorm_name="minimum",
    lb_implicator_name="lukasiewicz",
    ub_owa_method_name="exponential",
    ub_owa_method_base=2.5,
    lb_owa_method_name="harmonic",
)
```

The upper and lower OWA strategies are configured independently. A parameter
such as `ub_owa_method_base` is routed only to the upper OWA strategy.

## Configuring VQRS

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="vqrs",
    similarity="linear",
    lb_fuzzy_quantifier_name="linear",
    lb_fuzzy_quantifier_alpha=0.0,
    lb_fuzzy_quantifier_beta=0.5,
    ub_fuzzy_quantifier_name="quadratic",
    ub_fuzzy_quantifier_alpha=0.1,
    ub_fuzzy_quantifier_beta=0.8,
)
```

Lower and upper fuzzy quantifiers have separate prefixes so the two
approximation semantics can be configured independently.

## Using the same contract with similarity builders

`build_similarity_matrix` and `build_similarity_engine` use the same similarity
and similarity T-norm names:

```python
from frsutils import build_similarity_matrix

S = build_similarity_matrix(
    X,
    similarity="gaussian",
    similarity_sigma=0.5,
    similarity_tnorm="yager",
    similarity_tnorm_p=2.5,
)
```

```python
from frsutils import build_similarity_engine

engine = build_similarity_engine(
    X,
    engine="blockwise",
    block_size=256,
    similarity="gaussian",
    similarity_sigma=0.5,
    similarity_tnorm="yager",
    similarity_tnorm_p=2.5,
)
```

## Using the same contract with the scorer

`FuzzyRoughPositiveRegionScorer` exposes the current routed component
parameters as constructor parameters so they participate in scikit-learn
`get_params`, `set_params`, cloning, and grid-search-style workflows:

```python
from frsutils import FuzzyRoughPositiveRegionScorer

scorer = FuzzyRoughPositiveRegionScorer(
    model="itfrs",
    similarity="gaussian",
    similarity_sigma=0.5,
    similarity_tnorm="yager",
    similarity_tnorm_p=2.5,
    ub_tnorm_name="yager",
    ub_tnorm_p=2.0,
    lb_implicator_name="lukasiewicz",
)

scores = scorer.fit_score(X, y)
signed_boundary = scorer.signed_boundary_
```

The fitted `boundary_` attribute is retained as a backward-compatible alias of
`signed_boundary_`.

For current registered components, prefer the explicit scorer parameters shown
on this page. `extra_params` remains available for contract-defined flat
parameters that are not represented explicitly by a scorer release; unknown or
misspelled parameters are still rejected during computation.

## Public API scopes

The flat naming rule is shared across public endpoints, but each endpoint accepts
only parameters it can actually consume. This prevents irrelevant settings from
being silently ignored.

| Public endpoint | Accepted configuration scope |
| --- | --- |
| `compute_approximations` and approximation wrappers | similarity components plus components used by the selected fuzzy-rough model |
| `build_similarity_matrix` | similarity and similarity T-norm only |
| `build_similarity_engine` | similarity and similarity T-norm only |
| `build_fuzzy_rough_model` | components used by the selected fuzzy-rough model only |
| `FuzzyRoughPositiveRegionScorer` | approximation configuration through explicit constructor parameters or `extra_params` |

Execution controls such as `engine`, `block_size`, `backend`,
`return_similarity_matrix`, and `similarity_matrix` are named API arguments. They
are not flat component-configuration keys.

For example, this is rejected because a similarity builder cannot consume an
ITFRS upper T-norm setting:

```python
build_similarity_matrix(
    X,
    similarity="linear",
    ub_tnorm_name="minimum",
)
```

Likewise, this is rejected because ITFRS does not use OWA components:

```python
compute_approximations(
    X,
    y,
    model="itfrs",
    ub_owa_method_name="exponential",
)
```

## Flat config dictionaries and precedence

The same canonical names can be passed through a flat `config` mapping:

```python
config = {
    "type": "itfrs",
    "similarity": "gaussian",
    "similarity_sigma": 0.5,
    "similarity_tnorm": "minimum",
    "ub_tnorm_name": "minimum",
    "lb_implicator_name": "lukasiewicz",
}

result = compute_approximations(X, y, config=config)
```

For ordinary component values, additional flat keyword arguments override values
from `config`. The named `similarity` argument also acts as an explicit similarity
selection.

Model selection is deliberately stricter. The model can be supplied through the
`model`/`model_type` argument or the flat `type` key. Explicit model sources must
agree:

```python
compute_approximations(
    X,
    y,
    model="vqrs",
    config={"type": "itfrs"},
)
```

This raises `ValueError` instead of silently choosing one model. When no model
source is supplied, the public default is `"itfrs"`.

## Nested configuration is internal

FRsutils internally normalizes flat parameters into component specifications such
as:

```python
{"name": "yager", "params": {"p": 2.5}}
```

That nested representation is an implementation detail and is **not accepted at
the public API boundary**. For example, this is rejected:

```python
config = {
    "similarity": {"name": "gaussian", "params": {"sigma": 0.5}},
    "fr_model": {
        "type": "itfrs",
        "ub_tnorm": {"name": "minimum", "params": {}},
        "lb_implicator": {"name": "lukasiewicz", "params": {}},
    },
}

compute_approximations(X, y, config=config)
```

Keep public research configuration flat. This gives all public endpoints one
searchable naming contract, enables alias-specific typo validation, and avoids
exposing internal construction structure as a compatibility promise.

## Precomputed similarity matrices

A precomputed `similarity_matrix` can be supplied to dense approximation
execution. Because FRsutils did not construct that matrix, similarity provenance is
unknown unless it is tracked by the caller. Therefore similarity configuration must
not be supplied in the same call:

```python
S = build_similarity_matrix(X, similarity="gaussian", similarity_sigma=0.5)

result = compute_approximations(
    None,
    y,
    model="itfrs",
    similarity_matrix=S,
)

assert result.similarity is None
```

Passing `similarity`, `similarity_sigma`, `similarity_tnorm`, or other
`similarity_*` settings together with `similarity_matrix` is rejected because those
settings would not be used.

## Validation and typo protection

Public flat configuration is validated against the selected registered
component. Misspelled names and parameters that belong to a different alias fail
fast.

For example:

```python
compute_approximations(
    X,
    y,
    model="itfrs",
    ub_tnorm_name="yager",
    ub_tnorm_pp=2.0,
)
```

raises an error because the Yager parameter is named `p`, so the public key is
`ub_tnorm_p`.

Similarly:

```python
ub_tnorm_name="minimum"
ub_tnorm_p=2.0
```

is rejected because the minimum T-norm does not accept `p`.

This fail-fast behavior is intentional: scientific experiments should not
continue with silently ignored configuration parameters.

## Backward-compatible aliases

Older flat keys remain supported where the compatibility layer already defines
them, but new user code should use the canonical names on this page.

| Legacy key                  | Canonical key                                           |
| --------------------------- | ------------------------------------------------------- |
| `similarity_name`           | `similarity`                                            |
| `similarity_tnorm_name`     | `similarity_tnorm`                                      |
| `gaussian_similarity_sigma` | `similarity_sigma`                                      |
| `sigma`                     | `similarity_sigma` when Gaussian similarity is selected |
| `lb_alpha`                  | `lb_fuzzy_quantifier_alpha`                             |
| `lb_beta`                   | `lb_fuzzy_quantifier_beta`                              |
| `ub_alpha`                  | `ub_fuzzy_quantifier_alpha`                             |
| `ub_beta`                   | `ub_fuzzy_quantifier_beta`                              |

Canonical names are recommended for reproducible examples, documentation, and
new downstream packages.
