# OWA Weighting Strategies

This document provides a comprehensive overview of common **Ordered Weighted Averaging (OWA)** weight strategies used in fuzzy rough set models. These strategies are essential in computing **fuzzy infimum** and **fuzzy supremum** approximations.

---

## 1. Overview of OWA Strategies

OWA strategies produce normalized weight vectors of length `n` based on predefined mathematical rules. They are used in fuzzy logic models to **aggregate** values with positional importance, commonly for approximating lower and upper bounds.

All strategies are used as follows:

Assume OWA weights are represented by OWA= $\langle w_1, w_2, w_3, \ldots, w_n \rangle$ where OWA vector is sorted in either ascending or descending order. And another vector V= $\langle v_1, v_2, v_3, \ldots, v_n \rangle$ and V is not sorted (This could be the values of tnorm or implicator gotten in the interim calculations of upper and lower approximations). The sorted version of V is represented by t= $\langle t_1, t_2, t_3, \ldots, t_n \rangle$ which $t_1>t_2>t_3>...>t_n$. **t is always sorted in descending order** 

### For fuzzy **infimum**:

Since *infimum* operator is basically a *minimum*, we need to assign *higher* OWA weights to the *lower* values of vector V to accentuate on lower values. Hence if OWA is ordered as ascending order (namely $w_1<w_2<w_3<...<w_n$), then $\sum(w_i * t_i$) will mimic infimum operator.  

### For fuzzy **supremum**:

Since *suprimum* operator is basically a *maximum*, we need to assign *lower* OWA weights to the *lower* values of vector V to accentuate on higher values. Hence if OWA is ordered as descending order (namely $w_1>w_2>w_3>...>w_n$), then $\sum(w_i*t_i$) will mimic suprimum operator.

## 2. OWA Strategy Table

| Name                                    | Formula Example (Lower) | Parameters | Aliases                 | Reference |
| --------------------------------------- | ----------------------- | ---------- | ----------------------- | --------- |
| **Linear (Additive weights)**           | $w_i = 2_i / (n(n+1))$  | None       | linear, additive        | ref [1]   |
| **Exponential (Geometric Progression)** | $w_i ∝ base^i$          | base > 1   | exponential, exp, gp    | ref [2]   |
| **Harmonic (Inverse Additive)**         | $w_i ∝ 1 / i$           | None       | harmonic, harm, inv_add | ref [1]   |

All weights are normalized so that $\sum(w_i)$ = 1. Exponential weights are
computed from an exponent-shifted geometric progression whose largest raw value
is one. This produces the same normalized mathematical weights as
$base^1, \ldots, base^n$ while avoiding overflow for large `n`.

---

## 3. Notes in code

- Each strategy defines `_raw_weights` method in code.
- A unified `.weights(n: int, order: str = 'asc')` API is available on all strategies.
- These strategies are **pluggable** and registered under `OWAWeights` using the Registry pattern.

---

## 4. References

[1] Vluymans, Sarah; MacParthaláin, Neil; Cornelis, Chris; Saeys, Yvan, Weight selection strategies for ordered weighted average based fuzzy rough sets, 2019, Information Sciences. 501, 155–171. doi: https://doi.org/10.1016/j.ins.2019.05.085.

---

[2] Skowron, P., Faliszewski, P., & Lang, J., Finding a collective set of items: From proportional multirepresentation to group recommendation, Artificial Intelligence, 241, 191–216. DOI: 10.1016/j.artint.2016.09.003. 

---
