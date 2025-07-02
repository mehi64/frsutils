# OWA Weighting Strategies

This document provides a comprehensive overview of common **Ordered Weighted Averaging (OWA)** weight strategies used in fuzzy rough set models. These strategies are essential in computing **fuzzy infimum** and **fuzzy supremum** approximations.

---

## 1. Overview of OWA Strategies

OWA strategies produce normalized weight vectors of length `n` based on predefined mathematical rules. They are used in fuzzy logic models to **aggregate** values with positional importance, commonly for approximating lower and upper bounds.

All strategies are used as follows:

Assume OWA weights are represented by OWA=<w1, w2, w3, ..., wn> where OWA vector is sorted in either ascending or descending order. And another vector V=<v1, v2, ... , vn> and V is not sorted (This could be the values of tnorm or implicator gotten in the interim calculations of upper and lower approximations). The sorted version of V is represented by t=<t1, t2, t3, ... , tn> which t1>t2>t3>...>tn. **t is always sorted in descending order** 

- For fuzzy **infimum**:
  - since infimum operator is basically a minimum, we need to assign higher OWA weights to the lower values of vector V to accentuate on lower values. Hence if OWA is ordered as ascending order (namely w1<w2<w3<...<wn), then sum(wi*ti ) will mimic infimum operator.  
- For fuzzy **supremum**.
  - since suprimum operator is basically a maximum, we need to assign lower OWA weights to the lower values of vector V to accentuate on higher values. Hence if OWA is ordered as descending order (namely w1>w2>w3>...>wn), then sum(wi*ti ) will mimic suprimum operator.
---

## 2. OWA Strategy Table

| Name         | Formula Example (Lower)                         | Parameters        | Aliases         |Tested and working in FRsutils|
|--------------|--------------------------------------------------|-------------------|-----------------|-------------------|
| **Linear**   | w_i = 2i / (n(n+1))                              | None              | linear          |yes|
| **Exponential** | w_i ∝ base^i                                 | base > 1          | exponential, exp|yes, **but there is no reference that I used to implement the method from**|
| **Harmonic** | w_i ∝ 1 / i                                      | None              | harmonic, harm  |yes, **but there is no reference that I used to implement the method from**|
| **Logarithmic** | w_i ∝ log(i + 1)                             | None              | logarithmic, log|yes, **but there is no reference that I used to implement the method from**|

All weights are normalized so that sum(w_i) = 1.

---

## 3. Notes

- Each strategy defines `_raw_weights` method.
- A unified `.weights(n: int, order: str = 'asc')` API is available on all strategies.
- These strategies are **pluggable** and registered under `OWAWeights` using the Registry pattern.

---

## 4. References

????

---