# ITFRS model

## 1.Introduction

Implicator-Tnorm Fuzzy Rough Sets (ITFRS) are a core variant of fuzzy rough set models used for approximating uncertain or imprecise concepts by combining fuzzy logic with rough set theory. 
This model utilizes fuzzy implicators and t-norms as the building blocks for computing lower and upper approximations

---

## 2. Intuition

In many real-world applications, especially involving imprecise or vague information, binary logic and crisp set boundaries are inadequate. Fuzzy Rough Sets allow gradual membership and uncertainty modeling, and ITFRS provides a well-defined, 
interpretable, and mathematically grounded way to compute these approximations using standard fuzzy logic operators.

---

## 3. Definitions

Let:

  - U be the universe of discourse.
  - X be a fuzzy set in U.
  - R(x, y) be the fuzzy similarity (or tolerance) between elements x and y.
  - μ_X(y) be the membership degree of y in fuzzy set X.
  - I be a fuzzy implicator.
  - T be a t-norm.


**Lower Approximation in ITFRS**

The lower approximation of X at point x ∈ U is defined by:

\underline{R}(X)(x) = \inf_{y \in U} I(R(x, y), \mu_X(y))


  - I(R(x, y), μ_X(y)) represents the degree to which similarity implies membership.

  - The inf operator aggregates the minimal certainty across all comparisons.


**Upper Approximation in ITFRS**

The upper approximation of X at point x is:

\overline{R}(X)(x) = \sup_{y \in U} T(R(x, y), \mu_X(y))


  - T(R(x, y), μ_X(y)) reflects the degree to which similarity and membership co-occur.

  - The sup operator captures maximal potential inclusion.


## 4. Interpretability and Use Cases

  - **Lower Approximation**: Elements definitely belonging to the target concept.

  - **Upper Approximation**: Elements possibly belonging.

This separation is useful in:

  - Feature Selection

  - Instance-based classification

  - Uncertainty reasoning

---
### In ITFRS (THIS SECTION NEEDS TO BE CHECKED)
#### lower approximation for each instance: 
<img src="images/ITFRS/lower.JPG" alt="lower aaproximation" width="250"/>

#### upper approximation for each instance:
<img src="images/ITFRS/upper.JPG" alt="upper aproximation" width="250"/>


  - Since for the calculations of lower approximation, we calculate Inf which is basically a minimum, to exclude the same instance from calculations we don’t need anything because the main diagonal is set to 1.0 which is ignored by min operator. To be sure all is correct, inside code, we set main diagonal to 1.0
  - Since for the calculations of upper approximation, we calculate sup which is basically a maximum, to exclude the same instance from calculations we need to set the main diagonal to 0.0 which is ignored by max operator. Otherwise all upper approxamations will be 1.0.
  - In ITFRS, POS(x) = lower_approximation(x) where x  is a data instance, and datasets having crisp classes.????????