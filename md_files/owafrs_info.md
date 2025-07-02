# OWAFRS model

## 1.Introduction

Ordered Weighted Averaging Fuzzy Rough Sets (OWAFRS) are an advanced model in fuzzy rough set theory, used to estimate the lower and upper approximations of fuzzy sets by 
leveraging Ordered Weighted Averaging (OWA) aggregation operators. This model introduces a flexible and parameterizable mechanism to generalize the approximation operators
 using the importance of the values' order rather than their source.

---

## 2. Intuition

In classical fuzzy rough sets (ITFRS), the lower and upper approximations are defined using t-norms, implicators, and supremum/infimum operators. OWAFRS modifies this by replacing these crisp operators with OWA-based aggregators, 
making the model more robust and tunable in noisy or uncertain environments.

---

## 3. Definitions

Let:

  - U be the universe of discourse.
  - X be a fuzzy set in U.
  - R(x, y) be the fuzzy similarity (or tolerance) between elements x and y.
  - μ_X(y) be the membership degree of y in fuzzy set X.
  - I be a fuzzy implicator.
  - T be a t-norm.
  - OWA be a normalized weight vector of length n.


**Lower Approximation in OWAFRS**

The lower approximation of a fuzzy set X with respect to R and implicator I is:

\underline{R}_{OWA}(X)(x) = OWA_{desc} \big( \{ I(R(x, y), \mu_X(y)) \mid y ∈ U \} \big)


**Upper Approximation in OWAFRS**

The upper approximation is defined as:

\overline{R}_{OWA}(X)(x) = OWA_{desc} \big( \{ T(R(x, y), \mu_X(y)) \mid y ∈ U \} \big)

NOTE: **To see in which order OWA operator is multiplied, see the [OWA](owa_weights_info.md)**


**🔍 Why use OWA?**

OWA enables weighting the sorted values (e.g., top similarity scores or implication values), thus generalizing the traditional max/min operations. It balances strictness and flexibility in aggregation

---

## 3. Benefits of OWAFRS

  - **Flexibility**: Different OWA weight strategies (e.g., linear, exponential) offer customizable approximation behavior.
  - **Noise-resilience**: Softens the impact of outlier similarities or memberships.
  - **Unification**: OWAFRS generalizes other fuzzy rough set models (e.g., VQRS, ITFRS) as special cases depending on the OWA configuration.
---

### In OWAFRS
#### lower approximation for each instance: 
<img src="images/OWAFRS/lower.JPG" alt="lower aaproximation" width="250"/>

#### upper approximation for each instance:
<img src="images/OWAFRS/upper.JPG" alt="upper aproximation" width="250"/>

  - Since for the calculations of lower approximation, we calculate soft Inf which is basically a product, to exclude the same instance from calculations we set the main diagonal to 0.0

  - Since for the calculations of upper approximation, we calculate soft sup which is basically a product, to exclude the same instance from calculations we need to set the main diagonal to 0.0 which is ignored by max operator. Otherwise all upper approxamations will be 1.0.

#### In OWAFRS, POS(x) = ????????????????????????

