# Fuzzy Similarities in Fuzzy Logic

This document provides a structured overview of fuzzy similarity functions as implemented in the FRsutils framework. These functions help quantify the degree of resemblance between instances in normalized feature spaces.

## 1. Similarity Functions Overview and Properties

Fuzzy similarity functions are used to evaluate how similar two fuzzy values or vectors are. In the FRsutils framework, a similarity function `sim(x, y)` is used together with a T-norm `T(a, b)` to aggregate similarities feature-wise.

**Expected properties of similarity functions**:

* **Symmetry**: `sim(x, y) = sim(y, x)`
* **Boundedness**: `sim(x, y) ∈ [0, 1]`
* **Maximum at equality**: `sim(x, x) = 1`

---

## 2. Fuzzy Similarity Functions Table

| Name         | Formula                                 | Alias Names     | Reference Page |
| ------------ | ----------------------------------------------------  | --------------- | -------------- |
| **Linear**   | `sim = max(0, 1 - abs(x - y))`          | linear           | [1]|
| **Gaussian** | `sim = exp(-(x - y)^2/(2 * sigma^2))`       | gaussian, gauss | [1]  |

---

## 3. Notes

* Each similarity function supports pairwise vector diff input.
* Can be used with any T-norm for constructing a full similarity matrix.
* More functions (e.g., cosine, Jaccard, Tversky) are marked for future addition.

---

## 4. References

1. **FRsutils Core Source** — See implementation in `similarities.py`.

---

## 5. Example Usage

```python
from FRsutils.core.similarities import build_similarity_matrix

X = np.array([[0.1, 0.2], [0.2, 0.1]])

similarity_matrix = build_similarity_matrix(
    X,
    similarity="gaussian",
    sigma=0.5,
    similarity_tnorm="minimum"
)
print(similarity_matrix)
```

