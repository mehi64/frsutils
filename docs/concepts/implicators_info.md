# Fuzzy Implicators in Fuzzy Logic

This document provides a comprehensive overview of common **fuzzy implicators** used in fuzzy logic systems, including formulas, aliases, and reference sources.

## 1. Fuzzy Implicator Overview and Properties

Fuzzy implicators are functions used to generalize the implication operation ("if-then") in fuzzy logic. A fuzzy implicator `I(a, b)` usually satisfies these properties:

* **Boundary conditions**: `I(0, 0) = 1`, `I(1, 1) = 1`, `I(1, 0) = 0`
* **Monotonicity**: `I(a, b)` is non-increasing in `a` and non-decreasing in `b`
* **Identity**: `I(1, b) = b`

---

## 2. Fuzzy Implicators Table

**Note:** frsutils implementations of implicators are based on the formulae in the references present in this table.

| Name              | Formula                                         | Alias Names        | Reference    |
| ----------------- | ----------------------------------------------- | ------------------ | ------------ |
| **Lukasiewicz**   | `I(a, b) = min(1, 1 - a + b)`                   | Bounded Implicator | [1], Eq. (1) |
| **Gödel**         | `I(a, b) = 1 if a <= b; b otherwise`            | Gödel Implicator   | [1], Eq. (2) |
| **Reichenbach**   | `I(a, b) = 1 - a + ab`                          | Reichenbach        | [1], Eq. (3) |
| **Kleene-Dienes** | `I(a, b) = max(1 - a, b)`                       | KD Implicator      | [1], Eq. (4) |
| **Goguen**        | `I(a, b) = 1 if a <= b; b / a otherwise`        | Goguen Implicator  | [1], Eq. (5) |
| **Yager**         | `I(a, b) = b^a if a > 0 or b > 0; 1 otherwise`  | Yager Implicator   | [1], Eq. (7) |
| **Rescher**       | `I(a, b) = 1 if a = 0 and b = 0; 0 otherwise`   | Rescher Implicator | [1], Eq. (6) |
| **Weber**         | `I(a, b) = b if a == 1 ; 1 if a < 1`            | Weber Implicator   | [1], Eq. (8) |
| **Fodor**         | `I(a, b) = max(1 - a, b) if a > b; 1 otherwise` | Fodor Implicator   | [1], Eq. (9) |

---

## 3. Notes

* Implicators are crucial for defining fuzzy rules and approximations in fuzzy-rough sets.
* Not all implicators are equally useful in all contexts — some, like *Reichenbach*, are probabilistic, while others like *Gödel/Goedel* are logical.
* Some implicators are not differentiable and may not be suitable for gradient-based learning methods.

---

## 4. References

1. **Raquel Fernandez-Peralta (2025)** - *A Comprehensive Survey of Fuzzy Implication Functions*.

------

### Some hints on Implicators

- Since in this library, A(y) in implicators--if instance x has the same class of instance y, is used, therefore different implicators boil down to the same thing:
  - *Goedel* and *Gaines* produce the same results
  - *KD*, *Reichenbach* and *Luk* also produce the same results
- implicators work on scalar but can be vectorized with np.vectorize()
- implicators do not generate the same values for these (0,0) , (1,0) , (0,1) , (1,1). Their behavior is different. Do not draw a general conclusion on them.
