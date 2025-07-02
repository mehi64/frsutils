# T-Norms in Fuzzy Logic

This document provides a comprehensive overview of common **T-norms** used in fuzzy logic, including their formulas, alias names, theoretical properties, and scholarly references.

## 1. T-Norms Overview and Properties [wikipedia_1]

T-norms (Triangular norms) are binary operations used in fuzzy logic to model the intersection of fuzzy sets. They must satisfy the following properties:

* **Commutativity**: `T(a, b) = T(b, a)`
* **Associativity**: `T(a, T(b, c)) = T(T(a, b), c)`
* **Monotonicity**: If `a <= a'` and `b <= b'`, then `T(a, b) <= T(a', b')`
* **Boundary Condition**: `T(a, 1) = a`

Some T-norms also satisfy additional properties such as **nilpotency**, **strictness**, or **Archimedean** behavior.

---

## 2. T-Norms Table

| Name                  | Formula                                                 | Alias Names        | Reference Page                 |Tested and working in FRsutils|
| --------------------- | ------------------------------------------------------- | ------------------ | ----------------------------- |
| **Minimum**           | `T(a,b) = min(a,b)`                                     | Standard-intersection, Gödel    | IMEKO 2018, Eq. (4)           |yes|
| **Product**           | `T(a,b) = a * b`                                        | Algebraic, Product              | IMEKO 2018, Eq. (5)           |yes|
| **Lukasiewicz**       | `T(a,b) = max(0, a + b - 1)`                            | Bounded Difference              | IMEKO 2018, Eq. (6)           |yes|
| **Yager**             | `1 - min(1, ((1-a)^p + (1-b)^p)^(1/p))`   where p>0     | Yager                           | IMEKO 2018 (Yager T-Norm)     |yes|
| **Drastic Product**   | `T(a,b) = a if b == 1; b if a == 1; 0 otherwise`        | Drastic                         | wikipedia_1          |yes|
| **Einstein Product**  | `T(a,b) = ab / (2 - (a + b - ab))`                      | Einstein T-Norm                 | I. Silambarasan , S. Sriram         |yes|
| **Nilpotent Minimum** | `T(a,b) = min(a,b) if (a + b > 1) else 0`               | Nilpotent Min                   | Wikipedia_ref_1 |yes|
| **Hamacher Product** | `T(a,b) = 0 if a = b = 0, else ab / (a + b - ab)`        | Hamacher                        | Wikipedia_ref_1 |yes|


---


## 3. Notes

* Some t-norms are not suitable for numerical optimization due to non-differentiability (e.g., Minimum).
* There are some extensions to tnorms, like
  - Hamacher t-norms
  - Frank t-norms
  - Yager t-norms
  - Aczél–Alsina t-norms
  - Dombi t-norms
  - Sugeno–Weber t-norms
  
  
We have not implement yet. For those, see [[wikipedia_link]](https://en.wikipedia.org/wiki/Construction_of_t-norms#Hamacher_t-norms)

---

## 4. References

1. **I. Silambarasan , S. Sriram** - HAMACHER OPERATIONS ON PYTHAGOREAN FUZZY MATRICES
2. **IMEKO TC18 2018** - Claudio De Capua and Emilia Romeo, A Comparative Analysis of Fuzzy t-Norm Approaches to the Measurement Uncertainty Evaluation. See formulas (4), (5), (6), Yager.
3. **Adam Grabowski** - Basic Formal Properties of Triangular Norms and Conorms.
4. **Wikipedia_ref_1** - [link to page](https://en.wikipedia.org/wiki/T-norm#:~:text=of%20t%2Dnorms-,The%20drastic%20t%2Dnorm%20is%20the%20pointwise%20smallest%20t%2Dnorm,in%20%5B0%2C%201%5D)

---


