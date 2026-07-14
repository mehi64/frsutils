# Fuzzy Quantifiers in Fuzzy Logic

This document provides a short overview of the **fuzzy quantifiers** implemented in `frsutils.core.fuzzy_quantifiers`.

## 1. Fuzzy Quantifier Overview

The implemented quantifiers map values from `[0, 1]` to `[0, 1]`. Both use parameters `alpha` and `beta` with:

`0 <= alpha < beta <= 1`

They are non-decreasing and satisfy `Q(0) = 0` and `Q(1) = 1`.

---

## 2. Fuzzy Quantifier Table

| Name          | Formula                                   | Parameters      | Aliases         | Reference     |
| ------------- | ----------------------------------------- | --------------- | --------------- | ------------- |
| **Linear**    | Piecewise-linear quantifier shown below   | `alpha`, `beta` | linear          | [1], Eq. (7)  |
| **Quadratic** | Quadratic S-shaped quantifier shown below | `alpha`, `beta` | quadratic, quad | [2], Eq. (12) |

The linear quantifier is:

\[
Q_{\alpha,\beta}(x) =
\begin{cases}
0, & x \leq \alpha \\
\dfrac{x-\alpha}{\beta-\alpha}, & \alpha < x < \beta \\
1, & x \geq \beta
\end{cases}
\]

The quadratic quantifier is:

\[
Q_{\alpha,\beta}(x) =
\begin{cases}
0, & x \leq \alpha \\
\dfrac{2(x-\alpha)^2}{(\beta-\alpha)^2}, & \alpha < x \leq \dfrac{\alpha+\beta}{2} \\
1 - \dfrac{2(x-\beta)^2}{(\beta-\alpha)^2}, & \dfrac{\alpha+\beta}{2} < x \leq \beta \\
1, & x > \beta
\end{cases}
\]

---

## 3. Notes

- Both implementations support NumPy arrays and backend-aware computation.
- The quantifiers are used by VQRS to transform interim approximation ratios.
- Input validation requires finite values in `[0, 1]` unless disabled explicitly.

---

## 4. References

1. **F. Nasirzadeh, M. Khanzadi, and H. Mianabadi (2013)** — *A Fuzzy Group Decision Making Approach to Construction Project Risk Management*. International Journal of Industrial Engineering & Production Research, 24(1), 71–80. The piecewise-linear fuzzy linguistic quantifier is given in Eq. (7). [Free PDF](https://www.researchgate.net/profile/Hojjat-Mianabadi-2/publication/274696434_A_Fuzzy_Group_Decision_Making_Approach_to_Construction_Project_Risk_Management/links/55253af50cf201667be66713/A-Fuzzy-Group-Decision-Making-Approach-to-Construction-Project-Risk-Management.pdf)
2. **R. Jensen and C. Cornelis (2011)** — *Fuzzy-Rough Nearest Neighbour Classification and Prediction*. Theoretical Computer Science, 412(42), 5871–5884. The quadratic fuzzy quantifier used in VQRS is given in Eq. (12). [Free PDF](https://pure.aber.ac.uk/ws/portalfiles/portal/170509/tcs.pdf)

---
