# OWA weighting strategies

An ordered weighted averaging (OWA) operator first sorts its input values and
then computes a weighted sum. For values sorted in descending order,
\(z_1\ge z_2\ge\cdots\ge z_n\), and normalized weights
\(w_i\ge0\) with \(\sum_i w_i=1\),

\[
\operatorname{OWA}(z)=\sum_{i=1}^{n} w_i z_i.
\]

In `frsutils`, OWAFRS removes the self-comparison, sorts the remaining evidence
in descending order, and applies:

- ascending weights for the lower approximation, emphasizing smaller evidence
  values and approximating an infimum;
- descending weights for the upper approximation, emphasizing larger evidence
  values and approximating a supremum.

## Implemented weight families

| Name | Raw weight family before normalization | Parameters | Registered aliases |
| --- | --- | --- | --- |
| Linear | \(r_i=i\) | None | `linear`, `additive` |
| Exponential | \(r_i\propto b^i\) | \(b>1\) | `exponential`, `exp`, `gp` |
| Harmonic | \(r_i=1/i\) | None | `harmonic`, `harm`, `inv_add` |

For linear weights in ascending order, normalization gives

\[
w_i=\frac{2i}{n(n+1)}.
\]

The `weights(n, order=...)` method normalizes the raw values and explicitly sorts
them into ascending or descending order. The exponential implementation shifts
its exponents before normalization to avoid overflow while preserving the same
normalized mathematical weights.

## Configuration example

```python
from frsutils import compute_approximations

result = compute_approximations(
    X,
    y,
    model="owafrs",
    ub_owa_method_name="linear",
    lb_owa_method_name="linear",
)
```

## References

1. Yager, R. R. (1988). On ordered weighted averaging aggregation operators in
   multicriteria decisionmaking. *IEEE Transactions on Systems, Man, and
   Cybernetics*, 18(1), 183–190. <https://doi.org/10.1109/21.87068>
2. Cornelis, C., Verbiest, N., & Jensen, R. (2010). Ordered weighted average
   based fuzzy rough sets. In *Rough Set and Knowledge Technology*, 78–85.
   <https://doi.org/10.1007/978-3-642-16248-0_16>
3. Vluymans, S., Mac Parthaláin, N., Cornelis, C., & Saeys, Y. (2019). Weight
   selection strategies for ordered weighted average based fuzzy rough sets.
   *Information Sciences*, 501, 155–171.
   <https://doi.org/10.1016/j.ins.2019.05.085>
