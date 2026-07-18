# Fuzzy implicators

A fuzzy implicator generalizes Boolean implication to truth degrees in
\([0,1]\). In fuzzy-rough lower approximations, the first argument is normally a
relation value and the second argument is a class-membership value. Standard
implicators are non-increasing in the first argument, non-decreasing in the
second, and satisfy the classical corner conditions
\(I(0,0)=I(0,1)=I(1,1)=1\) and \(I(1,0)=0\).

## Implemented implicators

| Name | Formula | Registered aliases |
| --- | --- | --- |
| Łukasiewicz | \(\min(1,1-a+b)\) | `lukasiewicz`, `luk` |
| Gödel | \(1\) if \(a\le b\); otherwise \(b\) | `goedel` |
| Kleene–Dienes | \(\max(1-a,b)\) | `kleenedienes`, `kleene`, `kd` |
| Reichenbach | \(1-a+ab\) | `reichenbach` |
| Goguen | \(1\) if \(a\le b\); otherwise \(b/a\) | `goguen`, `product` |
| Rescher | \(1\) if \(a\le b\); otherwise \(0\) | `rescher` |
| Yager | \(b^a\), with \(I(0,0)=1\) | `yager` |
| Weber | \(b\) if \(a=1\); otherwise \(1\) | `weber` |
| Fodor | \(1\) if \(a\le b\); otherwise \(\max(1-a,b)\) | `fodor` |

The public implementations are vectorized and backend-aware; users do not need
`numpy.vectorize`.

## Binary class-membership note

For a crisp decision class, the consequent \(b\) is either zero or one. Under
that restriction:

- Łukasiewicz, Kleene–Dienes, and Reichenbach return the same values;
- Gödel and Goguen return the same values.

These equivalences do **not** hold for arbitrary fuzzy consequents. Other
implicators can still produce different lower approximations even with crisp
class labels.

## References

1. Baczyński, M., & Jayaram, B. (2008). *Fuzzy Implications*. Springer.
   <https://doi.org/10.1007/978-3-540-69082-5>
2. Radzikowska, A. M., & Kerre, E. E. (2002). A comparative study of fuzzy rough
   sets. *Fuzzy Sets and Systems*, 126(2), 137–155.
   <https://doi.org/10.1016/S0165-0114(01)00032-X>
