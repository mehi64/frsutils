# T-norms

A triangular norm (T-norm) is a binary operation
\(T:[0,1]^2\rightarrow[0,1]\) used to model fuzzy conjunction. A T-norm is
commutative, associative, monotone in both arguments, and satisfies
\(T(a,1)=a\).

`frsutils` uses T-norms both to aggregate feature-level similarities and in
fuzzy-rough upper approximations. All implementations accept scalar or
array-valued inputs through NumPy-compatible backends.

## Implemented T-norms

| Name | Formula | Parameters | Registered aliases |
| --- | --- | --- | --- |
| Minimum | \(\min(a,b)\) | None | `minimum`, `min`, `goedel`, `standardintersection` |
| Product | \(ab\) | None | `product`, `prod`, `algebraic` |
| Łukasiewicz | \(\max(0,a+b-1)\) | None | `lukasiewicz`, `luk`, `bounded`, `boundeddifference` |
| Drastic product | \(a\) if \(b=1\); \(b\) if \(a=1\); otherwise \(0\) | None | `drastic`, `drasticproduct` |
| Einstein product | \(ab/(2-a-b+ab)\) | None | `einstein`, `einsteinproduct` |
| Hamacher product | \(0\) when \(a=b=0\); otherwise \(ab/(a+b-ab)\) | None | `hamacher`, `hamacherproduct` |
| Nilpotent minimum | \(\min(a,b)\) if \(a+b>1\); otherwise \(0\) | None | `nilpotent`, `nilpotentminimum` |
| Yager | \(1-\min\{1,[(1-a)^p+(1-b)^p]^{1/p}\}\) | \(p>0\) | `yager`, `yg` |

The formulas above are the exact conventions implemented by the library. For
array reduction, associative T-norms are applied along the aggregation axis;
the Yager implementation uses the equivalent multi-argument expression.

## References

1. Klement, E. P., Mesiar, R., & Pap, E. (2000). *Triangular Norms*. Springer.
   <https://doi.org/10.1007/978-94-015-9540-7>
2. Radzikowska, A. M., & Kerre, E. E. (2002). A comparative study of fuzzy rough
   sets. *Fuzzy Sets and Systems*, 126(2), 137–155.
   <https://doi.org/10.1016/S0165-0114(01)00032-X>
