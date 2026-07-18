# Similarity functions

`frsutils` constructs fuzzy relations by comparing samples feature by feature
and aggregating the resulting similarities with a T-norm. A feature-level
similarity maps two values to the unit interval, where larger values represent
greater similarity.

## Implemented functions

For feature values \(x\) and \(y\), let \(d = x-y\).

| Name | Formula | Parameters | Registered aliases |
| --- | --- | --- | --- |
| Linear | \(\max(0, 1-|d|)\) | None | `linear` |
| Gaussian | \(\exp[-d^2/(2\sigma^2)]\) | \(\sigma>0\) | `gaussian`, `gauss` |

The linear function is most interpretable when feature differences are on a
meaningful scale, commonly after normalization. The Gaussian function provides
a smooth similarity whose decay is controlled by `sigma`.

For a data matrix with multiple features, `build_similarity_matrix` computes a
similarity matrix for each feature and combines the feature-level values using
the selected similarity T-norm. The diagonal is set to one.

## Public API example

```python
import numpy as np

from frsutils import build_similarity_matrix

X = np.array([[0.1, 0.2], [0.2, 0.1]], dtype=float)

similarity_matrix = build_similarity_matrix(
    X,
    similarity="gaussian",
    similarity_sigma=0.5,
    similarity_tnorm="minimum",
)
```

Use package-root imports in user code. The classes in `frsutils.core` are
implementation details and may change independently of the public API.

## References

1. Dubois, D., & Prade, H. (1990). Rough fuzzy sets and fuzzy rough sets.
   *International Journal of General Systems*, 17(2–3), 191–209.
   <https://doi.org/10.1080/03081079008935107>
2. Radzikowska, A. M., & Kerre, E. E. (2002). A comparative study of fuzzy rough
   sets. *Fuzzy Sets and Systems*, 126(2), 137–155.
   <https://doi.org/10.1016/S0165-0114(01)00032-X>
