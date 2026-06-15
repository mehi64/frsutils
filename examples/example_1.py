import numpy as np

from FRsutils.api import compute_approximations, compute_positive_region

# FRsutils expects numeric feature values on a comparable scale. In real
# experiments, normalize or scale your data before calling the fuzzy-rough API.
X = np.array(
    [
        [0.00, 0.10],
        [0.08, 0.18],
        [0.15, 0.12],
        [0.80, 0.82],
        [0.88, 0.90],
        [0.95, 0.86],
    ],
    dtype=float,
)
y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

result = compute_approximations(
    X,
    y,
    model="itfrs",
    similarity="linear",
)

print("lower approximation:", result.lower)
print("upper approximation:", result.upper)
print("boundary region:", result.boundary)
print("positive region:", result.positive_region)

# Shortcut when only positive-region scores are needed.
scores = compute_positive_region(
    X,
    y,
    model="itfrs",
    similarity="linear",
)
print("positive-region scores:", scores)