import numpy as np

from FRsutils.api import FuzzyRoughPositiveRegionScorer

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

scorer = FuzzyRoughPositiveRegionScorer(
    model="owafrs",
    similarity="linear",
)

scores = scorer.fit_score(X, y)
result = scorer.as_result()

print("POS scores: ", scores)
print("lower: ",result.lower)
print("upper: ",result.upper)