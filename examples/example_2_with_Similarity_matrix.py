import numpy as np

from frsutils import compute_approximations

similarity_mat = np.array([
            [1.00, 0.54, 0.37, 0.19, 0.10],
            [0.54, 1.00, 0.35, 0.29, 0.20],
            [0.37, 0.35, 1.00, 0.55, 0.73],
            [0.19, 0.29, 0.55, 1.00, 0.74],
            [0.10, 0.20, 0.73, 0.74, 1.00]
        ])

y = np.array([1, 1, 0, 1, 0], dtype=int)

result = compute_approximations(
    X=None,
    y=y,
    similarity_matrix=similarity_mat,
    model="itfrs",
)

print("lower approximation:", result.lower)
print("upper approximation:", result.upper)
print("boundary region:", result.boundary)
print("positive region:", result.positive_region)

