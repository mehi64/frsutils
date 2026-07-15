import numpy as np

from frsutils import compute_approximations, compute_positive_region

# frsutils expects numeric feature values on a comparable scale. In real
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

result_owafrs = compute_approximations(
    X,
    y,
    model="owafrs",
    similarity="linear",
    ub_tnorm_name="minimum",
    lb_implicator_name="lukasiewicz",
    ub_owa_method_name="exponential",
    ub_owa_method_base=2.5,
    lb_owa_method_name="harmonic",
)

print("owafrs lower approximation:", result_owafrs.lower)
print("owafrs upper approximation:", result_owafrs.upper)
print("owafrs boundary region:", result_owafrs.boundary)
print("owafrs positive region:", result_owafrs.positive_region)

# Shortcut when only positive-region scores are needed.
# scores = compute_positive_region(
#     X,
#     y,
#     model="itfrs",
#     similarity="linear",
# )
# print("positive-region scores:", scores)