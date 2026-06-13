# SPDX-License-Identifier: BSD-3-Clause
"""Exploratory example script for FRSMOTE usage.

This module is an exploratory usage script and is not part of the stable public API.
"""

import numpy as np
from frsampling import FRSMOTE
# from FRsutils.core.models.itfrs import ITFRS
# from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models import *
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import Counter

# Example imbalanced dataset
X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.15, 0.18],  # Minority class
    [0.25, 0.12],  # Minority class
    [0.19, 0.42],  # Minority class
    [0.8, 0.9],
    [0.82, 0.88],
    [0.85, 0.91],
    [0.87, 0.93]
])
y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0])  # Class 1 is minority

smote1 = FRSMOTE()

config = {
    'type': 'owafrs',
    'similarity_name': 'gaussian',
    'gaussian_similarity_sigma': 0.5,
    'similarity_tnorm_name': 'minimum',
    'lb_implicator_name': 'lukasiewicz',
    'ub_tnorm_name': 'product',
    'ub_owa_method_name': "linear",
    'lb_owa_method_name': "linear",
    'k_neighbors': 3,
    'random_state': None,
    'sampling_strategy' : "auto",
    'instance_ranking_strategy' : 'pos',
    'sampling_ratio' : {'1': 0.6, '2': 0.8},
    'bias_interpolation' : False

}

smote1.configure(model_registry=FuzzyRoughModel, **config)
X_res_smote, y_res_smote = smote1.fit_resample(X, y)

smote2 = FRSMOTE(**config)
X_res_smote, y_res_smote = smote2.fit_resample(X, y)

print(X_res_smote)
print(y_res_smote)


# --- Plotting ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0 (Maj)", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1 (Min)", alpha=0.6)
plt.title("Original Data")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_res_smote[y_res_smote == 0][:, 0], X_res_smote[y_res_smote == 0][:, 1], label="Class 0", alpha=0.6)
plt.scatter(X_res_smote[y_res_smote == 1][:, 0], X_res_smote[y_res_smote == 1][:, 1], label="Class 1", alpha=0.6)
# Highlight synthetic points (crude way)
n_original_minority = Counter(y)[1]
n_synthetic = Counter(y_res_smote)[1] - n_original_minority
if n_synthetic > 0:
        synthetic_X = X_res_smote[y_res_smote==1][-n_synthetic:]
        plt.scatter(synthetic_X[:, 0], synthetic_X[:, 1], label="Synthetic", marker='x', c='red', s=50)
plt.title("After FRSMOTE")
plt.legend()

plt.tight_layout()
plt.show()
