# SPDX-License-Identifier: BSD-3-Clause
"""Exploratory script for running FRSMOTE grid-search experiments.

This module is an exploratory usage script and is not part of the stable public API.
"""

from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from frsampling import FRSMOTE
from FRsutils.core.models import *  # Required to populate model registry
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

# ✅ Load dataset and predefined splits
data = np.load("datasets/temp_datasets/frsmote_ds/normalized_data_with_splits.npz")
X, y = data["X"], data["y"]
splits = joblib.load("datasets/temp_datasets/frsmote_ds/cv_splits.pkl")

# frsmote_obj = FRSMOTE(sampling_strategy='auto',
#                       instance_ranking_strategy = 'pos',
#                       k_neighbors = 5,
#                       bias_interpolation = True,
#                       random_state = 42,
#                       type = 'itfrs',
#                       similarity = "gaussian",
#                       similarity_tnorm = "minimum",
#                       gaussian_similarity_sigma = 0.1,
#                       ub_tnorm_name = "minimum",
#                       lb_implicator_name="lukasiewicz",
#                     #   ub_owa_method_name = "linear",
#                     #   lb_owa_method_name="linear",
#                       )

# frsmote_obj = FRSMOTE()

# ✅ Step 1: Create pipeline
pipe = Pipeline([
    ("frsmote", FRSMOTE()),  # uses default init, params set via grid
    ("svc", SVC())
])

# ✅ Step 2: Define hyperparameter grid
param_grid = {
    # FRSMOTE / fuzzy config
    "frsmote__type": ["owafrs"],
    "frsmote__similarity": ["gaussian", "linear"],
    "frsmote__similarity_tnorm": ["minimum"],
    "frsmote__gaussian_similarity_sigma": [0.1],
    "frsmote__ub_tnorm_name": ["minimum"],
    "frsmote__lb_implicator_name": ["lukasiewicz"],
    "frsmote__ub_owa_method_name": ["linear"],
    "frsmote__lb_owa_method_name": ["linear"],
    "frsmote__k_neighbors": [5],
    "frsmote__bias_interpolation": [False],
    "frsmote__random_state": [42],
    "frsmote__sampling_strategy": ["auto"],
    "frsmote__sampling_ratio": [1.0],
    "frsmote__instance_ranking_strategy": ["pos"],

    # SVC parameters
    "svc__C": [0.1],
    "svc__kernel": ["rbf"]
}

# ✅ Step 3: Define scoring dictionary
scoring = {
    "f1": make_scorer(f1_score),
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score)
}

# ✅ Step 4: Run GridSearchCV
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=splits,
    scoring=scoring,
    refit="f1",
    n_jobs=-1,
    verbose=2
)

grid.fit(X, y)

# ✅ Step 5: Export results
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_excel("temp/gridsearch_results.xlsx", index=False)

print("Best Params:\n", grid.best_params_)
print("Best F1 Score:", grid.best_score_)

# Optional model export
# joblib.dump(grid, "gridsearch_model.pkl")
