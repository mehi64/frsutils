# from sklearn.svm import SVC
# # from sklearn.pipeline import Pipeline
# from imblearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
# from FRsutils.core.models import * #needed for filling _class_registery. Because if not there cannot instantiate OWAFRS, VQRS, etc.
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

# # Load saved data and splits
# data = np.load("datasets/temp_datasets/frsmote_ds/normalized_data_with_splits.npz")
# X, y = data["X"], data["y"]
# splits = joblib.load("datasets/temp_datasets/frsmote_ds/cv_splits.pkl")

# frsmote = FRSMOTE()

# # Step 2: Define pipeline with custom FRSMOTE and SVC
# pipe = Pipeline([
#     ("frsmote", frsmote),
#     ("svc", SVC())
# ])

# # Step 3: Define hyperparameter grid
# param_grid = {
#     # FRSMOTE + ITFRS parameters
#     "frsmote__type": ["itfrs", "owafrs"],
#     "frsmote__similarity": ["gaussian", "linear"],
#     "frsmote__similarity_tnorm": ["minimum"],
#     "frsmote__gaussian_similarity_sigma": [0.1],
#     "frsmote__ub_tnorm_name": ["minimum"],
#     "frsmote__lb_implicator_name": ["lukasiewicz"],
#     "frsmote__ub_owa_method_name": ["linear"],
#     "frsmote__lb_owa_method_name": ["linear"],
#     "frsmote__k_neighbors":5,
#     "frsmote__bias_interpolation":False,
#     "frsmote__random_state":42,

#     # Optional: SVC parameters
#     "svc__C": [0.1],
#     "svc__kernel": ["rbf"]
# }

# scoring = {
#     "f1": make_scorer(f1_score),
#     "accuracy": make_scorer(accuracy_score),
#     "precision": make_scorer(precision_score),
#     "recall": make_scorer(recall_score)
# }


# # Step 4: Run grid search
# grid = GridSearchCV(estimator=pipe, 
#                     param_grid=param_grid, 
#                     cv=splits, 
#                     scoring=scoring,
#                     refit="f1", 
#                     n_jobs=-1, 
#                     verbose=2)
# grid.fit(X, y)

# # Extract all results as a DataFrame
# results_df = pd.DataFrame(grid.cv_results_)
# results_df.to_excel("temp/gridsearch_results.xlsx", index=False)

# # Step 5: Results
# print("Best Params:\n", grid.best_params_)
# print("Best F1 Score:", grid.best_score_)

# # joblib.dump(grid, "gridsearch_model.pkl")


from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.models import *  # Required to populate model registry
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score

# ✅ Load dataset and predefined splits
data = np.load("datasets/temp_datasets/frsmote_ds/normalized_data_with_splits.npz")
X, y = data["X"], data["y"]
splits = joblib.load("datasets/temp_datasets/frsmote_ds/cv_splits.pkl")

# ✅ Step 1: Create pipeline
pipe = Pipeline([
    ("frsmote", FRSMOTE()),  # uses default init, params set via grid
    ("svc", SVC())
])

# ✅ Step 2: Define hyperparameter grid
param_grid = {
    # FRSMOTE / fuzzy config
    "frsmote__type": ["itfrs", "owafrs"],
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
