"""
@file test_frsmote_pipeline_gridsearch_compat.py
@brief Pipeline/GridSearchCV compatibility tests for the standalone FRSMOTE.

These tests were moved out of the FRsutils root test suite because FRSMOTE now
belongs to the standalone `fuzzy_rough_oversampling` package. They verify that
FRSMOTE still exposes flat sklearn-compatible parameters and can be used inside
an imbalanced-learn Pipeline with GridSearchCV.
"""

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from fuzzy_rough_oversampling import FRSMOTE


def test_frsmote_exposes_flat_params_for_pipeline():
    """@brief Verify that FRSMOTE exposes flat params for sklearn pipelines."""
    est = FRSMOTE()
    params = est.get_params(deep=True)

    assert "k_neighbors" in params
    assert "similarity" in params
    assert "similarity_sigma" in params
    assert "type" in params
    assert "ub_tnorm_name" in params
    assert "lb_implicator_name" in params


def test_frsmote_gridsearch_runs_small_dataset():
    """@brief Verify a small GridSearchCV run with FRSMOTE inside a pipeline."""
    rng = np.random.RandomState(0)
    X = rng.rand(40, 5).astype(float)
    y = np.array([0] * 30 + [1] * 10)

    pipe = ImbPipeline([
        ("frsmote", FRSMOTE()),
        ("svc", SVC(kernel="linear")),
    ])

    param_grid = {
        "frsmote__type": ["itfrs"],
        "frsmote__similarity": ["gaussian"],
        "frsmote__similarity_sigma": [0.2, 0.5],
        "frsmote__similarity_tnorm": ["minimum"],
        "frsmote__ub_tnorm_name": ["minimum"],
        "frsmote__lb_implicator_name": ["lukasiewicz"],
        "frsmote__k_neighbors": [3, 5],
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=cv, n_jobs=1, error_score="raise")
    gs.fit(X, y)

    assert gs.best_estimator_ is not None
