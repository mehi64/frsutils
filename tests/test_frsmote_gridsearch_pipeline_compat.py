import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE

def test_frsmote_params_visible_for_pipeline():
    s = FRSMOTE()
    p = s.get_params()
    assert "k_neighbors" in p
    assert "ub_tnorm_name" in p
    assert "lb_implicator_name" in p

def test_frsmote_gridsearch_runs_small():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 4).astype(float)  # in [0,1]
    y = np.array([0] * 24 + [1] * 6)

    pipe = ImbPipeline([
        ("frsmote", FRSMOTE()),
        ("svc", SVC(kernel="linear"))
    ])

    param_grid = {
        "frsmote__k_neighbors": [3, 5],
        "frsmote__type": ["itfrs"],
        "frsmote__ub_tnorm_name": ["minimum"],
        "frsmote__lb_implicator_name": ["lukasiewicz"],
    }

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    gs = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=1, error_score="raise")
    gs.fit(X, y)

    assert gs.best_estimator_ is not None