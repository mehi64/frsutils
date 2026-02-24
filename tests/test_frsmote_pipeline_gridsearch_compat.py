import numpy as np

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from FRsutils.core.preprocess.FRSMOTE import FRSMOTE


def test_frsmote_exposes_flat_params_for_pipeline():
    est = FRSMOTE()
    params = est.get_params(deep=True)

    # core knobs
    assert "k_neighbors" in params
    assert "similarity" in params
    assert "similarity_sigma" in params

    # model knobs
    assert "type" in params
    assert "ub_tnorm_name" in params
    assert "lb_implicator_name" in params


def test_frsmote_gridsearch_runs_small_dataset():
    rng = np.random.RandomState(0)
    X = rng.rand(40, 5).astype(float)  # normalized to [0,1]
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
