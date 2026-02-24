"""
@file FRSMOTE.py
@brief FRSMOTE: Fuzzy Rough Set SMOTE implementation.

FRSMOTE is implemented as an imbalanced-learn compatible oversampler.
The estimator exposes a **flat parameter interface** (for sklearn / GridSearchCV),
while internally converting the flat params to a **nested config** for clean
component construction (similarity, similarity t-norm, fuzzy-rough model parts).

##############################################
# ✅ Quick Summary of Features
# - sklearn/imbalanced-learn compatible (fit_resample)
# - fuzzy-rough guided seed selection using positive region
# - flat params for GridSearchCV
# - internal nested config via normalize_flat_config_to_nested
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from imblearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
#
# pipe = Pipeline([
#   ("frsmote", FRSMOTE()),
#   ("svc", SVC())
# ])
#
# param_grid = {
#   "frsmote__similarity": ["gaussian"],
#   "frsmote__similarity_sigma": [0.2, 0.5],
#   "frsmote__k_neighbors": [3, 5],
# }
# gs = GridSearchCV(pipe, param_grid=param_grid, cv=3)
# gs.fit(X, y)
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

import FRsutils.utils.math_utils.math_utils as math_utils
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from FRsutils.core.preprocess.base_solo_fuzzy_rough_oversampler import BaseSoloFuzzyRoughOversampler

class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    DEFAULT_CONFIG: Dict[str, Any] = {
        "type": "itfrs",
        "ub_tnorm_name": "minimum",
        # Optional parameter for ub_tnorm_name='yager'
        "ub_tnorm_p": 2.0,
        "lb_implicator_name": "lukasiewicz",

        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
        # Optional parameter for *_owa_method_name='exponential'
        "ub_owa_method_base": 2.0,
        "lb_owa_method_base": 2.0,

        "lb_fuzzy_quantifier_name": "linear",
        "ub_fuzzy_quantifier_name": "linear",

        # New naming standard (preferred): <prefix>_fuzzy_quantifier_<param>
        "lb_fuzzy_quantifier_alpha": 0.1,
        "lb_fuzzy_quantifier_beta": 0.6,
        "ub_fuzzy_quantifier_alpha": 0.1,
        "ub_fuzzy_quantifier_beta": 0.6,

        # Legacy aliases (kept for backwards compatibility with old scripts)
        "lb_alpha": 0.1,
        "lb_beta": 0.6,
        "ub_alpha": 0.1,
        "ub_beta": 0.6,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if kwargs:
            self.configure(model_registry=FuzzyRoughModel, **kwargs)

    def _check_params(self, **kwargs):
        cfg = dict(kwargs)
        model_registry = cfg.pop("model_registry", None)

        k_neighbors = cfg.get("k_neighbors")
        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")

        bias_interpolation = cfg.get("bias_interpolation")
        if not isinstance(bias_interpolation, bool):
            raise TypeError("bias_interpolation must be a boolean.")

        random_state = cfg.get("random_state")
        if random_state is not None and not isinstance(
            random_state, (int, np.random.RandomState, np.random.Generator)
        ):
            raise TypeError("random_state must be None, an int, RandomState, or Generator.")

        if not isinstance(cfg.get("similarity"), str) or not cfg.get("similarity"):
            raise TypeError("similarity must be a non-empty string.")
        if not isinstance(cfg.get("similarity_tnorm"), str) or not cfg.get("similarity_tnorm"):
            raise TypeError("similarity_tnorm must be a non-empty string.")

        model_type = cfg.get("type")
        if not isinstance(model_type, str) or not model_type.strip():
            raise TypeError("type must be a non-empty string.")
        model_type = model_type.lower().strip()

        if model_registry is not None:
            model_registry.get_class(model_type)

        if model_type == "itfrs":
            if cfg.get("ub_tnorm") is None and not cfg.get("ub_tnorm_name"):
                raise ValueError("ITFRS requires 'ub_tnorm' or 'ub_tnorm_name'.")
            if cfg.get("lb_implicator") is None and not cfg.get("lb_implicator_name"):
                raise ValueError("ITFRS requires 'lb_implicator' or 'lb_implicator_name'.")
        elif model_type == "owafrs":
            if cfg.get("ub_tnorm") is None and not cfg.get("ub_tnorm_name"):
                raise ValueError("OWAFRS requires 'ub_tnorm' or 'ub_tnorm_name'.")
            if cfg.get("lb_implicator") is None and not cfg.get("lb_implicator_name"):
                raise ValueError("OWAFRS requires 'lb_implicator' or 'lb_implicator_name'.")
            if cfg.get("lb_owa_method") is None and not cfg.get("lb_owa_method_name"):
                raise ValueError("OWAFRS requires 'lb_owa_method' or 'lb_owa_method_name'.")
            if cfg.get("ub_owa_method") is None and not cfg.get("ub_owa_method_name"):
                raise ValueError("OWAFRS requires 'ub_owa_method' or 'ub_owa_method_name'.")
        elif model_type == "vqrs":
            if cfg.get("lb_fuzzy_quantifier") is None and not cfg.get("lb_fuzzy_quantifier_name"):
                raise ValueError("VQRS requires 'lb_fuzzy_quantifier' or 'lb_fuzzy_quantifier_name'.")
            if cfg.get("ub_fuzzy_quantifier") is None and not cfg.get("ub_fuzzy_quantifier_name"):
                raise ValueError("VQRS requires 'ub_fuzzy_quantifier' or 'ub_fuzzy_quantifier_name'.")
        else:
            raise ValueError(f"Unknown fuzzy-rough model type: {model_type}")

    def _fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X_resampled, y_resampled = [X.copy()], [y.copy()]

        for label in self._get_target_classes():
            n = self._get_num_samples(label)
            if n <= 0:
                continue
            p1s, nn_idx, _, selectable_idx = self._prepare_minority_samples(X, y, label)
            new_samples = self._generate_new_samples(X, n, p1s, selectable_idx, nn_idx, rng)
            if new_samples:
                X_resampled.append(np.asarray(new_samples))
                y_resampled.append(np.full(len(new_samples), label, dtype=y.dtype))

        return np.vstack(X_resampled), np.hstack(y_resampled)

    def _prepare_minority_samples(self, X, y, class_label):
        indices = np.where(y == class_label)[0]
        pos = self.positive_region[indices]
        if len(indices) <= 1:
            raise ValueError(f"Too few samples for class {class_label}")

        mask = pos > 0
        selectable = indices[mask]
        pos_vals = pos[mask]

        if len(selectable) < 2:
            selectable = indices
            pos_vals = pos

        X_sel = X[selectable]
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(selectable))).fit(X_sel)
        dists, idxs = nn.kneighbors(X_sel)

        p1_candidates = list(zip(selectable, pos_vals))
        return p1_candidates, idxs, dists, selectable

    def _generate_new_samples(self, X, n, p1s, selectable_idx, nn_idx, rng):
        out = []
        for _ in range(n):
            p1_idx, _ = math_utils._weighted_random_choice(p1s, rng)
            loc = np.where(selectable_idx == p1_idx)[0]
            if len(loc) == 0:
                continue
            loc = int(loc[0])

            neighbors = nn_idx[loc][1:]
            if len(neighbors) == 0:
                continue
            p2_idx = selectable_idx[rng.choice(neighbors)]

            p1_pos = float(self.positive_region[p1_idx])
            p2_pos = float(self.positive_region[p2_idx])
            lam = p2_pos / max(p1_pos + p2_pos, 1e-9) if self.bias_interpolation else float(rng.rand())

            sample = X[p1_idx] + lam * (X[p2_idx] - X[p1_idx])
            out.append(sample)
        return out

    def fit_resample(self, X, y):
        self.fit(X, y)
        return self._fit_resample(X, y)

    def _finalize_object(self):
        cfg = dict(getattr(self, "_object_config", {}))
        defaults = self._collect_default_config()

        def _get(key: str):
            return cfg.get(key, defaults.get(key))

        self.k_neighbors = _get("k_neighbors")
        self.bias_interpolation = _get("bias_interpolation")
        self.random_state = _get("random_state")

        self.sampling_strategy = _get("sampling_strategy")
        self.instance_ranking_strategy = _get("instance_ranking_strategy")
        self.sampling_ratio = _get("sampling_ratio")
        self.type = _get("type")