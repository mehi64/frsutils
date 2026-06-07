"""
@file frsmote.py
@brief FRSMOTE: fuzzy-rough positive-region guided SMOTE implementation.

FRSMOTE is an imbalanced-learn compatible oversampler. It exposes a flat
sklearn/GridSearchCV-friendly parameter interface while using FRsutils public API
internally to build similarity matrices and fuzzy-rough approximation models.

##############################################
# ✅ Quick Summary of Features
# Feature                              Description
# ----------------------------------------------------------------------------------
# FRSMOTE                              Positive-region guided SMOTE oversampler
# DEFAULT_CONFIG                       Defaults for ITFRS/OWAFRS/VQRS model params
# _prepare_minority_samples            Filters/ranks minority seeds by positive region
# _generate_new_samples                Interpolates synthetic samples between neighbors

# ✅ Design Patterns & Clean Code Notes
# - Template Method: extends BaseSoloFuzzyRoughOversampler hooks
# - Strategy Pattern: fuzzy-rough model/similarity behavior comes from FRsutils
# - Registry Pattern: registers itself as "frsmote" for build_oversampler
# - Adapter Pattern: flat estimator params are normalized internally
##############################################

##############################################
# ✅ How to Use - Examples
##############################################

# from fuzzy_rough_oversampling import FRSMOTE
#
# sampler = FRSMOTE(random_state=42)
# X_res, y_res = sampler.fit_resample(X, y)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from fuzzy_rough_oversampling._frsutils import get_fuzzy_rough_model_class
from fuzzy_rough_oversampling._sampling_utils import weighted_random_choice
from fuzzy_rough_oversampling.base import BaseSoloFuzzyRoughOversampler
from fuzzy_rough_oversampling.registry import register_oversampler


@register_oversampler(
    "frsmote",
    aliases=(
        "fr_smote",
        "fuzzy_rough_smote",
        "fuzzy-rough-smote",
    ),
)
class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    """
    @brief Fuzzy-rough positive-region guided SMOTE oversampler.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "type": "itfrs",
        "ub_tnorm_name": "minimum",
        "ub_tnorm_p": 2.0,
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
        "ub_owa_method_base": 2.0,
        "lb_owa_method_base": 2.0,
        "lb_fuzzy_quantifier_name": "linear",
        "ub_fuzzy_quantifier_name": "linear",
        "lb_fuzzy_quantifier_alpha": 0.1,
        "lb_fuzzy_quantifier_beta": 0.6,
        "ub_fuzzy_quantifier_alpha": 0.1,
        "ub_fuzzy_quantifier_beta": 0.6,
        "lb_alpha": 0.1,
        "lb_beta": 0.6,
        "ub_alpha": 0.1,
        "ub_beta": 0.6,
    }

    def __init__(self, **kwargs: Any) -> None:
        """
        @brief Initialize FRSMOTE with flat sklearn-compatible params.

        @param kwargs: Flat estimator/configuration parameters.
        """
        super().__init__(**kwargs)
        if kwargs:
            self.configure(**kwargs)

    def _check_params(self, **kwargs: Any) -> None:
        """
        @brief Validate FRSMOTE-specific and fuzzy-rough model params.

        @param kwargs: Flat estimator configuration.
        """
        cfg = dict(kwargs)

        k_neighbors = cfg.get("k_neighbors")
        if not isinstance(k_neighbors, int) or k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")

        bias_interpolation = cfg.get("bias_interpolation")
        if not isinstance(bias_interpolation, bool):
            raise TypeError("bias_interpolation must be a boolean.")

        random_state = cfg.get("random_state")
        if random_state is not None and not isinstance(
            random_state,
            (int, np.random.RandomState, np.random.Generator),
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

        get_fuzzy_rough_model_class(model_type)

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
        """
        @brief Generate synthetic minority samples and return resampled data.

        @param X: Feature matrix.
        @param y: Target labels.
        @return: Tuple (X_resampled, y_resampled).
        """
        rng = check_random_state(self.random_state)
        X_resampled, y_resampled = [X.copy()], [y.copy()]

        for label in self._get_target_classes():
            n_samples = self._get_num_samples(label)
            if n_samples <= 0:
                continue
            p1_candidates, nn_idx, _, selectable_idx = self._prepare_minority_samples(X, y, label)
            new_samples = self._generate_new_samples(
                X,
                n_samples,
                p1_candidates,
                selectable_idx,
                nn_idx,
                rng,
            )
            if new_samples:
                X_resampled.append(np.asarray(new_samples))
                y_resampled.append(np.full(len(new_samples), label, dtype=y.dtype))

        return np.vstack(X_resampled), np.hstack(y_resampled)

    def _prepare_minority_samples(self, X, y, class_label):
        """
        @brief Select minority seed candidates using positive-region scores.

        @param X: Feature matrix.
        @param y: Target labels.
        @param class_label: Minority class being oversampled.
        @return: Candidate weights, neighbor indices, distances, selectable indices.
        """
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
        """
        @brief Generate synthetic samples by neighbor interpolation.

        @param X: Feature matrix.
        @param n: Number of synthetic samples to generate.
        @param p1s: Weighted seed candidates.
        @param selectable_idx: Original indices available for interpolation.
        @param nn_idx: Neighbor indices in selectable_idx coordinates.
        @param rng: Random-state object.
        @return: List of synthetic samples.
        """
        out = []
        for _ in range(n):
            p1_idx, _ = weighted_random_choice(p1s, rng)
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
            if self.bias_interpolation:
                lam = p2_pos / max(p1_pos + p2_pos, 1e-9)
            else:
                lam = float(rng.rand())

            sample = X[p1_idx] + lam * (X[p2_idx] - X[p1_idx])
            out.append(sample)
        return out

    def fit_resample(self, X, y):
        """
        @brief Fit FRSMOTE and return resampled data.

        @param X: Feature matrix.
        @param y: Target labels.
        @return: Tuple (X_resampled, y_resampled).
        """
        self.fit(X, y)
        return self._fit_resample(X, y)

    def _finalize_object(self) -> None:
        """
        @brief Cache concrete estimator attributes after configuration/build.
        """
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


__all__ = ["FRSMOTE"]
