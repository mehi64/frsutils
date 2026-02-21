"""
@file FRSMOTE.py
@brief FRSMOTE: Fuzzy Rough Set SMOTE implementation.
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from FRsutils.core.preprocess.base_solo_fuzzy_rough_oversampler import BaseSoloFuzzyRoughOversampler
import FRsutils.utils.math_utils.math_utils as math_utils
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel

class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    def __init__(self, **kwargs):
        """
        @brief Initialize FRSMOTE.

        This initializer is intentionally lightweight. It does not build the internal
        fuzzy-rough model. Instead, it stores the clone-friendly configuration using
        `configure(...)`. Parameter validation is executed fail-fast during configure
        via the `_validate_config` hook chain.

        Important for GridSearchCV:
        - If instantiated with no parameters, the object stays UNCONFIGURED and can be
        configured later via `set_params(...)`.
        - If parameters are provided here, the object becomes CONFIGURED immediately.

        @param kwargs: Flat hyperparameter dictionary for both oversampling and fuzzy-rough model.
        """
        super().__init__(**kwargs)

        # Only configure if user provided any parameters.
        if kwargs:
            self.configure(model_registry=FuzzyRoughModel, **kwargs)



    def _check_params(self, **kwargs):
    #     """
    #     @brief Validate FRSMOTE configuration (clone-friendly).

    #     @param kwargs: Flat configuration dictionary.
    #     @raises ValueError/TypeError: If invalid.
    #     """
    #     # Remove registry if passed through validation hook
    #     kwargs = dict(kwargs)
    #     kwargs.pop("model_registry", None)

    #     # Allow being called with no kwargs (e.g., older call sites).
    #     if not kwargs and hasattr(self, "_object_config"):
    #         kwargs = dict(self._object_config)

    #     # ---- FRSMOTE core parameters ----
    #     k_neighbors = kwargs.get("k_neighbors")
    #     if k_neighbors is None or not isinstance(k_neighbors, int) or k_neighbors <= 0:
    #         raise ValueError("k_neighbors must be provided and be a positive integer.")

    #     bias_interpolation = kwargs.get("bias_interpolation")
    #     if bias_interpolation is None or not isinstance(bias_interpolation, bool):
    #         raise ValueError("bias_interpolation must be provided and be a boolean.")

    #     sampling_strategy = kwargs.get("sampling_strategy", "auto")
    #     if not (
    #         isinstance(sampling_strategy, str)
    #         or isinstance(sampling_strategy, dict)
    #         or isinstance(sampling_strategy, float)
    #         or isinstance(sampling_strategy, int)
    #         or callable(sampling_strategy)
    #     ):
    #         raise TypeError("sampling_strategy must be a str, dict, number, or callable.")

    #     instance_ranking_strategy = kwargs.get("instance_ranking_strategy", "pos")
    #     if not (isinstance(instance_ranking_strategy, str) or isinstance(instance_ranking_strategy, dict)):
    #         raise TypeError("instance_ranking_strategy must be a string or a dict.")

    #     if isinstance(instance_ranking_strategy, str) and instance_ranking_strategy not in {"pos", "lower", "upper"}:
    #         raise ValueError("instance_ranking_strategy must be one of {'pos','lower','upper'} or a dict.")

    #     sampling_ratio = kwargs.get("sampling_ratio", None)
    #     if sampling_ratio is not None and not (isinstance(sampling_ratio, (int, float)) or isinstance(sampling_ratio, dict)):
    #         raise TypeError("sampling_ratio must be None, a number, or a dict.")

    #     random_state = kwargs.get("random_state", None)
    #     if random_state is not None and not isinstance(random_state, (int, np.random.RandomState)):
    #         raise TypeError("random_state must be None, an int, or a numpy.random.RandomState.")

    #     # ---- Fuzzy-rough model minimal config checks (by type) ----
    #     model_type = kwargs.get("type", None)
    #     if model_type is None:
    #         return

    #     model_type = str(model_type).lower().strip()

    #     if model_type == "itfrs":
    #         if kwargs.get("ub_tnorm") is None and not kwargs.get("ub_tnorm_name"):
    #             raise ValueError("ITFRS requires 'ub_tnorm' or 'ub_tnorm_name'.")
    #         if kwargs.get("lb_implicator") is None and not kwargs.get("lb_implicator_name"):
    #             raise ValueError("ITFRS requires 'lb_implicator' or 'lb_implicator_name'.")

    #     elif model_type == "owafrs":
    #         if kwargs.get("ub_tnorm") is None and not kwargs.get("ub_tnorm_name"):
    #             raise ValueError("OWAFRS requires 'ub_tnorm' or 'ub_tnorm_name'.")
    #         if kwargs.get("lb_implicator") is None and not kwargs.get("lb_implicator_name"):
    #             raise ValueError("OWAFRS requires 'lb_implicator' or 'lb_implicator_name'.")
    #         if kwargs.get("lb_owa_method") is None and not kwargs.get("lb_owa_method_name"):
    #             raise ValueError("OWAFRS requires 'lb_owa_method' or 'lb_owa_method_name'.")
    #         if kwargs.get("ub_owa_method") is None and not kwargs.get("ub_owa_method_name"):
    #             raise ValueError("OWAFRS requires 'ub_owa_method' or 'ub_owa_method_name'.")

    #     elif model_type == "vqrs":
    #         if kwargs.get("lb_fuzzy_quantifier") is None and not kwargs.get("lb_fuzzy_quantifier_name"):
    #             raise ValueError("VQRS requires 'lb_fuzzy_quantifier' or 'lb_fuzzy_quantifier_name'.")
    #         if kwargs.get("ub_fuzzy_quantifier") is None and not kwargs.get("ub_fuzzy_quantifier_name"):
    #             raise ValueError("VQRS requires 'ub_fuzzy_quantifier' or 'ub_fuzzy_quantifier_name'.")
        pass

    def _fit_resample(self, X, y):
        rng = check_random_state(self.random_state)
        X_resampled, y_resampled = [X.copy()], [y.copy()]

        for label in self._get_target_classes():
            n = self._get_num_samples(label)
            if n == 0: continue
            p1s, nn_idx, _, selectable_idx = self._prepare_minority_samples(X, y, label)
            new_samples = self._generate_new_samples(X, n, p1s, selectable_idx, nn_idx, rng)
            if new_samples:
                X_resampled.append(np.array(new_samples))
                y_resampled.append(np.full(len(new_samples), label, dtype=y.dtype))

        return np.vstack(X_resampled), np.hstack(y_resampled)

    def _prepare_minority_samples(self, X, y, class_label):
        indices = np.where(y == class_label)[0]
        pos = self.positive_region[indices]

        if len(indices) <= 1:
            raise ValueError(f"Too few samples for {class_label}")

        mask = pos > 0
        selectable = indices[mask]
        pos_vals = pos[mask]

        if len(selectable) < 2:
            if len(indices) >= 2:
                selectable = indices
                pos_vals = pos
            else:
                raise ValueError("Insufficient data with POS > 0")

        X_sel = X[selectable]
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(selectable))).fit(X_sel)
        dists, idxs = nn.kneighbors(X_sel)

        p1_candidates = list(zip(selectable, pos_vals))
        return p1_candidates, idxs, dists, selectable

    def _generate_new_samples(self, X, n, p1s, selectable_idx, nn_idx, rng):
        out = []
        for _ in range(n):
            print("FRSMOTE_processing " + str(_) + "/" + str(n))
            p1_idx, _ = math_utils._weighted_random_choice(p1s, rng)
            loc = np.where(selectable_idx == p1_idx)[0]
            if len(loc) == 0: continue
            loc = loc[0]

            neighbors = nn_idx[loc][1:]
            if len(neighbors) == 0: continue
            p2_idx = selectable_idx[rng.choice(neighbors)]

            p1_pos = self.positive_region[p1_idx]
            p2_pos = self.positive_region[p2_idx]
            lam = p2_pos / max(p1_pos + p2_pos, 1e-9) if self.bias_interpolation else rng.rand()

            sample = X[p1_idx] + lam * (X[p2_idx] - X[p1_idx])
            out.append(sample)
        return out

    def fit_resample(self, X, y):
        """Resample the dataset."""
        self.fit(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)

        return X_resampled, y_resampled
    
    def _build_from_config(self, **config):
        pass

    def _finalize_object(self):
        setattr(self, 'k_neighbors', self._object_config['k_neighbors'])
        setattr(self, 'bias_interpolation', self._object_config['bias_interpolation'])
        setattr(self, 'random_state', self._object_config['random_state'])
        setattr(self, 'sampling_strategy', self._object_config['sampling_strategy'])
        setattr(self, 'instance_ranking_strategy', self._object_config['instance_ranking_strategy'])
        setattr(self, 'sampling_ratio', self._object_config['sampling_ratio'])

       