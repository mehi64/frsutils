"""
@file FRSMOTE.py
@brief FRSMOTE: Fuzzy Rough Set SMOTE implementation.
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
from FRsutils.core.preprocess.base_solo_fuzzy_rough_oversampler import BaseSoloFuzzyRoughOversampler
import FRsutils.utils.math_utils.math_utils as math_utils

class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    def _check_params(self):
        if not isinstance(self.k_neighbors, int) or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")
        if not isinstance(self.bias_interpolation, bool):
            raise ValueError("bias_interpolation must be boolean.")

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