import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
import warnings
import FRsutils.utils.math_utils.math_utils as math_utils
import FRsutils.core.preprocess.Base_solo_FR_resampler as bfrrs
from FRsutils.core.approximations import FuzzyRoughModel_Base

# --- FRSMOTE Implementation ---

class FRSMOTE(bfrrs.BaseSoloFuzzyRoughResampler):
    """
    Fuzzy Rough Set based SMOTE (FRSMOTE) Oversampler.
    """
    def __init__(self,
                 fr_model : FuzzyRoughModel_Base,
                 k_neighbors=5,
                 sampling_strategy = 'auto',
                 bias_interpolation=False,
                 random_state=None):
        """
        NOTE: correctness of data will be checked in fit() function
        fr_model: fuzzy rough model e.g ITFR, VQRS, OWAFRS
        bias_interpolation: find lambda in best way or random
        """
        super().__init__(fr_model=fr_model, 
                         sampling_strategy=sampling_strategy)
    
        self.k_neighbors = k_neighbors
        self.bias_interpolation = bias_interpolation
        self.random_state = random_state

    def _check_params(self):
        """
        @brief Checks correctness of input parameters of FRSMOTE

        @throws ValueError If k_neighbors is not a positive integer
        @throws ValueError If bias_interpolation is not boolean
        """
        super()._check_params()

        if not isinstance(self.k_neighbors, int) or self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be a positive integer.")
        if not isinstance(self.bias_interpolation, bool):
             raise ValueError("bias_interpolation must be a boolean.")
        
    def _fit_resample(self, X, y):
        """Perform FRSMOTE oversampling"""
        random_state = check_random_state(self.random_state)
        X_resampled_list = [X.copy()]
        y_resampled_list = [y.copy()]

        target_classes = self._get_target_classes()
        for class_label in target_classes:
            num_samples_to_generate = self._get_num_samples(class_label)
            if num_samples_to_generate == 0:
                continue
            
            # p1 selection
            p1_candidates, \
            nn_indices_in_selectable, \
            distances,\
            indices_selectable_orig = self._prepare_minority_samples(X, y, class_label)

            # generate samples
            new_samples = self._generate_new_samples(X,
                              num_samples_to_generate,
                              p1_candidates,
                              indices_selectable_orig,
                              nn_indices_in_selectable,
                              random_state)
            
            if new_samples:
                X_resampled_list.append(np.array(new_samples))
                y_resampled_list.append(np.full(len(new_samples), class_label, dtype=y.dtype))

        return np.vstack(X_resampled_list), np.hstack(y_resampled_list)

    def _prepare_minority_samples(self, X, y, class_label):
        minority_indices = np.where(y == class_label)[0]
        X_minority_norm = X[minority_indices]
        pos_minority = self.positive_region[minority_indices]

        if len(minority_indices) <= 1:
            # warnings.warn(f"Cannot perform SMOTE for class {class_label} with <= 1 sample.")
            raise ValueError(f"Cannot perform SMOTE for class {class_label} with <= 1 sample.")

        # Filter selectable points (POS > 0) - Use original indices
        if np.any(pos_minority < 0.0):
            raise ValueError(f"positive region membership value of some data instances for class {class_label} have negative values in {self.__class__.__name__}.")

        selectable_mask = pos_minority > 0
        selectable_indices = minority_indices[selectable_mask]
        pos_selectable = pos_minority[selectable_mask]
        
        # Need normalized data corresponding to selectable points for KNN
        X_selectable = X[selectable_indices]

        if len(selectable_indices) < 2:
            warnings.warn(f"Only {len(selectable_indices)} points with POS > 0 found for class {class_label}. {self.__class__.__name__} may be unreliable. Consider FRS params. Using all minority points for neighbor search if possible.")
            # raise ValueError(f"Only {len(selectable_indices)} points with POS > 0 found for class {class_label}. {self.__class__.__name__} may be unreliable. Consider FRS params. Using all minority points for neighbor search if possible.")
            if len(minority_indices) >= 2:
                X_selectable = X_minority_norm
                selectable_indices = minority_indices
                pos_selectable = pos_minority
            else:
                warnings.warn(f"{len(selectable_indices)} points with POS > 0 found for class {class_label}. and len(minority_indices) < 2 for {self.__class__.__name__}.")
                raise ValueError(f"{len(selectable_indices)} points with POS > 0 found for class {class_label}. and len(minority_indices) < 2 for {self.__class__.__name__}.")
        
        # Fit NN on *normalized* selectable minority points
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(selectable_indices)))
        nn.fit(X_selectable)
        
        # Find neighbors for each selectable point within the *selectable* set
        distances, nn_indices_in_selectable = nn.kneighbors(X_selectable, return_distance=True)

        # Prepare for weighted selection of base point p1 (using original indices)
        p1_candidates = list(zip(selectable_indices, pos_selectable))
        
        return p1_candidates, nn_indices_in_selectable, distances, selectable_indices

    def _generate_new_samples(self, 
                              X,
                              num_samples_to_generate,
                              p1_candidates,
                              indices_selectable_orig,
                              nn_indices_in_selectable,
                              random_state):
        new_samples = []
        epsilon = 1e-9

        for _ in range(num_samples_to_generate):
            # Weighted selection of p1's original index
            p1_idx_orig, _ = math_utils._weighted_random_choice(p1_candidates, random_state)
            if p1_idx_orig is None: continue

            # Find index of p1 within the selectable set (indices_selectable_orig)
            p1_idx_in_selectable_set = np.where(indices_selectable_orig == p1_idx_orig)[0]
            if len(p1_idx_in_selectable_set) == 0: continue
            p1_idx_in_selectable_set = p1_idx_in_selectable_set[0]

            # Get neighbors of p1 (these indices are relative to the selectable set)
            p1_neighbors_indices_in_selectable = nn_indices_in_selectable[p1_idx_in_selectable_set][1:] # Exclude self
            if len(p1_neighbors_indices_in_selectable) == 0: continue

            # Randomly choose one neighbor p2 (index relative to selectable set)
            p2_idx_in_selectable_set = random_state.choice(p1_neighbors_indices_in_selectable)

            # Get original index of p2
            p2_idx_orig = indices_selectable_orig[p2_idx_in_selectable_set]

            # Get POS memberships for bias calculation
            p1_pos = self.positive_region[p1_idx_orig]
            p2_pos = self.positive_region[p2_idx_orig]

            # Calculate lambda
            if self.bias_interpolation:
                denominator = max(p1_pos + p2_pos, epsilon) # Ensure denominator is positive
                lambda_ = p2_pos / denominator
                lambda_ = np.clip(lambda_, 0.0, 1.0)
            else:
                lambda_ = random_state.rand()

            # Interpolate in *original* feature space
            p1_orig = X[p1_idx_orig]
            p2_orig = X[p2_idx_orig]
            new_sample = p1_orig + lambda_ * (p2_orig - p1_orig)
            new_samples.append(new_sample)

        return new_samples
        