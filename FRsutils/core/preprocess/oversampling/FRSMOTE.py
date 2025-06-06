import numpy as np
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors
import warnings
import FRsutils.utils.math_utils.math_utils as math_utils
from FRsutils.core.preprocess.base_solo_fuzzy_rough_oversampler import BaseSoloFuzzyRoughOversampler
from FRsutils.core.approximations import BaseFuzzyRoughModel

from FRsutils.utils.constructor_utils.fr_model_builder import build_fuzzy_rough_model
from FRsutils.utils.constructor_utils.tnorm_builder import build_tnorm
from FRsutils.utils.constructor_utils.similarity_builder import build_similarity
from FRsutils.core.similarities import calculate_similarity_matrix



# --- FRSMOTE Implementation ---

class FRSMOTE(BaseSoloFuzzyRoughOversampler):
    """
    @brief Fuzzy Rough Set based SMOTE (FRSMOTE) Oversampler.
    """

    def __init__(self,                
                 fr_model_type='ITFRS',
                 lb_implicator_type='reichenbach',
                 ub_tnorm_type='product',
                 owa_weighting_strategy_type='linear',
                 fuzzy_quantifier_type='quadratic',
                 alpha_lower=0.1,
                 beta_lower=0.6,
                 alpha_upper=0.2,
                 beta_upper=1.0,
                 similarity_type='gaussian',
                 gaussian_similarity_sigma=0.2,
                 similarity_tnorm_type='minimum',
                 instance_ranking_strategy='pos',
                 sampling_strategy='auto',
                 k_neighbors=5,
                 bias_interpolation=False,
                 random_state=None):
        
        super().__init__(fr_model_type=fr_model_type,
                        lb_implicator_type=lb_implicator_type,
                        ub_tnorm_type=ub_tnorm_type,
                        owa_weighting_strategy_type=owa_weighting_strategy_type,
                        fuzzy_quantifier_type=fuzzy_quantifier_type,
                        alpha_lower=alpha_lower,
                        beta_lower=beta_lower,
                        alpha_upper=alpha_upper,
                        beta_upper=beta_upper,
                        similarity_type=similarity_type,
                        gaussian_similarity_sigma=gaussian_similarity_sigma,
                        similarity_tnorm_type=similarity_tnorm_type,
                        instance_ranking_strategy=instance_ranking_strategy,
                        sampling_strategy=sampling_strategy,
                        k_neighbors = k_neighbors,
                        bias_interpolation = bias_interpolation,
                        random_state = random_state)
        
              

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
  
    def supported_strategies(self):
        return  {'auto', 'balance_minority'}  
    
    # def get_params(self, deep=True):
    #     """
    #     @brief Returns all parameters including nested fuzzy rough model parameters.

    #     @param deep If True, will return parameters of nested objects.

    #     @return Dictionary of parameter names and values.
    #     """
    #     # Start with known top-level parameters
    #     params = {
    #         'fr_model_name': self.fr_model_type,
    #         'similarity_name': self.similarity_name,
    #         'similarity_tnorm_name': self.similarity_tnorm_name,
    #         'instance_ranking_strategy_name': self.instance_ranking_strategy_name,
    #         'sampling_strategy': self.sampling_strategy,
    #         'k_neighbors': self.k_neighbors,
    #         'bias_interpolation': self.bias_interpolation,
    #         'random_state': self.random_state,
    #         'fr_model_params': self.fr_model_params
    #     }

    #     # Add fuzzy rough model parameters (those passed via **kwargs in init)
    #     if hasattr(self, 'fr_model_params'):
    #         for k, v in self.fr_model_params.items():
    #             params[f'{k}'] = v

    #     return params
    
    # def set_params(self, **params):
    #     """
    #     @brief Sets the parameters including nested fuzzy rough model parameters.

    #     @param params Dictionary of parameters to set.

    #     @return self
    #     """
    #     # Separate top-level and nested fuzzy rough parameters
    #     fr_model_params = self.fr_model_params.copy() if hasattr(self, 'fr_model_params') else {}

    #     for key, value in params.items():
    #         if key.startswith("fr_model_params__"):
    #             inner_key = key[len("fr_model_params__"):]
    #             fr_model_params[inner_key] = value
    #         else:
    #             setattr(self, key, value)

    #     self.fr_model_params = fr_model_params
    #     return self