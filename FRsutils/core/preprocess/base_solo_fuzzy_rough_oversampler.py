import numpy as np
from collections import Counter
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_X_y
from FRsutils.core.approximations import BaseFuzzyRoughModel
from abc import ABC, abstractmethod
import warnings
from FRsutils.core.models.itfrs import ITFRS
from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models.vqrs import VQRS
from FRsutils.core.similarities import calculate_similarity_matrix, GaussianSimilarity, LinearSimilarity
from FRsutils.core.preprocess.base_allpurpose_fuzzy_rough_oversampler import BaseAllPurposeFuzzyRoughOversampler
from FRsutils.utils.fuzzy_rough_dataset_validation_utils import compatible_dataset_with_FuzzyRough
# import imblearn.over_sampling.base.BaseOverSampler


class BaseSoloFuzzyRoughOversampler(BaseAllPurposeFuzzyRoughOversampler):
    """Base class with FRS calculations.
       This class of resamplers just use Fuzzy-rough sets without combining with any other model
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
                 random_state=None
                 ):
        
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
                        sampling_strategy=sampling_strategy)
        
        self.k_neighbors = k_neighbors
        self.bias_interpolation = bias_interpolation
        self.random_state = random_state
        
    
    def fit(self, X, y):
        """
        @brief Validates the input dataset and builds the fuzzy-rough model.

        @param X Normalized feature matrix (2D np.ndarray).
        @param y Target class labels (1D np.ndarray).

        @return self
        """
        compatible_dataset_with_FuzzyRough(X, y)
        self._check_params()

        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        self.target_stats_ = Counter(y)

        # self.fr_model_params['X'] = X
        # self.fr_model_params['labels'] = y
        self.ensure_built(X, y)

        # self.lower_app = self.fr_model.lower_approximation()
        # self.upper_app = self.fr_model.upper_approximation()
        # self.positive_region = self.lower_app

        return self

    def fit_resample(self, X, y):
        """Resample the dataset."""
        self.fit(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)

        return X_resampled, y_resampled
    
    def transformm(self, X, y=None):
        """
        @brief Applies the resampling transformation to the dataset.
        NOTE: This is necessary to be compatible with scikit learn pipelines

        @details
        Calls fit_resample() using the original data. Compatible with sklearn Pipelines.

        @param X Input feature matrix (2D np.ndarray)
        @param y Target labels (1D np.ndarray)

        @return Tuple[np.ndarray, np.ndarray]: Resampled (X_resampled, y_resampled)

        @raises ValueError If y is None or invalid
        """
        if y is None:
            raise ValueError("y cannot be None when using transform() in resampling context.")

        return self.fit_resample(X, y)

    
    @abstractmethod
    def _check_params(self):
        """
        checks correctness of parameters specific to this object.
        Each derived class must implements its own
        """
        pass
        # raise NotImplementedError("Subclasses must implement _check_params.")
    
    @abstractmethod
    def _fit_resample(self, X, y):
        """Placeholder for the actual resampling logic in subclasses."""
        raise NotImplementedError("Subclasses must implement _fit_resample.")

    @abstractmethod
    def _prepare_minority_samples(self, X, y, class_label):
        """
        @brief with a custom algorithm, it selects a set of minority class
            instances for generating new instances.
        
        @param      X, the non-decision features of dataset
        @type       np.ndarray
        
        @param      y, the vector of classes of samples in X
        @type       np.ndarray

        @param      class_label, the class label of minority class
        @type       int or float???
        
        @return     a list of samples as indexes of X selected for generating samples based on
        @rtype      np.ndarray
        """
        pass

    
    @abstractmethod
    def _generate_new_samples(self):
        pass
        