import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from FRsutils.core.approximations import FuzzyRoughModel_Base
from abc import ABC, abstractmethod
import warnings

allowed_sampling_strategies = ["auto", "xxxxx"]

# Inherits from BaseEstimator to integrate with scikit-learn tooling
# (e.g., Pipeline, GridSearchCV, clone, etc.).
class BaseSoloFuzzyRoughResampler(ABC, BaseEstimator, TransformerMixin):
    """Base class with FRS calculations.
       This class of resamplers just use Fuzzy-rough sets without combining with any other model
    """
    def __init__(self,
                 fr_model : FuzzyRoughModel_Base,
                 sampling_strategy = 'auto'):
        self.fr_model = fr_model
        self.sampling_strategy = sampling_strategy

        self.lower_app = self.fr_model.lower_approximation()
        self.upper_app = self.fr_model.upper_approximation()

        # TODO: check this. Is that correct?what about all models?
        self.positive_region = self.lower_app
        # self.boundary_region = self.fr_model.boundary_region()

    def _compatible_dataset_with_FuzzyRough(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        @brief Validates a dataset consisting of a 2D array `X` and a 1D array `y`
        to be sure it is a valid dataset for fuzzy-rough calculations.

        @details This function ensures:
        - `X` is a 2D NumPy array of floating-point numbers.
        - All elements in `X` are within the range [0.0, 1.0].
        - `y` is a 1D NumPy array.
        - The length of `y` matches the first dimension of `X`.

        @param X A 2D NumPy array of float values representing features. Must be in range [0.0, 1.0].
        @param y A 1D NumPy array whose length is equal to the number of rowa in `X`.

        @throws TypeError If `X` or `y` is not a NumPy array or if `X` is not float.
        @throws ValueError If the shape, dimensionality, or value range conditions are not satisfied.
        """

        ## Check if X is a NumPy ndarray
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy ndarray.")

        ## Check if X is 2-dimensional
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        ## Check if X has float-type elements
        if not np.issubdtype(X.dtype, np.floating):
            raise TypeError("X elements must be of float type.")

        ## Check if all elements in X are within [0.0, 1.0]
        if np.any(X < 0.0) or np.any(X > 1.0):
            raise ValueError("All elements in X must be in the range [0.0, 1.0].")

        ## Check if y is a NumPy ndarray
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy ndarray.")

        ## Check if y is 1-dimensional
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        ## Check if length of y matches the second dimension of X
        if len(y) != X.shape[0]:
            raise ValueError("Length of y must be equal to the first dimension of X.")

        ## All checks passed
        print("Dataset is valid.")


    def fit(self, X, y):
        """@brief Mainly checks the correctness of dataset and FRSMOTE parameters.
            The fit function is called in fit_resample automatically.
            by the user. So, we placed the data and parameter checks here and in.
            NOTE: fit function is needed with the same name because we need out 
            resampler be compatible with pipelines, grid search, etc. in scikit learn
        """
        self._compatible_dataset_with_FuzzyRough(X,y)
        self._check_params()
        # accept_sparse=False means FRSMOTE cannot work with 
        # datasetructures designed to store sparse matrices. We
        # implemented FRSMOTE with numpy arrays. 
        X, y = check_X_y(X, y, accept_sparse=False)
        self.n_features_in_ = X.shape[1]
        self.classes_, _ = np.unique(y, return_counts=True)
        
        self.target_stats_ = Counter(y)
        
        return self

    def fit_resample(self, X, y):
        """Resample the dataset."""
        self.fit(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)

        return X_resampled, y_resampled
    
    def transform(self, X, y=None):
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

    # _get_target_classes and _get_num_samples remain the same as in FRSMOTE
    def _get_target_classes(self):
        """Determine which classes to oversample based on sampling_strategy."""
        
        if self.sampling_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        elif isinstance(self.sampling_strategy, dict):
            return list(self.sampling_strategy.keys())
        # Add more strategy handling if needed (float, list, callable)
        else:
            warnings.warn(f"Unsupported sampling_strategy: {self.sampling_strategy}. Using 'auto'.")
            return [cls for cls in self.classes_ if cls != max(self.target_stats_, key=self.target_stats_.get)]

    def _get_num_samples(self, class_label):
        """Determine number of samples to generate for a class."""
        if self.sampling_strategy == 'auto':
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
        elif isinstance(self.sampling_strategy, dict):
            # Ensure target count is not less than current count
            target_count = max(self.target_stats_[class_label], self.sampling_strategy[class_label])
        else: # Default to balancing against majority if strategy is unclear
             warnings.warn(f"Interpreting sampling_strategy '{self.sampling_strategy}' as 'auto'.")
             majority_class = max(self.target_stats_, key=self.target_stats_.get)
             target_count = self.target_stats_[majority_class]

        current_count = self.target_stats_[class_label]
        return max(0, target_count - current_count)

    @abstractmethod
    def _check_params(self):
        """
        checks correctness of parameters specific to this object.
        Each derived class must implements its own
        """
        if self.sampling_strategy not in allowed_sampling_strategies:
            raise ValueError(f"Invalid strategy '{self.sampling_strategy}'. Allowed values are: {allowed_sampling_strategies}")

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
        