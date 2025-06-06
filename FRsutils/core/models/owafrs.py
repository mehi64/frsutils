# frutil/models/owafrs.py
"""
OWAFRS implementation.
"""

import FRsutils.core.tnorms as tn
import FRsutils.core.owa_weights as owa_weights
from FRsutils.core.approximations import BaseFuzzyRoughModel
import numpy as np

class OWAFRS(BaseFuzzyRoughModel):
    def __init__(self, 
                 similarity_matrix: np.ndarray,
                 labels: np.ndarray,
                 tnorm: tn.TNorm,
                 implicator,
                 lower_app_weights_method : str,
                 upper_app_weights_method : str):
        super().__init__(similarity_matrix, labels)
        self.tnorm = tnorm
        self.implicator = np.vectorize(implicator)

        # sum_vals = np.sum(lower_approximation_weights) + np.sum(upper_approximation_weights)
        # assert np.isclose(sum_vals, 2.0), "lower & upper weights, each must sum up to 1.0"
        # assert not np.all((lower_approximation_weights >= 0) & (lower_approximation_weights <= 1))
        # assert not np.all((upper_approximation_weights >= 0) & (upper_approximation_weights <= 1))
        # assert not len(lower_approximation_weights) == len(upper_approximation_weights)
        
        if upper_app_weights_method not in ['sup_weights_linear']:
            raise ValueError(f"Unsupported weight type: {upper_app_weights_method}")
        if lower_app_weights_method not in ['inf_weights_linear']:
            raise ValueError(f"Unsupported weight type: {lower_app_weights_method}")
        
        n = len(labels)
        # We generate one less element regarding weights, because in calculations of
        # OWA, the same instance will be excluded from calculations. Therefore, 
        # lower and upper approximations have slightly higher values which are 
        # more realistic. 
        # TODO: check this to be sure the higher values are more realistic 
        if(upper_app_weights_method == 'sup_weights_linear'):
            self.upper_approximation_weights = owa_weights.owa_suprimum_weights_linear(n - 1)
        if(lower_app_weights_method == 'inf_weights_linear'):
            self.lower_approximation_weights = owa_weights.owa_infimum_weights_linear(n - 1)
     

    def lower_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        
        # Since for the calculations of lower approximation, 
        # we use OWA operator which is a essentailly a product, 
        # to exclude the same instance from calculations we set
        # the diagonal to 0.0 which
        # is ignored by the product operator.

        np.fill_diagonal(implication_vals, 0.0)
        # sort row-wise, (left contains the minumum value)
        sorted_matrix = np.sort(implication_vals, axis=1)
        # change the sort order, (left contains the max value)
        sorted_matrix = sorted_matrix[:, ::-1]
        # since the right column contains 0 all the time (because we set 
        # the main diagonal to zero and then sorted that), we remove the right column
        sorted_matrix = sorted_matrix[:, :-1]

        result = np.matmul(sorted_matrix, self.lower_approximation_weights)
        return result

    def upper_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
        
        # Since for the calculations of upper approximation, 
        # we use OWA operator which is a essentailly a product, 
        # to exclude the same instance from calculations we set
        # the diagonal to 0.0 which
        # is ignored by the product operator.

        np.fill_diagonal(tnorm_vals, 0.0)
        sorted_matrix = np.sort(tnorm_vals, axis=1)
        sorted_matrix = sorted_matrix[:, ::-1]
        sorted_matrix = sorted_matrix[:, :-1]

        result = np.matmul(sorted_matrix, self.upper_approximation_weights)
        return result