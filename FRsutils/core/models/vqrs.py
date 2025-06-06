"""
VQRS implementation.
"""

from FRsutils.core.approximations import BaseFuzzyRoughModel
import FRsutils.core.tnorms as tn
import numpy as np
import FRsutils.core.fuzzy_quantifiers as fq


class VQRS(BaseFuzzyRoughModel):
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 alpha_lower: float,
                 beta_lower: float,
                 alpha_upper: float,
                 beta_upper: float):
        super().__init__(similarity_matrix, labels)
        
        if(alpha_lower > beta_lower) or (alpha_upper > beta_upper):
            raise ValueError("alpha_lower and beta_lower should be smaller than alpha_upper and beta_upper")
        
        self.alpha_lower = alpha_lower
        self.beta_lower = beta_lower
        self.alpha_upper = alpha_upper
        self.beta_upper = beta_upper

        self.tnorm = tn.MinTNorm()

    def _interim_calculations(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # Since for the calculations of lower and upper approximation in VQRS, 
        # we use a sum operator, to exclude the same instance from calculations we set to 0.0 which
        # is ignored by sum operator. To be sure all is correct,
        # inside code, we set main diagonal to 0.0
        np.fill_diagonal(tnorm_vals, 0.0)
        # print(tnorm_vals)
        nominator =  np.sum(tnorm_vals, axis=1)

        # since the similarity of each instance with itself must not be in the calculations,
        # we can set the main diagonal of the similarity_matrix to 0.0 as well. But instead we reduce 1.0 from the sum
        denominator =  np.sum(self.similarity_matrix, axis=1) - 1.0

        result = nominator / denominator
        return result

    def lower_approximation(self):
        result = self._interim_calculations()
        result = fq.fuzzy_quantifier_quadratic(result,
                                          self.alpha_lower,
                                          self.beta_lower)

        return result

    def upper_approximation(self):
        result = self._interim_calculations()
        result = fq.fuzzy_quantifier_quadratic(result,
                                          self.alpha_upper,
                                          self.beta_upper)

        return result

        