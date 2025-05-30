# frutil/models/itfrs.py
"""
ITFRS implementation.
"""

from FRsutils.core.approximations import FuzzyRoughModel_Base
import FRsutils.core.tnorms as tn
import numpy as np

class ITFRS(FuzzyRoughModel_Base):
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 tnorm: tn.TNorm, 
                 implicator):
        super().__init__(similarity_matrix, labels)
        self.tnorm = tnorm
        self.implicator = np.vectorize(implicator)

    def lower_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.implicator(self.similarity_matrix, label_mask)
        
        # Since for the calculations of lower approximation, 
        # we calculate Inf which is basically a minimum, 
        # to exclude the same instance from calculations we don’t
        #  need anything because the diagonal is set to 1.0 which
        #  is ignored by min operator. To be sure all is correct,
        #  inside code, we set main diagonal to 1.0
        np.fill_diagonal(implication_vals, 1.0)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self):
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # Since for the calculations of upper approximation, 
        # we calculate sup which is basically a maximum, 
        # to exclude the same instance from calculations we 
        # need to set the main diagonal to 0.0 which is ignored 
        # by max operator.  Otherwise all upper approxamations will be 1.0.
        np.fill_diagonal(tnorm_vals, 0.0)
        return np.max(tnorm_vals, axis=1)