# SPDX-License-Identifier: BSD-3-Clause
"""OWA weight generators for ordered fuzzy-rough aggregation.

This module belongs to the core fuzzy-rough computation layer.
"""

import numbers

import numpy as np
from abc import abstractmethod
from frsutils.utils.constructor_utils.registry_factory_mixin import RegistryFactoryMixin


class OWAWeights(RegistryFactoryMixin):
    """Abstract base class for OWA weight strategies.
    
    Subclasses must implement `_raw_weights(n)` that generates unnormalized weights.
    This class handles normalization and sorting.
    """
    def weights(self, n: int, order: str = 'asc') -> np.ndarray:
        """Unified method to retrieve OWA weights in specified order.
                
                Parameters
                ----------
                n : int
                    Number of weights to compute.
                order : str
                    'asc' for increasing weights, 'desc' for decreasing weights.
                
                Returns
                -------
                np.ndarray
                    Normalized OWA weight vector.
                
        """
        if not isinstance(order, str):
            raise ValueError("order must be a string; either 'asc' or 'desc'.")

        order = order.lower()
        if order not in ('asc', 'desc'):
            raise ValueError("order must be a string; either 'asc' or 'desc'")
        
        self._validate_n(n)
        raw_weights = self._raw_weights(n)

        norm = self._normalize(raw_weights)

        if order == 'asc':
            norm = np.sort(norm)
        elif order == 'desc':
            norm = np.sort(norm)[::-1]

        return norm

    @abstractmethod
    def _raw_weights(self, n: int) -> np.ndarray:
        """Generate unnormalized, unsorted weight values.
                
                Parameters
                ----------
                n : int
                    Number of weights
                
                Returns
                -------
                np.ndarray
                    Raw weights (to be normalized and sorted)
                
        """
        raise NotImplementedError("Subclasses must implement _raw_weights(n)")

    def _validate_n(self, n: int):
        """Validates the number of weights.
                
                Raises ValueError if:
                - n is not a positive integer
                - n > 20 for ExponentialOWAWeightStrategy due to potential weight overflow
                
                Parameters
                ----------
                n : int
                    Number of weights
                
                Raises
                ------
                ValueError
                    On invalid or unsafe values
                
        """
        if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n <= 0:
            raise ValueError("n must be a positive integer")

        # Exponential strategy has numerical stability risks for large n
        if self.__class__.__name__ == "ExponentialOWAWeights" and n > 20:
            raise ValueError("In ExponentialOWAWeights, n cannot be greater than 20 due to risk of weight explosion.")


    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        """Normalizes weights.
                
                Parameters
                ----------
                weights : np.ndarray
                    Unnormalized weights
                order : object
                    'asc' or 'desc'
                
                Returns
                -------
                np.ndarray
                    Normalized and ordered weights
                
                Raises
                ------
                ValueError
                    If weights are not valid or order is invalid
                
        """
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            raise ValueError("Invalid weight normalization")
        norm = weights / total
        norm_sum = norm.sum()
        assert np.isclose(norm_sum, 1.0)
        return norm


@OWAWeights.register("linear", "additive")
class LinearOWAWeights(OWAWeights):
    """Linear OWA weighting strategy.
    
    Generates linearly increasing weights.
    """
    def _raw_weights(self, n: int) -> np.ndarray:
        return np.arange(1, n + 1, dtype=np.longdouble)

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def to_dict(self) -> dict:
        """Serialize the component configuration to a dictionary."""
        return {"type": self.__class__.__name__, "name": "linear", "params": self._get_params()}


@OWAWeights.register("exponential", "exp", "gp")
class ExponentialOWAWeights(OWAWeights):
    """Exponential OWA weighting strategy.
    
    Generates exponentially increasing weights controlled by a base.
    """
    def __init__(self, base: float = 2.0):
        """Initialize the ExponentialOWAWeights instance."""
        self.validate_params(base=base)
        self.base = base

    def _raw_weights(self, n: int) -> np.ndarray:
        return self.base ** np.arange(1, n + 1, dtype=np.longdouble)

    def _get_params(self) -> dict:
        return {"base": self.base}

    def to_dict(self) -> dict:
        """Serialize the component configuration to a dictionary."""
        return {"type": self.__class__.__name__, "name": "exponential", "params": self._get_params()}

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        base = kwargs.get("base")
        if (
            base is None
            or isinstance(base, bool)
            or not isinstance(base, numbers.Real)
            or not np.isfinite(base)
            or base <= 1
        ):
            raise ValueError("Parameter 'base' must be > 1")


@OWAWeights.register("harmonic", "harm", "inv_add")
class HarmonicOWAWeights(OWAWeights):
    """Harmonic OWA weighting strategy.
    
    Generates weights inversely proportional to index (1/i).
    """
    def _raw_weights(self, n: int) -> np.ndarray:
        return 1.0 / np.arange(1, n + 1, dtype=np.longdouble)

    def _get_params(self) -> dict:
        return {}

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate constructor parameters for this component."""
        pass

    def to_dict(self) -> dict:
        """Serialize the component configuration to a dictionary."""
        return {"type": self.__class__.__name__, "name": "harmonic", "params": self._get_params()}

