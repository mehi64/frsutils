# SPDX-License-Identifier: BSD-3-Clause
"""OWAFRS model implementation for OWA-based fuzzy-rough approximations.

This module belongs to the core fuzzy-rough computation layer.
"""

from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
import frsutils.core.tnorms as tn
import frsutils.core.owa_weights as owa_weights
import frsutils.core.implicators as imp
from frsutils.core.models.owafrs_components import build_owafrs_components_from_config
from frsutils.utils.logger.logger_util import get_logger
import numpy as np


@FuzzyRoughModel.register("owafrs")
class OWAFRS(FuzzyRoughModel):
    """Dense NumPy reference model for OWA-based fuzzy-rough approximations.

    OWAFRS computes lower and upper approximations from a fully materialized
    similarity matrix, class labels, an upper T-norm, a lower implicator, and
    lower/upper OWA weighting strategies. Scalable blockwise execution and
    optional CuPy-backed similarity blocks are handled outside this class by
    :mod:`frsutils.core.approximation_engines`.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Dense pairwise similarity matrix.
    labels : array-like of shape (n_samples,)
        Class labels aligned with the similarity matrix.
    ub_tnorm : TNorm
        T-norm used for upper-approximation evidence.
    lb_implicator : Implicator
        Implicator used for lower-approximation evidence.
    ub_owa_method, lb_owa_method : OWAWeights
        OWA weighting strategies for upper and lower approximation aggregation.
    logger : object, optional
        Optional logger used for diagnostic messages.
    """

    def __init__(self,
                 similarity_matrix: np.ndarray,
                 labels: np.ndarray,
                 ub_tnorm: tn.TNorm,
                 lb_implicator: imp.Implicator,
                 ub_owa_method: owa_weights.OWAWeights,
                 lb_owa_method: owa_weights.OWAWeights,
                 logger=None):
        
        """Initialize the dense OWAFRS reference model."""
        if labels is None:
            raise ValueError("labels must be provided.")

        labels_array = np.asarray(labels)
        if labels_array.ndim != 1:
            raise ValueError("labels must be a one-dimensional array-like object.")
        if labels_array.shape[0] < 2:
            raise ValueError("OWAFRS requires at least two samples for OWA aggregation.")

        super().__init__(similarity_matrix,
                          labels_array,
                          logger=logger)
        self.validate_params(ub_tnorm=ub_tnorm, 
                             lb_implicator=lb_implicator,
                             ub_owa_method=ub_owa_method,
                             lb_owa_method=lb_owa_method)

        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator

        self.ub_owa_method = ub_owa_method
        self.lb_owa_method = lb_owa_method

        n = self.labels.shape[0]

        # n-1 because the same instance is not included in calculations
        self.lb_owa_weights = lb_owa_method.weights(n-1, order='asc')
        self.ub_owa_weights = ub_owa_method.weights(n-1, order='desc')

        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def lower_approximation(self) -> np.ndarray:
        """Compute dense OWAFRS lower approximation values.

        Returns
        -------
        lower : ndarray of shape (n_samples,)
            OWA-aggregated lower approximation values.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        
        # to omitt the same instance from calculations, the
        # main diagonal is set to 0.0. then each row is sorted in descending order
        # then the last row is sliced because the last row always contains 0.0.
        # finally, the matrix is multiplied by the owa weights.
        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.lb_owa_weights)


    def upper_approximation(self) -> np.ndarray:
        """Compute dense OWAFRS upper approximation values.

        Returns
        -------
        upper : ndarray of shape (n_samples,)
            OWA-aggregated upper approximation values.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        
        # to omitt the same instance from calculations, the
        # main diagonal is set to 0.0. then each row is sorted in descending order
        # then the last row is sliced because the last row always contains 0.0.
        # finally, the matrix is multiplied by the owa weights.
        np.fill_diagonal(tnorm_vals, 0.0)
        sorted_matrix = np.sort(tnorm_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.ub_owa_weights)

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the OWAFRS model to a dictionary.

        Parameters
        ----------
        include_data : bool, default=False
            If True, include ``similarity_matrix`` and ``labels`` in the output.

        Returns
        -------
        data : dict
            Serialized OWAFRS component configuration and optional data arrays.
        """
        data = {
            "type": "owafrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict(),
            "ub_owa_method": self.ub_owa_method.to_dict(),
            "lb_owa_method": self.lb_owa_method.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data
    
    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "OWAFRS":
        """Reconstruct an OWAFRS model from a serialized dictionary.

        Parameters
        ----------
        data : dict
            Serialized dictionary produced by :meth:`to_dict`.
        similarity_matrix : array-like of shape (n_samples, n_samples), optional
            Dense similarity matrix. When provided, this value overrides any
            matrix embedded in ``data``.
        labels : array-like of shape (n_samples,), optional
            Class labels. When provided, this value overrides any labels embedded
            in ``data``.
        logger : object, optional
            Optional logger for the reconstructed model.

        Returns
        -------
        OWAFRS
            Reconstructed dense OWAFRS model.
        """
        
        # Rebuild operators
        tnorm = tn.TNorm.from_dict(data["ub_tnorm"])
        implicator = imp.Implicator.from_dict(data["lb_implicator"])

        # External arrays intentionally override embedded arrays so serialized
        # component configs can be reused on new datasets.
        sim = similarity_matrix if similarity_matrix is not None else (np.array(data["similarity_matrix"]) if "similarity_matrix" in data else None)
        lbl = labels if labels is not None else (np.array(data["labels"]) if "labels" in data else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in data or as arguments.")

        ub_owa_method = owa_weights.OWAWeights.from_dict(data["ub_owa_method"])
        lb_owa_method = owa_weights.OWAWeights.from_dict(data["lb_owa_method"])
        
        return cls(sim, lbl, tnorm, implicator, ub_owa_method, lb_owa_method, logger=logger)

   
    def describe_params_detailed(self) -> dict:
        """Return detailed parameter metadata for this component."""
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed(),
            "ub_owa_method": self.ub_owa_method.describe_params_detailed(),
            "lb_owa_method": self.lb_owa_method.describe_params_detailed()
        }

    @classmethod
    def validate_params(cls, **kwargs):
        """Validate OWAFRS component objects at construction time."""
        
        tnrm = kwargs.get("ub_tnorm")
        if tnrm is None or not isinstance(tnrm, tn.TNorm):
            raise ValueError("Parameter 'tnorm' must be provided and be an instance of derived classes from TNorm.")

        impli = kwargs.get("lb_implicator")
        if impli is None or not isinstance(impli, imp.Implicator):
            raise ValueError("Parameter 'implicator' must be provided and be an instance of derived classes from Implicator.")

        ub_owa_method = kwargs.get("ub_owa_method")
        if ub_owa_method is None or not isinstance(ub_owa_method, owa_weights.OWAWeights):
            raise ValueError("Parameter 'ub_owa_method' must be provided and be an instance of derived classes from OWAWeights.")
        
        lb_owa_method = kwargs.get("lb_owa_method")
        if lb_owa_method is None or not isinstance(lb_owa_method, owa_weights.OWAWeights):
            raise ValueError("Parameter 'lb_owa_method' must be provided and be an instance of derived classes from OWAWeights.")

    def _get_params(self) -> dict:
        """Return dense OWAFRS component and data parameters."""
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "ub_owa_method":self.ub_owa_method,
            "lb_owa_method":self.lb_owa_method,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "OWAFRS":
        """Create a dense OWAFRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), default=None
            Optional dense similarity matrix. Overrides matrix data in ``config``
            when provided.
        labels : array-like of shape (n_samples,), default=None
            Optional label vector. Overrides labels in ``config`` when provided.
        **config : dict
            Flat OWAFRS config, internal nested config under ``_nested_config``,
            serialized component specs, or direct component instances.

        Returns
        -------
        OWAFRS
            Dense OWAFRS model configured with the requested components.
        """
        config = dict(config)
        ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method = build_owafrs_components_from_config(
            config,
            require_explicit_components=True,
        )

        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either as arguments or in the config dictionary.")

        logger = config.get("logger", None)
        return cls(sim, lbl, ub_tnorm, lb_implicator, ub_owa_method, lb_owa_method, logger=logger)
