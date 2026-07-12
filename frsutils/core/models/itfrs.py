# SPDX-License-Identifier: BSD-3-Clause
"""Dense NumPy reference implementation of the ITFRS model.

This module defines the direct ITFRS model that consumes a precomputed
similarity matrix. Backend-aware blockwise execution is provided by the
public approximation API and the approximation-engine layer.
"""

from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
import frsutils.core.tnorms as tn
import frsutils.core.implicators as imp
from frsutils.core.models.itfrs_components import build_itfrs_components_from_config
import numpy as np


@FuzzyRoughModel.register("itfrs")
class ITFRS(FuzzyRoughModel):
    """Dense Implicator-TNorm Fuzzy Rough Set approximation model.

    This class is the direct NumPy reference implementation used with an
    already materialized pairwise similarity matrix. Use
    ``compute_approximations(..., model="itfrs", engine="blockwise")`` for
    blockwise execution and optional CuPy-backed similarity blocks.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Precomputed pairwise similarity matrix.
    labels : array-like of shape (n_samples,)
        Class labels aligned with the similarity matrix. Labels are stored as a
        NumPy array so direct dense computations can use vectorized indexing.
    ub_tnorm : TNorm
        T-norm used for the upper approximation.
    lb_implicator : Implicator
        Implicator used for the lower approximation.
    logger : object, default=None
        Optional logger.
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 ub_tnorm: tn.TNorm, 
                 lb_implicator: imp.Implicator,
                 logger=None):
        """Initialize the ITFRS instance."""
        label_array = np.asarray(labels) if labels is not None else None
        super().__init__(similarity_matrix,
                          label_array,
                          logger=logger)
        
        self.validate_params(ub_tnorm=ub_tnorm, 
                             lb_implicator=lb_implicator)

        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator
        
        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def lower_approximation(self) -> np.ndarray:
        """Compute the lower approximation using the implicator.
                
                Returns
                -------
                np.ndarray
                    Lower approximation array (n,)
                
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 1.0)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self) -> np.ndarray:
        """Compute the upper approximation using the T-norm.
                
                Returns
                -------
                np.ndarray
                    Upper approximation array (n,)
                
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        return np.max(tnorm_vals, axis=1)

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the ITFRS model to a dictionary.
                
                Parameters
                ----------
                include_data : bool
                    If True, include similarity_matrix and labels in the output.
                
                Returns
                -------
                dict
                    Dictionary representation of the model.
                
        """
        data = {
            "type": "itfrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data


    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "ITFRS":
        """Reconstruct an ITFRS model from a serialized dictionary.
                
                Parameters
                ----------
                data : dict
                    Serialized dictionary (from to_dict)
                similarity_matrix : object
                    Optional matrix to override or fill in if not in data
                labels : object
                    Optional label vector to override or fill in if not in data
                logger : object
                    Optional logger
                
                Returns
                -------
                'ITFRS'
                    ITFRS instance
                
        """
        
        # Rebuild operators
        tnorm = tn.TNorm.from_dict(data["ub_tnorm"])
        implicator = imp.Implicator.from_dict(data["lb_implicator"])

        # Use matrix and labels from dict if present, else fallback to args
        sim = np.array(data["similarity_matrix"]) if "similarity_matrix" in data else similarity_matrix
        lbl = np.array(data["labels"]) if "labels" in data else labels

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in data or as arguments.")

        return cls(sim, lbl, tnorm, implicator, logger=logger)

    def describe_params_detailed(self) -> dict:
        """Describe internal T-norm and implicator parameters.
                
                Returns
                -------
                dict
                    Dictionary describing parameters of components.
                
        """
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed()
        }
    
    def _get_params(self) -> dict:
        """Describe internal T-norm and implicator parameters.
                
                Returns
                -------
                dict
                    Dictionary containing T-norm and implicator used in itfrs.
                
        """
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "similarity_matrix":self.similarity_matrix,
            "labels":self.labels
        }

    @classmethod
    def validate_params(cls, **kwargs):
        """validation hook.
        
        Parameters
        ----------
        kwargs : object
            Parameter value.
        """
        
        tnrm = kwargs.get("ub_tnorm")
        if tnrm is None or not isinstance(tnrm, tn.TNorm):
            raise ValueError("Parameter 'tnorm' must be provided and be an instance of derived classes from TNorm.")

        impli = kwargs.get("lb_implicator")
        if impli is None or not isinstance(impli, imp.Implicator):
            raise ValueError("Parameter 'implicator' must be provided and be an instance of derived classes from Implicator.")


    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "ITFRS":
        """Create a dense ITFRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), default=None
            Optional dense similarity matrix. Overrides matrix data in `config`
            when provided.
        labels : array-like of shape (n_samples,), default=None
            Optional label vector. Overrides labels in `config` when provided.
        **config : dict
            Flat ITFRS config, internal nested config under `_nested_config`, or
            serialized operator specs.

        Returns
        -------
        ITFRS
            Dense ITFRS model configured with the requested operators.
        """
        config = dict(config)
        ub_tnorm, lb_implicator = build_itfrs_components_from_config(
            config,
            require_explicit_components=True,
        )

        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either as arguments or in the config dictionary.")

        logger = config.get("logger", None)
        return cls(sim, lbl, ub_tnorm, lb_implicator, logger=logger)
