# SPDX-License-Identifier: BSD-3-Clause
"""Dense NumPy reference implementation of the VQRS model.

The direct :class:`VQRS` class consumes a materialized similarity matrix and is
kept as the readable dense reference path. Scalable NumPy/CuPy-aware execution
for public workflows belongs to the blockwise approximation-engine layer.
"""

import numpy as np
import frsutils.core.tnorms as tn
from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.vqrs_components import build_vqrs_components_from_config
from frsutils.core.models.vqrs_math import compute_vqrs_interim_ratio



@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """Dense reference VQRS model using fuzzy quantifiers.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Materialized pairwise similarity matrix.
    labels : array-like of shape (n_samples,)
        Class labels aligned with ``similarity_matrix``. Labels are stored as a
        NumPy array so direct dense computations can use vectorized indexing.
    lb_fuzzy_quantifier : FuzzyQuantifier
        Fuzzy quantifier applied to the VQRS interim ratio for the lower
        approximation.
    ub_fuzzy_quantifier : FuzzyQuantifier
        Fuzzy quantifier applied to the VQRS interim ratio for the upper
        approximation.
    logger : object, optional
        Logger used by the base fuzzy-rough model.

    Notes
    -----
    This class is the dense NumPy reference implementation. Public blockwise
    execution, including optional CuPy-backed accumulators, is implemented in
    :mod:`frsutils.core.approximation_engines` and exposed through
    :func:`frsutils.compute_approximations`.
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 lb_fuzzy_quantifier: FuzzyQuantifier,
                 ub_fuzzy_quantifier: FuzzyQuantifier,
                 logger=None):
        """Initialize a dense VQRS reference model."""
        label_array = np.asarray(labels) if labels is not None else None
        super().__init__(similarity_matrix,
                         label_array,
                         logger=logger)
        
        self.validate_params(
            lb_fuzzy_quantifier=lb_fuzzy_quantifier,
            ub_fuzzy_quantifier=ub_fuzzy_quantifier
        )

        self.lb_fuzzy_quantifier = lb_fuzzy_quantifier
        self.ub_fuzzy_quantifier = ub_fuzzy_quantifier
        self.tnorm = tn.MinTNorm()

        self.logger.debug(f"{self.__class__.__name__} initialized.")


    def _interim_calculations(self) -> np.ndarray:
        """Compute the VQRS interim ratio before fuzzy quantification."""
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # Exclude self-comparisons from both the supporting and total mass.
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)
        nonself_mask = ~np.eye(self.similarity_matrix.shape[0], dtype=bool)
        denominator = np.sum(
            self.similarity_matrix,
            axis=1,
            where=nonself_mask,
            initial=0.0,
        )

        return compute_vqrs_interim_ratio(numerator, denominator, xp=np)

    def lower_approximation(self) -> np.ndarray:
        """Compute the lower approximation using the fuzzy quantifier.
                
                Returns
                -------
                np.ndarray
                    Lower approximation array (n,)
                
        """
        return self.lb_fuzzy_quantifier(self._interim_calculations())

    def upper_approximation(self) -> np.ndarray:
        """Compute the upper approximation using the fuzzy quantifier.
                
                Returns
                -------
                np.ndarray
                    Upper approximation array (n,)
                
        """
        return self.ub_fuzzy_quantifier(self._interim_calculations())

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the VQRS model to a dictionary.
                
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
            "type": "vqrs",
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.to_dict(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.to_dict()
        }

        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict, similarity_matrix=None, labels=None, logger=None) -> "VQRS":
        """Reconstruct an VQRS model from a serialized dictionary.
                
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
                'VQRS'
                    VQRS instance
                
        """
        fq_lower = FuzzyQuantifier.from_dict(data["lb_fuzzy_quantifier"])
        fq_upper = FuzzyQuantifier.from_dict(data["ub_fuzzy_quantifier"])

        # Use matrix and labels from dict if present, else fallback to args
        sim = np.array(data["similarity_matrix"]) if "similarity_matrix" in data else similarity_matrix
        lbl = np.array(data["labels"]) if "labels" in data else labels

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in data or as arguments.")

        return cls(sim, lbl, fq_lower, fq_upper, logger=logger)

    def _get_params(self) -> dict:
        """Describe internal lower and upper fuzzy_quantifier parameters.
                
                Returns
                -------
                dict
                    Dictionary containing lower and upper fuzzy_quantifier used in vqrs.
                
        """
        return {
            "tnorm": self.tnorm,
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier,
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "VQRS":
        """Create a dense VQRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), default=None
            Optional dense similarity matrix. Overrides matrix data in ``config``
            when provided.
        labels : array-like of shape (n_samples,), default=None
            Optional label vector. Overrides labels in ``config`` when provided.
        **config : dict
            Flat VQRS config, internal nested config under ``_nested_config``, or
            serialized fuzzy-quantifier specs.

        Returns
        -------
        VQRS
            Dense VQRS model configured with the requested fuzzy quantifiers.
        """
        config = dict(config)
        lb_fuzzy_quantifier, ub_fuzzy_quantifier, _ = build_vqrs_components_from_config(
            config,
            require_explicit_components=True,
        )

        sim = similarity_matrix if similarity_matrix is not None else (np.array(config["similarity_matrix"]) if "similarity_matrix" in config else None)
        lbl = labels if labels is not None else (np.array(config["labels"]) if "labels" in config else None)

        if sim is None or lbl is None:
            raise ValueError("similarity_matrix and labels must be provided either in config or as arguments.")

        logger = config.get("logger", None)
        return cls(sim, lbl, lb_fuzzy_quantifier, ub_fuzzy_quantifier, logger=logger)
    
    @classmethod
    def validate_params(cls, **kwargs):
        """validation hook.
        
        Parameters
        ----------
        kwargs : object
            Parameter value.
        """
        fq_l = kwargs.get("lb_fuzzy_quantifier")
        fq_u = kwargs.get("ub_fuzzy_quantifier")

        if fq_l is None or not isinstance(fq_l, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_lower must be a valid FuzzyQuantifier instance.")
        
        if fq_u is None or not isinstance(fq_u, FuzzyQuantifier):
            raise ValueError("fuzzy_quantifier_upper must be a valid FuzzyQuantifier instance.")

    def describe_params_detailed(self) -> dict:
        """Return detailed parameter metadata for this component."""
        return {
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.describe_params_detailed(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.describe_params_detailed()
        }
