# SPDX-License-Identifier: BSD-3-Clause
"""VQRS model implementation for variable-precision fuzzy-rough approximations.

This module belongs to the core fuzzy-rough computation layer.
"""

import numpy as np
import FRsutils.core.tnorms as tn
from FRsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from FRsutils.core.models.fuzzy_rough_model import FuzzyRoughModel



@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """VQRS model for fuzzy rough approximation using fuzzy quantifiers.
    
    Parameters
    ----------
    similarity_matrix : object
        Pairwise similarity matrix (n x n)
    labels : object
        Corresponding label vector (n,)
    fuzzy_quantifier_lower : object
        FuzzyQuantifier instance for lower approx
    fuzzy_quantifier_upper : object
        FuzzyQuantifier instance for upper approx
    """
    def __init__(self, 
                 similarity_matrix: np.ndarray, 
                 labels: np.ndarray, 
                 lb_fuzzy_quantifier: FuzzyQuantifier,
                 ub_fuzzy_quantifier: FuzzyQuantifier,
                 logger=None):
        """Initialize the VQRS instance."""
        super().__init__(similarity_matrix, 
                         labels, 
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
        """R_Ay which is then feed into upper and lower quantifiers"""
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)

        # to ommit the same instace from calculations, we set the main diagonal to 0.0
        # later on the sum operator ignores that.
        np.fill_diagonal(tnorm_vals, 0.0)
        numerator = np.sum(tnorm_vals, axis=1)

        # since the similarity matrix has the main diagonal 1.0, the sum needs to cbe reduced by 1.0
        # so we reduce
        denominator = np.sum(self.similarity_matrix, axis=1) - 1.0

        interim = numerator / denominator
        return interim

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
        """Serialize the VQS model to a dictionary.
                
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
        """Create a VQRS instance from a configuration dictionary.
                
                Parameters
                ----------
                config : dict
                    Serialized config dict (can include tnorm, fuzzy quantifiers, and optionally data)
                similarity_matrix : object
                    Optional override for similarity matrix
                labels : object
                    Optional override for label vector
                
                Returns
                -------
                'VQRS'
                    VQRS instance
                
        """
        nested = config.pop("_nested_config", None)

        lb_fuzzy_quantifier = None
        ub_fuzzy_quantifier = None

        if isinstance(nested, dict):
            fr_cfg = nested.get("fr_model", {})
            lb_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("lb_fuzzy_quantifier"))
            ub_fuzzy_quantifier = FuzzyQuantifier.create_from_spec(fr_cfg.get("ub_fuzzy_quantifier"))

        # Backward-compatible: flat config keys
        if ub_fuzzy_quantifier is None:
            ub_fq = config.get("ub_fuzzy_quantifier")
            if isinstance(ub_fq, dict):
                ub_fuzzy_quantifier = FuzzyQuantifier.from_dict(ub_fq)
            elif ub_fq is not None:
                ub_fuzzy_quantifier = ub_fq
            else:
                name = config.get("ub_fuzzy_quantifier_name")
                # Prefer new flat naming: ub_fuzzy_quantifier_alpha/beta
                params = {k[len("ub_fuzzy_quantifier_"):]: v for k, v in config.items() if k.startswith("ub_fuzzy_quantifier_") and k != "ub_fuzzy_quantifier_name"}
                if not params:
                    # legacy: ub_alpha/ub_beta (handled by normalizer but keep safety)
                    params = {k[len("ub_"):]: v for k, v in config.items() if k.startswith("ub_") and k in {"ub_alpha", "ub_beta"}}
                ub_fuzzy_quantifier = FuzzyQuantifier.create(name, **params)

        if lb_fuzzy_quantifier is None:
            lb_fq = config.get("lb_fuzzy_quantifier")
            if isinstance(lb_fq, dict):
                lb_fuzzy_quantifier = FuzzyQuantifier.from_dict(lb_fq)
            elif lb_fq is not None:
                lb_fuzzy_quantifier = lb_fq
            else:
                name = config.get("lb_fuzzy_quantifier_name")
                params = {k[len("lb_fuzzy_quantifier_"):]: v for k, v in config.items() if k.startswith("lb_fuzzy_quantifier_") and k != "lb_fuzzy_quantifier_name"}
                if not params:
                    params = {k[len("lb_"):]: v for k, v in config.items() if k.startswith("lb_") and k in {"lb_alpha", "lb_beta"}}
                lb_fuzzy_quantifier = FuzzyQuantifier.create(name, **params)

        # Handle matrix and labels
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
