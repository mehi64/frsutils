# SPDX-License-Identifier: BSD-3-Clause
"""Dense NumPy reference implementation of the VQRS model.

Exact blockwise NumPy and optional CuPy-backed execution are provided by the
approximation-engine layer and exposed through the public API.
"""

import numpy as np

import frsutils.core.tnorms as tn
from frsutils.core.fuzzy_quantifiers import FuzzyQuantifier
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.vqrs_components import build_vqrs_components_from_config
from frsutils.core.models.vqrs_math import compute_vqrs_interim_ratio


@FuzzyRoughModel.register("vqrs")
class VQRS(FuzzyRoughModel):
    """Dense Vaguely Quantified Rough Set reference model.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Dense fuzzy-relation matrix using the ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Class labels aligned with the relation matrix.
    lb_fuzzy_quantifier : FuzzyQuantifier
        Quantifier applied to the interim ratio for the lower approximation.
    ub_fuzzy_quantifier : FuzzyQuantifier
        Quantifier applied to the interim ratio for the upper approximation.
    logger : logging.Logger or None, default=None
        Optional logger.
    """

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        lb_fuzzy_quantifier: FuzzyQuantifier,
        ub_fuzzy_quantifier: FuzzyQuantifier,
        logger=None,
    ):
        """Initialize a dense VQRS reference model."""
        label_array = np.asarray(labels) if labels is not None else None
        super().__init__(similarity_matrix, label_array, logger=logger)
        self.validate_params(
            lb_fuzzy_quantifier=lb_fuzzy_quantifier,
            ub_fuzzy_quantifier=ub_fuzzy_quantifier,
        )
        self.lb_fuzzy_quantifier = lb_fuzzy_quantifier
        self.ub_fuzzy_quantifier = ub_fuzzy_quantifier
        self.tnorm = tn.MinTNorm()
        self.logger.debug(f"{self.__class__.__name__} initialized.")

    def _interim_calculations(self) -> np.ndarray:
        """Compute the non-self VQRS support-to-similarity ratio."""
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.tnorm(self.similarity_matrix, label_mask)
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
        """Compute dense VQRS lower-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Lower quantifier values applied to the interim ratio.
        """
        return self.lb_fuzzy_quantifier(self._interim_calculations())

    def upper_approximation(self) -> np.ndarray:
        """Compute dense VQRS upper-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Upper quantifier values applied to the interim ratio.
        """
        return self.ub_fuzzy_quantifier(self._interim_calculations())

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the VQRS model.

        Parameters
        ----------
        include_data : bool, default=False
            Include ``similarity_matrix`` and ``labels`` when True.

        Returns
        -------
        dict
            Serialized quantifier configuration and optional data arrays.
        """
        data = {
            "type": "vqrs",
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.to_dict(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.to_dict(),
        }
        if include_data:
            data["similarity_matrix"] = self.similarity_matrix.tolist()
            data["labels"] = self.labels.tolist()
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict,
        similarity_matrix=None,
        labels=None,
        logger=None,
    ) -> "VQRS":
        """Reconstruct a VQRS model from serialized configuration.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        similarity_matrix : array-like or None, default=None
            Matrix used when serialized data are absent.
        labels : array-like or None, default=None
            Labels used when serialized data are absent.
        logger : logging.Logger or None, default=None
            Optional logger.

        Returns
        -------
        VQRS
            Reconstructed dense VQRS model.
        """
        lower_quantifier = FuzzyQuantifier.from_dict(data["lb_fuzzy_quantifier"])
        upper_quantifier = FuzzyQuantifier.from_dict(data["ub_fuzzy_quantifier"])
        sim = (
            np.array(data["similarity_matrix"])
            if "similarity_matrix" in data
            else similarity_matrix
        )
        lbl = np.array(data["labels"]) if "labels" in data else labels
        if sim is None or lbl is None:
            raise ValueError(
                "similarity_matrix and labels must be provided either in data "
                "or as arguments."
            )
        return cls(sim, lbl, lower_quantifier, upper_quantifier, logger=logger)

    def _get_params(self) -> dict:
        """Return component and data parameters used by this model."""
        return {
            "tnorm": self.tnorm,
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier,
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels,
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "VQRS":
        """Create a dense VQRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), optional
            Optional dense matrix overriding matrix data in ``config``.
        labels : array-like of shape (n_samples,), optional
            Optional labels overriding label data in ``config``.
        **config : dict
            Flat VQRS config, nested config under ``_nested_config``, or
            serialized quantifier specifications.

        Returns
        -------
        VQRS
            Configured dense VQRS model.
        """
        config = dict(config)
        lower_quantifier, upper_quantifier, _ = build_vqrs_components_from_config(
            config,
            require_explicit_components=True,
        )
        sim = similarity_matrix
        if sim is None and "similarity_matrix" in config:
            sim = np.array(config["similarity_matrix"])
        lbl = labels
        if lbl is None and "labels" in config:
            lbl = np.array(config["labels"])
        if sim is None or lbl is None:
            raise ValueError(
                "similarity_matrix and labels must be provided either in config "
                "or as arguments."
            )
        return cls(
            sim,
            lbl,
            lower_quantifier,
            upper_quantifier,
            logger=config.get("logger"),
        )

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate VQRS fuzzy-quantifier components.

        Parameters
        ----------
        **kwargs : object
            Must contain lower and upper fuzzy quantifiers.

        Raises
        ------
        ValueError
            If either quantifier has the wrong type.
        """
        lower_quantifier = kwargs.get("lb_fuzzy_quantifier")
        upper_quantifier = kwargs.get("ub_fuzzy_quantifier")
        if not isinstance(lower_quantifier, FuzzyQuantifier):
            raise ValueError(
                "fuzzy_quantifier_lower must be a valid FuzzyQuantifier instance."
            )
        if not isinstance(upper_quantifier, FuzzyQuantifier):
            raise ValueError(
                "fuzzy_quantifier_upper must be a valid FuzzyQuantifier instance."
            )

    def describe_params_detailed(self) -> dict:
        """Return detailed metadata for the configured VQRS quantifiers."""
        return {
            "lb_fuzzy_quantifier": self.lb_fuzzy_quantifier.describe_params_detailed(),
            "ub_fuzzy_quantifier": self.ub_fuzzy_quantifier.describe_params_detailed(),
        }
