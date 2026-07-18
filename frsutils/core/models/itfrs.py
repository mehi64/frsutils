# SPDX-License-Identifier: BSD-3-Clause
"""Dense NumPy reference implementation of the ITFRS model.

Backend-aware exact blockwise execution is provided by the approximation-engine
layer and exposed through the public API.
"""

import numpy as np

import frsutils.core.implicators as imp
import frsutils.core.tnorms as tn
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.itfrs_components import build_itfrs_components_from_config


@FuzzyRoughModel.register("itfrs")
class ITFRS(FuzzyRoughModel):
    """Dense Implicator-TNorm Fuzzy Rough Set approximation model.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Dense fuzzy-relation matrix using the ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Class labels aligned with the relation matrix.
    ub_tnorm : TNorm
        T-norm used to aggregate upper-approximation evidence.
    lb_implicator : Implicator
        Implicator used to aggregate lower-approximation evidence.
    logger : logging.Logger or None, default=None
        Optional logger.
    """

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        ub_tnorm: tn.TNorm,
        lb_implicator: imp.Implicator,
        logger=None,
    ):
        """Initialize a dense ITFRS reference model."""
        label_array = np.asarray(labels) if labels is not None else None
        super().__init__(similarity_matrix, label_array, logger=logger)

        self.validate_params(
            ub_tnorm=ub_tnorm,
            lb_implicator=lb_implicator,
        )
        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator
        self.logger.debug(f"{self.__class__.__name__} initialized.")

    def lower_approximation(self) -> np.ndarray:
        """Compute dense ITFRS lower-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Row-wise minimum implicator values after excluding each diagonal
            self-comparison.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 1.0)
        return np.min(implication_vals, axis=1)

    def upper_approximation(self) -> np.ndarray:
        """Compute dense ITFRS upper-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Row-wise maximum T-norm values after excluding each diagonal
            self-comparison.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        return np.max(tnorm_vals, axis=1)

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the ITFRS model.

        Parameters
        ----------
        include_data : bool, default=False
            Include ``similarity_matrix`` and ``labels`` when True.

        Returns
        -------
        dict
            Serialized component configuration and optional data arrays.
        """
        data = {
            "type": "itfrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict(),
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
    ) -> "ITFRS":
        """Reconstruct an ITFRS model from serialized configuration.

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
        ITFRS
            Reconstructed dense ITFRS model.
        """
        tnorm = tn.TNorm.from_dict(data["ub_tnorm"])
        implicator = imp.Implicator.from_dict(data["lb_implicator"])
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
        return cls(sim, lbl, tnorm, implicator, logger=logger)

    def describe_params_detailed(self) -> dict:
        """Return detailed metadata for the configured ITFRS components."""
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed(),
        }

    def _get_params(self) -> dict:
        """Return component and data parameters used by this model."""
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels,
        }

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate ITFRS component objects at construction time.

        Parameters
        ----------
        **kwargs : object
            Must contain ``ub_tnorm`` and ``lb_implicator``.

        Raises
        ------
        ValueError
            If either component has the wrong type.
        """
        tnorm = kwargs.get("ub_tnorm")
        if tnorm is None or not isinstance(tnorm, tn.TNorm):
            raise ValueError(
                "Parameter 'tnorm' must be provided and be an instance of "
                "a TNorm subclass."
            )

        implicator = kwargs.get("lb_implicator")
        if implicator is None or not isinstance(implicator, imp.Implicator):
            raise ValueError(
                "Parameter 'implicator' must be provided and be an instance of "
                "an Implicator subclass."
            )

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "ITFRS":
        """Create a dense ITFRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), optional
            Optional dense matrix overriding matrix data in ``config``.
        labels : array-like of shape (n_samples,), optional
            Optional labels overriding label data in ``config``.
        **config : dict
            Flat ITFRS config, nested config under ``_nested_config``, or
            serialized component specifications.

        Returns
        -------
        ITFRS
            Configured dense ITFRS model.
        """
        config = dict(config)
        ub_tnorm, lb_implicator = build_itfrs_components_from_config(
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
                "similarity_matrix and labels must be provided either as "
                "arguments or in the config dictionary."
            )
        return cls(
            sim,
            lbl,
            ub_tnorm,
            lb_implicator,
            logger=config.get("logger"),
        )
