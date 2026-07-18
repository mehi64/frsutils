# SPDX-License-Identifier: BSD-3-Clause
"""Dense NumPy reference implementation of the OWAFRS model.

Exact row-buffered blockwise execution is provided by the approximation-engine
layer. Optional CuPy support is limited to similarity-block generation.
"""

import numpy as np

import frsutils.core.implicators as imp
import frsutils.core.owa_weights as owa_weights
import frsutils.core.tnorms as tn
from frsutils.core.models.fuzzy_rough_model import FuzzyRoughModel
from frsutils.core.models.owafrs_components import build_owafrs_components_from_config


@FuzzyRoughModel.register("owafrs")
class OWAFRS(FuzzyRoughModel):
    """Dense OWA-based fuzzy-rough approximation model.

    Parameters
    ----------
    similarity_matrix : ndarray of shape (n_samples, n_samples)
        Dense fuzzy-relation matrix using the ``rows_are_queries`` orientation.
    labels : array-like of shape (n_samples,)
        Class labels aligned with the relation matrix.
    ub_tnorm : TNorm
        T-norm used for upper-approximation evidence.
    lb_implicator : Implicator
        Implicator used for lower-approximation evidence.
    ub_owa_method : OWAWeights
        OWA weighting strategy for upper-approximation aggregation.
    lb_owa_method : OWAWeights
        OWA weighting strategy for lower-approximation aggregation.
    logger : logging.Logger or None, default=None
        Optional logger.
    """

    def __init__(
        self,
        similarity_matrix: np.ndarray,
        labels: np.ndarray,
        ub_tnorm: tn.TNorm,
        lb_implicator: imp.Implicator,
        ub_owa_method: owa_weights.OWAWeights,
        lb_owa_method: owa_weights.OWAWeights,
        logger=None,
    ):
        """Initialize a dense OWAFRS reference model."""
        labels_array = np.asarray(labels) if labels is not None else None
        super().__init__(similarity_matrix, labels_array, logger=logger)
        self.validate_params(
            ub_tnorm=ub_tnorm,
            lb_implicator=lb_implicator,
            ub_owa_method=ub_owa_method,
            lb_owa_method=lb_owa_method,
        )
        self.ub_tnorm = ub_tnorm
        self.lb_implicator = lb_implicator
        self.ub_owa_method = ub_owa_method
        self.lb_owa_method = lb_owa_method

        n_nonself = self.labels.shape[0] - 1
        self.lb_owa_weights = lb_owa_method.weights(n_nonself, order="asc")
        self.ub_owa_weights = ub_owa_method.weights(n_nonself, order="desc")
        self.logger.debug(f"{self.__class__.__name__} initialized.")

    def lower_approximation(self) -> np.ndarray:
        """Compute dense OWAFRS lower-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            OWA-aggregated implication values after removing one diagonal
            self-comparison from each row.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        implication_vals = self.lb_implicator(self.similarity_matrix, label_mask)
        np.fill_diagonal(implication_vals, 0.0)
        sorted_matrix = np.sort(implication_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.lb_owa_weights)

    def upper_approximation(self) -> np.ndarray:
        """Compute dense OWAFRS upper-approximation values.

        Returns
        -------
        ndarray of shape (n_samples,)
            OWA-aggregated T-norm values after removing one diagonal
            self-comparison from each row.
        """
        label_mask = (self.labels[:, None] == self.labels[None, :]).astype(float)
        tnorm_vals = self.ub_tnorm(self.similarity_matrix, label_mask)
        np.fill_diagonal(tnorm_vals, 0.0)
        sorted_matrix = np.sort(tnorm_vals, axis=1)[:, ::-1][:, :-1]
        return np.matmul(sorted_matrix, self.ub_owa_weights)

    def to_dict(self, include_data: bool = False) -> dict:
        """Serialize the OWAFRS model.

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
            "type": "owafrs",
            "ub_tnorm": self.ub_tnorm.to_dict(),
            "lb_implicator": self.lb_implicator.to_dict(),
            "ub_owa_method": self.ub_owa_method.to_dict(),
            "lb_owa_method": self.lb_owa_method.to_dict(),
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
    ) -> "OWAFRS":
        """Reconstruct an OWAFRS model from serialized configuration.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.
        similarity_matrix : array-like or None, default=None
            Optional matrix override. External data take precedence over
            serialized matrix data.
        labels : array-like or None, default=None
            Optional label override. External data take precedence over
            serialized labels.
        logger : logging.Logger or None, default=None
            Optional logger.

        Returns
        -------
        OWAFRS
            Reconstructed dense OWAFRS model.
        """
        tnorm = tn.TNorm.from_dict(data["ub_tnorm"])
        implicator = imp.Implicator.from_dict(data["lb_implicator"])
        upper_owa = owa_weights.OWAWeights.from_dict(data["ub_owa_method"])
        lower_owa = owa_weights.OWAWeights.from_dict(data["lb_owa_method"])

        sim = similarity_matrix
        if sim is None and "similarity_matrix" in data:
            sim = np.array(data["similarity_matrix"])
        lbl = labels
        if lbl is None and "labels" in data:
            lbl = np.array(data["labels"])
        if sim is None or lbl is None:
            raise ValueError(
                "similarity_matrix and labels must be provided either in data "
                "or as arguments."
            )
        return cls(
            sim,
            lbl,
            tnorm,
            implicator,
            upper_owa,
            lower_owa,
            logger=logger,
        )

    def describe_params_detailed(self) -> dict:
        """Return detailed metadata for the configured OWAFRS components."""
        return {
            "ub_tnorm": self.ub_tnorm.describe_params_detailed(),
            "lb_implicator": self.lb_implicator.describe_params_detailed(),
            "ub_owa_method": self.ub_owa_method.describe_params_detailed(),
            "lb_owa_method": self.lb_owa_method.describe_params_detailed(),
        }

    @classmethod
    def validate_params(cls, **kwargs) -> None:
        """Validate OWAFRS component objects at construction time.

        Parameters
        ----------
        **kwargs : object
            Must contain the upper T-norm, lower implicator, and both OWA
            weighting strategies.

        Raises
        ------
        ValueError
            If any component has the wrong type.
        """
        tnorm = kwargs.get("ub_tnorm")
        if not isinstance(tnorm, tn.TNorm):
            raise ValueError(
                "Parameter 'tnorm' must be provided and be an instance of "
                "a TNorm subclass."
            )
        implicator = kwargs.get("lb_implicator")
        if not isinstance(implicator, imp.Implicator):
            raise ValueError(
                "Parameter 'implicator' must be provided and be an instance of "
                "an Implicator subclass."
            )
        upper_owa = kwargs.get("ub_owa_method")
        if not isinstance(upper_owa, owa_weights.OWAWeights):
            raise ValueError(
                "Parameter 'ub_owa_method' must be provided and be an instance "
                "of an OWAWeights subclass."
            )
        lower_owa = kwargs.get("lb_owa_method")
        if not isinstance(lower_owa, owa_weights.OWAWeights):
            raise ValueError(
                "Parameter 'lb_owa_method' must be provided and be an instance "
                "of an OWAWeights subclass."
            )

    def _get_params(self) -> dict:
        """Return component and data parameters used by this model."""
        return {
            "ub_tnorm": self.ub_tnorm,
            "lb_implicator": self.lb_implicator,
            "ub_owa_method": self.ub_owa_method,
            "lb_owa_method": self.lb_owa_method,
            "similarity_matrix": self.similarity_matrix,
            "labels": self.labels,
        }

    @classmethod
    def from_config(cls, similarity_matrix=None, labels=None, **config: dict) -> "OWAFRS":
        """Create a dense OWAFRS instance from flat or nested configuration.

        Parameters
        ----------
        similarity_matrix : array-like of shape (n_samples, n_samples), optional
            Optional dense matrix overriding matrix data in ``config``.
        labels : array-like of shape (n_samples,), optional
            Optional labels overriding label data in ``config``.
        **config : dict
            Flat OWAFRS config, nested config under ``_nested_config``,
            serialized component specifications, or direct component objects.

        Returns
        -------
        OWAFRS
            Configured dense OWAFRS model.
        """
        config = dict(config)
        components = build_owafrs_components_from_config(
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
        return cls(sim, lbl, *components, logger=config.get("logger"))
