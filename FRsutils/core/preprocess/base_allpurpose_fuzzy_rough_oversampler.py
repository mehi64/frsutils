""" 
@file base_allpurpose_fuzzy_rough_oversampler.py
@brief Abstract base class for oversampling using Fuzzy Rough Sets.
...
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import warnings
from imblearn.over_sampling.base import BaseOverSampler

import FRsutils.utils.validation_utils.validation_utils as valutil
from FRsutils.utils.constructor_utils.lazy_constructible_mixin import LazyConstructibleMixin

class BaseAllPurposeFuzzyRoughOversampler(LazyConstructibleMixin, ABC, BaseOverSampler):
    DEFAULT_CONFIG: Dict[str, Any] = {
        "sampling_strategy": "auto",
        "instance_ranking_strategy": "pos",
        "sampling_ratio": None,
        "type": "itfrs",
    }

    @classmethod
    def _collect_default_config(cls) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for c in reversed(cls.mro()):
            dc = getattr(c, "DEFAULT_CONFIG", None)
            if isinstance(dc, dict):
                merged.update(dc)
        return merged

    def __init__(self, **kwargs):
        cfg = self._collect_default_config()
        cfg.update(kwargs)

        super().__init__(sampling_strategy=cfg.get("sampling_strategy", "auto"))

        self.sampling_strategy = cfg.get("sampling_strategy", "auto")

        inst_rank = cfg.get("instance_ranking_strategy", "pos")
        if isinstance(inst_rank, str):
            inst_rank = valutil.validate_ranking_strategy_choice(inst_rank)
        elif not isinstance(inst_rank, dict):
            raise TypeError("instance_ranking_strategy must be a string or a dict.")
        self.instance_ranking_strategy = inst_rank

        self.sampling_ratio = cfg.get("sampling_ratio", None)

        model_type = cfg.get("type", "itfrs")
        if not isinstance(model_type, str) or not model_type.strip():
            raise TypeError("type must be a non-empty string.")
        self.type = model_type

    def _get_target_classes(self):
        if self.instance_ranking_strategy == "pos":
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            return [cls for cls in self.classes_ if cls != majority_class]
        if isinstance(self.instance_ranking_strategy, dict):
            return list(self.instance_ranking_strategy.keys())

        warnings.warn(
            f"Unsupported strategy: {self.instance_ranking_strategy}. Falling back to 'pos'."
        )
        majority_class = max(self.target_stats_, key=self.target_stats_.get)
        return [cls for cls in self.classes_ if cls != majority_class]

    def _get_num_samples(self, class_label):
        if self.sampling_ratio is None:
            majority_class = max(self.target_stats_, key=self.target_stats_.get)
            target_count = self.target_stats_[majority_class]
            return max(0, target_count - self.target_stats_[class_label])

        if isinstance(self.sampling_ratio, dict):
            key = class_label if class_label in self.sampling_ratio else str(class_label)
            if key not in self.sampling_ratio:
                raise ValueError(
                    f"sampling_ratio dict does not contain a ratio for class '{class_label}'."
                )
            ratio = self.sampling_ratio[key]
            if not isinstance(ratio, (int, float)):
                raise TypeError("sampling_ratio per-class values must be numeric.")
            aa = self.target_stats_[class_label] * ratio
        elif isinstance(self.sampling_ratio, (int, float)):
            aa = self.target_stats_[class_label] * self.sampling_ratio
        else:
            raise TypeError("sampling_ratio must be None, a number, or a dict.")

        return int(round(aa)) + 1

    def _validate_config(self, *, model_registry=None, **config):
        super()._validate_config(model_registry=model_registry, **config)
        self._check_params(model_registry=model_registry, **config)

    def configure(self, *, model_registry=None, **config):
        merged = self._collect_default_config()
        merged.update(config)

        self.sampling_strategy = merged.get("sampling_strategy", self.sampling_strategy)
        self.sampling_ratio = merged.get("sampling_ratio", self.sampling_ratio)

        if "instance_ranking_strategy" in merged:
            v = merged["instance_ranking_strategy"]
            if isinstance(v, str):
                v = valutil.validate_ranking_strategy_choice(v)
            elif not isinstance(v, dict):
                raise TypeError("instance_ranking_strategy must be a string or a dict.")
            self.instance_ranking_strategy = v

        if "type" in merged:
            if not isinstance(merged["type"], str) or not merged["type"].strip():
                raise TypeError("type must be a non-empty string.")
            self.type = merged["type"]

        return super().configure(model_registry=model_registry, **merged)

    def get_params(self, deep: bool = True):
        params: Dict[str, Any] = self._collect_default_config()

        if hasattr(self, "_object_config") and isinstance(getattr(self, "_object_config"), dict):
            params.update(self._object_config)

        params["sampling_strategy"] = getattr(self, "sampling_strategy", params.get("sampling_strategy", "auto"))
        params["instance_ranking_strategy"] = getattr(
            self, "instance_ranking_strategy", params.get("instance_ranking_strategy", "pos")
        )
        params["sampling_ratio"] = getattr(self, "sampling_ratio", params.get("sampling_ratio", None))
        params["type"] = getattr(self, "type", params.get("type", "itfrs"))

        if not deep:
            return dict(params)

        deep_params: Dict[str, Any] = {}
        for k, v in list(params.items()):
            if hasattr(v, "get_params") and callable(getattr(v, "get_params")):
                for sub_k, sub_v in v.get_params(deep=True).items():
                    deep_params[f"{k}__{sub_k}"] = sub_v
        params.update(deep_params)
        return params

    @abstractmethod
    def _check_params(self, **kwargs):
        raise NotImplementedError("_check_params must be implemented")

    @abstractmethod
    def fit_resample(self, X, y):
        raise NotImplementedError("fit_resample must be implemented")