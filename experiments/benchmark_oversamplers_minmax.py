"""
@file benchmark_oversamplers_minmax.py
@brief Benchmark FRSMOTE vs. selected baselines with incremental caching (resume across multiple runs).

This module provides a modular, pipeline-friendly benchmarking framework for oversampling methods.
It supports:
- Single KEEL .dat files (StratifiedKFold)
- Multiple KEEL .dat files from a directory (batch benchmark)
- KEEL predefined folds (single dataset folder containing *_tra.dat / *_tst.dat)
- Multiple KEEL fold-dataset folders from a root directory (batch benchmark)
- Quick sanity checks via selected sklearn datasets (optional)

##############################################
# ✅ Quick Summary of Features
# - imblearn Pipeline evaluation (scaler -> sampler -> classifier)
# - Adapter for smote-variants to behave like an imblearn sampler (fit_resample)
# - Batch benchmarking across many datasets
# - Metrics commonly used in oversampling papers:
#   Balanced Accuracy, Macro-F1, G-Mean, MCC, ROC-AUC, PR-AUC,
#   Minority-class Precision/Recall/F1 (binary/multiclass), runtime breakdown
# - Incremental execution:
#   - Stores fold-level results into a SQLite cache so you can resume across multiple days
#   - Re-aggregates final tables from cached results (no need to rerun completed folds)
# - Clean outputs:
#   (1) fold-level results CSV (exported from cache)
#   (2) dataset-level aggregated results CSV
#   (3) overall summary CSV (ranks + win/tie/loss vs FRSMOTE)
#   (4) paper-ready markdown tables
#   (5) Wilcoxon vs. FRSMOTE table
# - Nonparametric stats utilities:
#   Friedman test + Nemenyi Critical Difference (CD) for average ranks

##############################################
# ✅ Summary Table of Design Patterns
# Category                Name                Usage & Where Applied
# ----------------------------------------------------------------------------------
# Design Pattern          Adapter             SmoteVariantsSamplerAdapter, TimedSamplerAdapter
# Design Pattern          Factory-ish         build_samplers(), build_classifiers()
# Design Pattern          Strategy            Metric functions are pluggable via MetricSpec
# Architecture            Modular Pipelines   evaluate_fold(), evaluate_dataset(), summarize_*
# Clean Code              SRP, DRY, Fail-Fast, Explicit configuration via BenchmarkConfig
##############################################

##############################################
# ✅ How to Use - Examples
##############################################
# 1) Benchmark a single KEEL .dat file with StratifiedKFold:
# python -m experiments.benchmark_oversamplers --keel-dat /path/to/dataset.dat --out-dir out/
#
# 2) Benchmark MANY KEEL .dat files in a directory (non-fold files):
# python -m experiments.benchmark_oversamplers --keel-dat-dir /path/to/keel_datasets --out-dir out/
#
# 3) Benchmark a single KEEL predefined-folds dataset folder:
# python -m experiments.benchmark_oversamplers --keel-cv-dir /path/to/dataset_folds --out-dir out/
#
# 4) Benchmark MANY KEEL predefined-folds dataset folders under a root:
# python -m experiments.benchmark_oversamplers --keel-cv-root /path/to/cv_root --out-dir out/
#
# 5) Quick sklearn sanity check (binary reduced variants):
# python -m experiments.benchmark_oversamplers --sklearn breast_cancer --out-dir out/
#
# Notes:
# - Requires: numpy, pandas, scikit-learn, imbalanced-learn
# - For smote-variants baselines: pip install smote-variants
# - For stats (Friedman p-value + Nemenyi CD): scipy recommended
##############################################
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import sqlite3
import hashlib
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek


# =============================================================================
# Import helpers (FRSMOTE + KEEL utils)
# =============================================================================

def import_frsmote_class() -> type:
    """
    @brief Attempt to import FRSMOTE class from common project/module locations.

    @return FRSMOTE class.

    @exception ImportError If FRSMOTE cannot be imported from known paths.
    """
    import_errors: List[str] = []
    candidates = [
        ("fuzzy_rough_oversampling", "FRSMOTE"),
        ("FRSMOTE", "FRSMOTE"),  # legacy fallback if running with a standalone FRSMOTE.py on PYTHONPATH
    ]

    for mod_name, cls_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            return getattr(mod, cls_name)
        except Exception as e:
            import_errors.append(f"{mod_name}.{cls_name}: {e}")

    raise ImportError(
        "Could not import FRSMOTE from expected locations. Install fuzzy_rough_oversampling or check the import path. Tried:\n  - "
        + "\n  - ".join(import_errors)
    )


def import_keel_utils():
    """
    @brief Import KEEL dataset helpers from project.

    Expected symbols:
    - parse_keel_file
    - discover_keel_cv_folds
    - KeelCVLoader

    @return (parse_keel_file, discover_keel_cv_folds, KeelCVLoader)

    @exception ImportError If any required symbol cannot be imported.
    """
    import_errors: List[str] = []
    resolved: Dict[str, Any] = {}

    candidates = [
        ("FRsutils.utils.dataset_utils.KEEL_DS_loader_utility", "parse_keel_file"),
        ("FRsutils.utils.dataset_utils.KEEL_CV_Utility", "discover_keel_cv_folds"),
        ("FRsutils.utils.dataset_utils.KEEL_CV_Utility", "KeelCVLoader"),
        ("KEEL_DS_loader_utility", "parse_keel_file"),
        ("KEEL_CV_Utility", "discover_keel_cv_folds"),
        ("KEEL_CV_Utility", "KeelCVLoader"),
    ]

    for mod_name, sym in candidates:
        try:
            mod = __import__(mod_name, fromlist=[sym])
            resolved[sym] = getattr(mod, sym)
        except Exception as e:
            import_errors.append(f"{mod_name}.{sym}: {e}")

    required = ["parse_keel_file", "discover_keel_cv_folds", "KeelCVLoader"]
    if not all(k in resolved for k in required):
        raise ImportError(
            "Could not import KEEL utilities. Missing: "
            + ", ".join([k for k in required if k not in resolved])
            + "\nTried:\n  - "
            + "\n  - ".join(import_errors)
        )

    return resolved["parse_keel_file"], resolved["discover_keel_cv_folds"], resolved["KeelCVLoader"]


# =============================================================================
# Adapters: smote-variants + timing wrapper
# =============================================================================

class SmoteVariantsSamplerAdapter(BaseEstimator):
    """
    @brief Adapter to use a smote-variants oversampler inside an imblearn Pipeline.

    smote-variants convention:
      - oversampler.sample(X, y) -> (X_res, y_res)

    imblearn convention:
      - sampler.fit_resample(X, y) -> (X_res, y_res)

    @param oversampler_cls smote_variants oversampler class (callable).
    @param oversampler_kwargs Dict of kwargs passed to oversampler constructor.
    @param random_state Random seed forwarded when supported.
    """

    def __init__(
        self,
        oversampler_cls: Any,
        oversampler_kwargs: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = 42,
    ):
        self.oversampler_cls = oversampler_cls
        self.oversampler_kwargs = oversampler_kwargs
        self.random_state = random_state

        # timing
        self.last_resample_time_sec_: Optional[float] = None

    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        @brief Return parameters for sklearn clone compatibility.
        """
        return {
            "oversampler_cls": self.oversampler_cls,
            "oversampler_kwargs": self.oversampler_kwargs,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """
        @brief Set parameters for sklearn clone compatibility.
        """
        for k, v in params.items():
            setattr(self, k, v)
        return self


    def fit(self, X, y):
        """
        @brief No-op fit for sklearn compatibility.

        @return self
        """
        return self

    def fit_resample(self, X, y):
        """
        @brief Resample using smote-variants oversampler.sample(X, y).

        @param X Feature matrix.
        @param y Target vector.

        @return (X_resampled, y_resampled)
        """
        kwargs = {} if self.oversampler_kwargs is None else dict(self.oversampler_kwargs)

        # Try to pass random_state if supported by constructor
        if self.random_state is not None and "random_state" not in kwargs:
            try:
                os_obj = self.oversampler_cls(random_state=self.random_state, **kwargs)
            except TypeError:
                os_obj = self.oversampler_cls(**kwargs)
        else:
            os_obj = self.oversampler_cls(**kwargs)

        t0 = time.time()
        X_res, y_res = os_obj.sample(X, y)
        self.last_resample_time_sec_ = float(time.time() - t0)

        return np.asarray(X_res), np.asarray(y_res)



class UnitIntervalClipper(BaseEstimator):
    """
    @brief Transformer that clips features into [0, 1].

    This is a defensive step against tiny floating-point excursions beyond [0, 1]
    after MinMax scaling (or other numeric transforms). It is useful for samplers
    that strictly require unit-interval inputs (e.g., FRSMOTE).
    """

    def fit(self, X, y=None):
        """
        @brief No-op fit.
        """
        return self

    def transform(self, X):
        """
        @brief Clip X into [0, 1].

        @param X Feature matrix.
        @return clipped feature matrix.
        """
        return np.clip(X, 0.0, 1.0)


class TimedSamplerAdapter(BaseEstimator):
    """
    @brief Timing wrapper for any imblearn-like sampler.

    The wrapped object must implement fit_resample(X, y).

    Adds:
      - last_resample_time_sec_

    @param sampler The underlying sampler instance.
    """

    def __init__(self, sampler: Any):
        self.sampler = sampler
        self.last_resample_time_sec_: Optional[float] = None

    def fit(self, X, y):
        """
        @brief No-op fit.

        @return self
        """
        return self

    def fit_resample(self, X, y):
        """
        @brief Delegate resampling and measure runtime.

        @return (X_resampled, y_resampled)
        """
        t0 = time.time()
        X_res, y_res = self.sampler.fit_resample(X, y)
        self.last_resample_time_sec_ = float(time.time() - t0)
        return X_res, y_res

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        @brief sklearn compatibility.
        """
        return {"sampler": self.sampler}

    def set_params(self, **params):
        """
        @brief sklearn compatibility.
        """
        if "sampler" in params:
            self.sampler = params["sampler"]
        return self

    def __getattr__(self, item):
        """
        @brief Delegate attribute access to underlying sampler.

        @param item attribute name
        """
        return getattr(self.sampler, item)


# =============================================================================
# Metrics: commonly used in oversampling papers
# =============================================================================

@dataclass(frozen=True)
class MetricSpec:
    """
    @brief Metric specification.

    @param name Column name used in outputs.
    @param fn Callable(metric_fn(y_true, y_pred, y_score, labels, minority_label)) -> float|nan
    @param higher_is_better Whether larger metric indicates better performance.
    """
    name: str
    fn: Callable[..., float]
    higher_is_better: bool = True


def _infer_minority_label(y: np.ndarray) -> Any:
    """
    @brief Infer minority label (least frequent class) from y.

    @param y Label vector.

    @return minority label.
    """
    values, counts = np.unique(y, return_counts=True)
    return values[int(np.argmin(counts))]


def _safe_roc_auc(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> float:
    """
    @brief Compute ROC-AUC for binary or multiclass when scores available.

    - Binary: roc_auc_score(y_bin, y_score)
    - Multiclass: roc_auc_score(one-vs-rest, macro average) if y_score is 2D

    @return float or np.nan
    """
    if y_score is None:
        return float("nan")

    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            pos = sorted(classes)[-1]
            y_bin = (y_true == pos).astype(int)
            return float(roc_auc_score(y_bin, y_score))
        # multiclass: require score matrix (n_samples, n_classes)
        if y_score.ndim == 2 and y_score.shape[1] == len(classes):
            return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")

    return float("nan")


def _safe_pr_auc(y_true: np.ndarray, y_score: Optional[np.ndarray]) -> float:
    """
    @brief Compute PR-AUC (Average Precision) for binary or multiclass when scores available.

    - Binary: average_precision_score(y_bin, y_score)
    - Multiclass: average_precision_score(one-vs-rest, macro) if y_score is 2D

    @return float or np.nan
    """
    if y_score is None:
        return float("nan")

    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            pos = sorted(classes)[-1]
            y_bin = (y_true == pos).astype(int)
            return float(average_precision_score(y_bin, y_score))
        if y_score.ndim == 2 and y_score.shape[1] == len(classes):
            return float(average_precision_score(y_true, y_score, average="macro"))
    except Exception:
        return float("nan")

    return float("nan")


def build_default_metrics() -> List[MetricSpec]:
    """
    @brief Build a list of metrics commonly used in oversampling/imbalance papers.

    Included:
    - balanced_acc
    - f1_macro
    - gmean_macro (geometric mean of class recalls)
    - mcc
    - roc_auc (binary or multiclass OVR macro)
    - pr_auc (binary or multiclass macro)
    - minority_precision / minority_recall / minority_f1
    - specificity (binary only; otherwise nan)

    @return list of MetricSpec
    """
    def balanced_acc(y_true, y_pred, **_):
        return float(balanced_accuracy_score(y_true, y_pred))

    def f1_macro(y_true, y_pred, **_):
        return float(f1_score(y_true, y_pred, average="macro"))

    def gmean_macro(y_true, y_pred, **_):
        # imblearn.metrics.geometric_mean_score supports multiclass (macro)
        return float(geometric_mean_score(y_true, y_pred, average="macro"))

    def mcc(y_true, y_pred, **_):
        return float(matthews_corrcoef(y_true, y_pred))

    def roc_auc(y_true, y_pred, y_score=None, **_):
        return _safe_roc_auc(y_true, y_score)

    def pr_auc(y_true, y_pred, y_score=None, **_):
        return _safe_pr_auc(y_true, y_score)

    def minority_precision(y_true, y_pred, minority_label=None, **_):
        if minority_label is None:
            minority_label = _infer_minority_label(y_true)
        return float(precision_score(y_true, y_pred, average=None, labels=[minority_label])[0])

    def minority_recall(y_true, y_pred, minority_label=None, **_):
        if minority_label is None:
            minority_label = _infer_minority_label(y_true)
        return float(recall_score(y_true, y_pred, average=None, labels=[minority_label])[0])

    def minority_f1(y_true, y_pred, minority_label=None, **_):
        if minority_label is None:
            minority_label = _infer_minority_label(y_true)
        return float(f1_score(y_true, y_pred, average=None, labels=[minority_label])[0])

    def specificity(y_true, y_pred, **_):
        # binary only: TN/(TN+FP)
        classes = np.unique(y_true)
        if len(classes) != 2:
            return float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=sorted(classes))
        tn, fp = cm[0, 0], cm[0, 1]
        denom = tn + fp
        return float(tn / denom) if denom > 0 else float("nan")

    return [
        MetricSpec("balanced_acc", balanced_acc, True),
        MetricSpec("f1_macro", f1_macro, True),
        MetricSpec("gmean_macro", gmean_macro, True),
        MetricSpec("mcc", mcc, True),
        MetricSpec("roc_auc", roc_auc, True),
        MetricSpec("pr_auc", pr_auc, True),
        MetricSpec("minority_precision", minority_precision, True),
        MetricSpec("minority_recall", minority_recall, True),
        MetricSpec("minority_f1", minority_f1, True),
        MetricSpec("specificity", specificity, True),
    ]


# =============================================================================
# Configuration and discovery
# =============================================================================

@dataclass
class BenchmarkConfig:
    """
    @brief Benchmark configuration.

    @param random_state Random seed.
    @param n_splits StratifiedKFold splits for single-file datasets.
    @param repeats Number of repeated CV runs (with different seeds) to stabilize results.
    @param metrics List of MetricSpec.
    @param primary_metric Name of primary metric used for ranking.
    @param scalers Backward-compat switch; if False disables all scaling.
    @param scaler_before_sampler Scaling BEFORE sampler: none|minmax|standard.
    @param scaler_after_sampler Scaling AFTER sampler: none|minmax|standard.
    @param one_hot_encode_keel One-hot encoding for KEEL categorical features.
    @param normalize_keel Normalize numeric KEEL features to [0,1] before scaling.
    @param max_datasets Optional cap for batch runs.
    """
    random_state: int = 42
    n_splits: int = 5
    repeats: int = 1
    metrics: Optional[List[MetricSpec]] = None
    primary_metric: str = "balanced_acc"
    scalers: bool = True
    scaler_before_sampler: str = "minmax"   # none|minmax|standard
    scaler_after_sampler: str = "standard"  # none|minmax|standard 
    one_hot_encode_keel: bool = False
    normalize_keel: bool = True
    max_datasets: Optional[int] = None

    def ensure_metrics(self) -> "BenchmarkConfig":
        """
        @brief Ensure metrics list is populated.

        @return self
        """
        if self.metrics is None:
            self.metrics = build_default_metrics()
        return self


@dataclass(frozen=True)
class DatasetTask:
    """
    @brief A single dataset evaluation task.

    @param name Display name.
    @param kind One of: "keel_dat", "keel_cv", "sklearn"
    @param path Path to dataset (file or folder) or sklearn dataset name.
    """
    name: str
    kind: str
    path: str


def discover_keel_dat_files(dat_dir: Union[str, Path], recursive: bool = True) -> List[Path]:
    """
    @brief Discover KEEL single .dat files inside a directory.

    Filters out fold files containing 'tra.dat' or 'tst.dat'.

    @param dat_dir Directory path.
    @param recursive Whether to search recursively.
    @return list of .dat file paths.
    """
    dat_dir = Path(dat_dir)
    if not dat_dir.exists():
        raise FileNotFoundError(f"Directory not found: {dat_dir}")

    pattern = "**/*.dat" if recursive else "*.dat"
    files = [p for p in dat_dir.glob(pattern) if p.is_file()]

    # Exclude CV fold files
    filtered = []
    for p in files:
        lower = p.name.lower()
        if re.search(r"(?:_|\b)(tra|tst)\.dat$", lower):
            continue
        filtered.append(p)

    return sorted(filtered)


def discover_keel_cv_dataset_dirs(cv_root: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    @brief Discover KEEL predefined-fold dataset folders under a root directory.

    A folder is treated as a dataset-fold directory if it contains at least one '*tra.dat' and one '*tst.dat'.

    @param cv_root Root directory.
    @param recursive If True, search nested directories; otherwise only immediate children.
    @return list of dataset-fold directories.
    """
    cv_root = Path(cv_root)
    if not cv_root.exists():
        raise FileNotFoundError(f"Directory not found: {cv_root}")

    dirs = []
    candidates = cv_root.rglob("*") if recursive else cv_root.iterdir()
    for p in candidates:
        if not p.is_dir():
            continue
        files = [f.name.lower() for f in p.iterdir() if f.is_file() and f.name.lower().endswith(".dat")]
        has_tra = any("tra" in fn and fn.endswith(".dat") for fn in files)
        has_tst = any("tst" in fn and fn.endswith(".dat") for fn in files)
        if has_tra and has_tst:
            dirs.append(p)

    return sorted(dirs)


# =============================================================================
# Samplers and classifiers
# =============================================================================

def _get_sv_attr(sv_mod: Any, name: str) -> Any:
    """
    @brief Get smote_variants class by attribute name with a helpful error message.
    """
    if not hasattr(sv_mod, name):
        raise AttributeError(f"smote_variants has no attribute '{name}'. Check your smote-variants version.")
    return getattr(sv_mod, name)


def build_samplers(
    random_state: int = 42,
    frsmote_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[Any]]:
    """
    @brief Construct samplers dictionary: sampler_name -> sampler_object.

    Includes:
    - FRSMOTE (your method)
    - imbalanced-learn baselines
    - selected smote-variants baselines (adapter wrapped)

    Returned samplers are wrapped with TimedSamplerAdapter where applicable.

    @param random_state Seed for methods supporting it.
    @param frsmote_kwargs Optional kwargs merged into FRSMOTE constructor.

    @return dict of sampler name -> sampler instance (or None for "NoSampling").
    """
    frsmote_kwargs = frsmote_kwargs or {}
    FRSMOTE = import_frsmote_class()

    samplers: Dict[str, Optional[Any]] = {}

    # ---- no sampler
    samplers["NoSampling"] = None

    # ---- FRSMOTE
    samplers["FRSMOTE"] = TimedSamplerAdapter(FRSMOTE(random_state=random_state, **frsmote_kwargs))

    # ---- imbalanced-learn baselines
    samplers["SMOTE"] = TimedSamplerAdapter(SMOTE(random_state=random_state, k_neighbors=5))
    samplers["ADASYN"] = TimedSamplerAdapter(ADASYN(random_state=random_state, n_neighbors=5))
    samplers["BorderlineSMOTE"] = TimedSamplerAdapter(BorderlineSMOTE(random_state=random_state, k_neighbors=5, m_neighbors=10))
    samplers["SVMSMOTE"] = TimedSamplerAdapter(SVMSMOTE(random_state=random_state, k_neighbors=5))
    samplers["KMeansSMOTE"] = TimedSamplerAdapter(KMeansSMOTE(random_state=random_state))
    samplers["SMOTEENN"] = TimedSamplerAdapter(SMOTEENN(random_state=random_state))
    samplers["SMOTETomek"] = TimedSamplerAdapter(SMOTETomek(random_state=random_state))

    # ---- smote-variants selected baselines
    try:
        import smote_variants as sv
    except Exception as e:
        raise ImportError(
            "smote-variants is required for the requested baselines.\n"
            "Install it (e.g., pip install smote-variants) and retry.\n"
            f"Import error: {e}"
        )

    smote_variant_map = {
        "SV_Borderline_SMOTE1": "Borderline_SMOTE1",
        "SV_Borderline_SMOTE2": "Borderline_SMOTE2",
        "SV_Safe_Level_SMOTE": "Safe_Level_SMOTE",
        "SV_ProWSyn": "ProWSyn",
        "SV_MWMOTE": "MWMOTE",
        "SV_SMOTE_IPF": "SMOTE_IPF",
        "SV_DBSMOTE": "DBSMOTE",
    }

    for pretty, attr in smote_variant_map.items():
        cls = _get_sv_attr(sv, attr)
        samplers[pretty] = SmoteVariantsSamplerAdapter(cls, oversampler_kwargs={}, random_state=random_state)

    return samplers


def build_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    @brief Build classifier baselines.

    @return dict classifier name -> sklearn estimator
    """
    return {
        "LogReg": LogisticRegression(max_iter=5000),
        "SVC_RBF": SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state),
    }


# =============================================================================
# Evaluation core
# =============================================================================

@dataclass
class FoldResult:
    """
    @brief Fold-level result record.

    @param dataset Dataset name.
    @param protocol Evaluation protocol description.
    @param classifier Classifier name.
    @param sampler Sampler name.
    @param repeat Repeat index (0..repeats-1).
    @param fold Fold index.
    @param metrics Dict metric_name -> value
    @param resample_time_sec Sampler runtime, if available.
    @param fit_time_sec Total pipeline fit time.
    @param error Error message (empty if success).
    """
    dataset: str
    protocol: str
    classifier: str
    sampler: str
    repeat: int
    fold: int
    metrics: Dict[str, float]
    resample_time_sec: float
    fit_time_sec: float
    error: str




# =============================================================================
# Incremental result store (SQLite)
# =============================================================================

def compute_benchmark_id(
    cfg: BenchmarkConfig,
    sampler_names: Sequence[str],
    classifier_names: Sequence[str],
    frsmote_kwargs: Optional[Dict[str, Any]],
) -> str:
    """
    @brief Compute a short, stable benchmark id for caching.

    The id is derived from:
      - key BenchmarkConfig fields (n_splits, repeats, primary_metric, scaling modes)
      - sampler names and classifier names
      - FRSMOTE kwargs (if provided)

    @param cfg BenchmarkConfig.
    @param sampler_names List of sampler display names.
    @param classifier_names List of classifier names.
    @param frsmote_kwargs Optional FRSMOTE kwargs.

    @return short benchmark id (12 hex chars).
    """
    payload = {
        "n_splits": cfg.n_splits,
        "repeats": cfg.repeats,
        "primary_metric": cfg.primary_metric,
        "scalers": cfg.scalers,
        "scaler_before_sampler": getattr(cfg, "scaler_before_sampler", None),
        "scaler_after_sampler": getattr(cfg, "scaler_after_sampler", None),
        "samplers": list(sampler_names),
        "classifiers": list(classifier_names),
        "frsmote_kwargs": frsmote_kwargs or {},
        "metric_names": [m.name for m in (cfg.metrics or [])],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


class ResultStore:
    """
    @brief SQLite-backed cache for fold-level benchmark results.

    The store enables:
      - incremental runs across multiple days
      - skipping already-computed fold results
      - regenerating final reports (aggregated CSV / tables / stats) from cache

    Schema:
      fold_results(
        benchmark_id TEXT,
        dataset TEXT,
        protocol TEXT,
        classifier TEXT,
        sampler TEXT,
        repeat INTEGER,
        fold INTEGER,
        resample_time_sec REAL,
        fit_time_sec REAL,
        error TEXT,
        created_at TEXT,
        <metric columns ...>,
        PRIMARY KEY(benchmark_id, dataset, protocol, classifier, sampler, repeat, fold)
      )

    @param db_path Path to SQLite db file.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        """
        @brief Open connection lazily.

        @return sqlite3.Connection
        """
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        return self._conn

    def close(self) -> None:
        """
        @brief Close db connection.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def init_schema(self) -> None:
        """
        @brief Create base table if missing.
        """
        con = self.connect()
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS fold_results (
                benchmark_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                protocol TEXT NOT NULL,
                classifier TEXT NOT NULL,
                sampler TEXT NOT NULL,
                repeat INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                resample_time_sec REAL,
                fit_time_sec REAL,
                error TEXT,
                created_at TEXT,
                PRIMARY KEY (benchmark_id, dataset, protocol, classifier, sampler, repeat, fold)
            );
            """
        )
        con.commit()

    def _existing_columns(self) -> List[str]:
        con = self.connect()
        rows = con.execute("PRAGMA table_info(fold_results);").fetchall()
        return [r[1] for r in rows]

    def ensure_metric_columns(self, metric_names: Sequence[str]) -> None:
        """
        @brief Ensure metric columns exist (REAL), adding them dynamically if needed.

        @param metric_names metric names to store as columns.
        """
        con = self.connect()
        cols = set(self._existing_columns())
        for m in metric_names:
            if m not in cols:
                con.execute(f'ALTER TABLE fold_results ADD COLUMN "{m}" REAL;')
        con.commit()

    def has_record(
        self,
        benchmark_id: str,
        dataset: str,
        protocol: str,
        classifier: str,
        sampler: str,
        repeat: int,
        fold: int,
    ) -> bool:
        """
        @brief Check whether a fold record already exists.

        @return True if exists.
        """
        con = self.connect()
        row = con.execute(
            """
            SELECT 1 FROM fold_results
            WHERE benchmark_id=? AND dataset=? AND protocol=? AND classifier=? AND sampler=? AND repeat=? AND fold=?
            LIMIT 1;
            """,
            (benchmark_id, dataset, protocol, classifier, sampler, int(repeat), int(fold)),
        ).fetchone()
        return row is not None

    def upsert_fold_result(
        self,
        benchmark_id: str,
        row: FoldResult,
        overwrite: bool,
    ) -> None:
        """
        @brief Insert or update a fold result.

        @param benchmark_id benchmark id.
        @param row FoldResult
        @param overwrite If True, replaces existing row; otherwise keeps existing.
        """
        con = self.connect()

        created_at = datetime.now(timezone.utc).isoformat()

        # Flatten
        flat: Dict[str, Any] = {
            "benchmark_id": benchmark_id,
            "dataset": row.dataset,
            "protocol": row.protocol,
            "classifier": row.classifier,
            "sampler": row.sampler,
            "repeat": int(row.repeat),
            "fold": int(row.fold),
            "resample_time_sec": float(row.resample_time_sec) if row.resample_time_sec is not None else None,
            "fit_time_sec": float(row.fit_time_sec) if row.fit_time_sec is not None else None,
            "error": row.error or "",
            "created_at": created_at,
        }
        for k, v in row.metrics.items():
            flat[k] = None if v is None or not np.isfinite(v) else float(v)

        cols = list(flat.keys())
        placeholders = ",".join(["?"] * len(cols))
        col_sql = ",".join([f'"{c}"' for c in cols])

        if overwrite:
            # SQLite UPSERT
            update_cols = [c for c in cols if c not in ("benchmark_id", "dataset", "protocol", "classifier", "sampler", "repeat", "fold")]
            set_sql = ",".join([f'"{c}"=excluded."{c}"' for c in update_cols])
            sql = f"""
            INSERT INTO fold_results ({col_sql}) VALUES ({placeholders})
            ON CONFLICT(benchmark_id, dataset, protocol, classifier, sampler, repeat, fold)
            DO UPDATE SET {set_sql};
            """
            con.execute(sql, [flat[c] for c in cols])
        else:
            # INSERT OR IGNORE
            sql = f"""
            INSERT OR IGNORE INTO fold_results ({col_sql}) VALUES ({placeholders});
            """
            con.execute(sql, [flat[c] for c in cols])

        con.commit()

    def load_fold_dataframe(self, benchmark_id: str) -> pd.DataFrame:
        """
        @brief Load all fold results for a given benchmark id into a DataFrame.

        @param benchmark_id benchmark id.
        @return DataFrame
        """
        con = self.connect()
        df = pd.read_sql_query(
            "SELECT * FROM fold_results WHERE benchmark_id=?;",
            con,
            params=(benchmark_id,),
        )
        return df

    def list_benchmark_ids(self) -> List[str]:
        """
        @brief List benchmark ids present in the store.

        @return list of benchmark ids.
        """
        con = self.connect()
        rows = con.execute("SELECT DISTINCT benchmark_id FROM fold_results ORDER BY benchmark_id;").fetchall()
        return [r[0] for r in rows]


def _get_scores(pipe: Pipeline, X_test: np.ndarray) -> Optional[np.ndarray]:
    """
    @brief Get continuous scores for AUC metrics.

    Preference:
      - predict_proba
      - decision_function

    Returns:
      - Binary: 1D scores
      - Multiclass: 2D score matrix if predict_proba available
      - None if unavailable
    """
    clf = pipe.steps[-1][1]
    if hasattr(clf, "predict_proba"):
        try:
            proba = pipe.predict_proba(X_test)
            return np.asarray(proba)
        except Exception:
            pass
    if hasattr(clf, "decision_function"):
        try:
            dec = pipe.decision_function(X_test)
            return np.asarray(dec)
        except Exception:
            pass
    return None


def evaluate_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sampler: Optional[Any],
    clf: Any,
    metrics: Sequence[MetricSpec],
    use_scaler: bool = True,
    scaler_before_sampler: str = "minmax",
    scaler_after_sampler: str = "standard",
) -> Tuple[Dict[str, float], float, float]:
    """
    @brief Fit (scaler -> sampler -> classifier) and compute metrics for one fold.

    @param X_train Train features.
    @param y_train Train labels.
    @param X_test Test features.
    @param y_test Test labels.
    @param sampler Sampler instance with fit_resample, or None (no sampling).
    @param clf Classifier estimator.
    @param metrics Metrics to compute.
    @param use_scaler Backward-compat flag; if False disables all scaling.
    @param scaler_before_sampler Scaling BEFORE sampler: none|minmax|standard.
    @param scaler_after_sampler Scaling AFTER sampler: none|minmax|standard.

    @return (metric_dict, resample_time_sec, fit_time_sec)
    """
    steps = []
    if use_scaler:
        sb = (scaler_before_sampler or "none").lower().strip()
        sa = (scaler_after_sampler or "none").lower().strip()

        if sb == "minmax":
            steps.append(("pre_scaler", MinMaxScaler(clip=True)))
            # Defensive clip to satisfy strict [0,1] samplers (e.g., FRSMOTE)
            steps.append(("pre_clip", UnitIntervalClipper()))
        elif sb == "standard":
            steps.append(("pre_scaler", StandardScaler()))

    if sampler is not None:
        steps.append(("sampler", sampler))

    if use_scaler:
        sa = (scaler_after_sampler or "none").lower().strip()
        if sa == "minmax":
            steps.append(("post_scaler", MinMaxScaler(clip=True)))
        elif sa == "standard":
            steps.append(("post_scaler", StandardScaler()))

    steps.append(("clf", clf))

    pipe = Pipeline(steps)

    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = float(time.time() - t0)

    y_pred = pipe.predict(X_test)

    # try to fetch scores for AUC metrics
    y_score = _get_scores(pipe, X_test)

    # infer minority based on train set (more appropriate for imbalance)
    minority_label = _infer_minority_label(y_train)

    out: Dict[str, float] = {}
    for ms in metrics:
        try:
            out[ms.name] = float(ms.fn(
                y_true=y_test,
                y_pred=y_pred,
                y_score=y_score,
                labels=np.unique(y_train),
                minority_label=minority_label,
            ))
        except Exception:
            out[ms.name] = float("nan")

    # resample runtime if available
    resample_time = float("nan")
    if sampler is not None:
        # TimedSamplerAdapter stores it on wrapper; SmoteVariantsSamplerAdapter stores on itself
        resample_time = getattr(pipe.named_steps.get("sampler", None), "last_resample_time_sec_", float("nan"))
        if resample_time is None:
            resample_time = float("nan")

    return out, float(resample_time), fit_time


def evaluate_stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    samplers: Dict[str, Optional[Any]],
    classifiers: Dict[str, Any],
    cfg: BenchmarkConfig,
    store: Optional[ResultStore] = None,
    benchmark_id: Optional[str] = None,
    skip_existing: bool = True,
    overwrite_existing: bool = False,
) -> List[FoldResult]:
    """
    @brief Evaluate all sampler+classifier pairs using StratifiedKFold on a dataset.

    @param X Feature matrix.
    @param y Labels.
    @param dataset_name Display name.
    @param samplers Sampler dict.
    @param classifiers Classifier dict.
    @param cfg BenchmarkConfig.

    @return list of FoldResult
    """
    cfg = cfg.ensure_metrics()
    protocol = f"StratifiedKFold(n_splits={cfg.n_splits})"
    results: List[FoldResult] = []

    for rep in range(cfg.repeats):
        rep_seed = cfg.random_state + rep
        cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=rep_seed)

        for clf_name, clf0 in classifiers.items():
            for sampler_name, sampler0 in samplers.items():
                for fold_i, (tr, te) in enumerate(cv.split(X, y)):
                    # Incremental caching: skip already computed folds
                    if store is not None and benchmark_id is not None and skip_existing:
                        if store.has_record(benchmark_id, dataset_name, protocol, clf_name, sampler_name, rep, fold_i):
                            continue

                    X_tr, y_tr = X[tr], y[tr]
                    X_te, y_te = X[te], y[te]

                    clf = clone(clf0)
                    sampler = None if sampler0 is None else clone(sampler0)

                    try:
                        md, rs_t, ft = evaluate_fold(
                            X_tr, y_tr, X_te, y_te,
                            sampler=sampler,
                            clf=clf,
                            metrics=cfg.metrics or [],
                            use_scaler=cfg.scalers,
                            scaler_before_sampler=cfg.scaler_before_sampler,
                            scaler_after_sampler=cfg.scaler_after_sampler,
                        )
                        fr = FoldResult(
                            dataset=dataset_name,
                            protocol=protocol,
                            classifier=clf_name,
                            sampler=sampler_name,
                            repeat=rep,
                            fold=fold_i,
                            metrics=md,
                            resample_time_sec=rs_t,
                            fit_time_sec=ft,
                            error="",
                        )
                        results.append(fr)
                        if store is not None and benchmark_id is not None:
                            store.upsert_fold_result(benchmark_id, fr, overwrite=overwrite_existing)
                    except Exception as e:
                        fr = FoldResult(
                            dataset=dataset_name,
                            protocol=protocol,
                            classifier=clf_name,
                            sampler=sampler_name,
                            repeat=rep,
                            fold=fold_i,
                            metrics={m.name: float("nan") for m in (cfg.metrics or [])},
                            resample_time_sec=float("nan"),
                            fit_time_sec=float("nan"),
                            error=f"{type(e).__name__}: {e}",
                        )
                        results.append(fr)
                        if store is not None and benchmark_id is not None:
                            store.upsert_fold_result(benchmark_id, fr, overwrite=overwrite_existing)
                        continue

    return results


def evaluate_keel_predefined_folds(
    keel_cv_dir: Union[str, Path],
    dataset_name: str,
    samplers: Dict[str, Optional[Any]],
    classifiers: Dict[str, Any],
    cfg: BenchmarkConfig,
    store: Optional[ResultStore] = None,
    benchmark_id: Optional[str] = None,
    skip_existing: bool = True,
    overwrite_existing: bool = False,
) -> List[FoldResult]:
    """
    @brief Evaluate all sampler+classifier pairs on a KEEL predefined-folds dataset folder.

    Expects a directory containing fold files: *_tra.dat and *_tst.dat.

    @param keel_cv_dir Dataset fold directory.
    @param dataset_name Display name.
    @param samplers Sampler dict.
    @param classifiers Classifier dict.
    @param cfg BenchmarkConfig.

    @return list of FoldResult
    """
    cfg = cfg.ensure_metrics()
    protocol = "KEEL_PredefinedFolds"

    parse_keel_file, discover_keel_cv_folds, KeelCVLoader = import_keel_utils()
    fold_pairs = discover_keel_cv_folds(str(keel_cv_dir))
    loader = KeelCVLoader(fold_pairs, one_hot_encode=cfg.one_hot_encode_keel, normalize=cfg.normalize_keel)

    results: List[FoldResult] = []

    for rep in range(cfg.repeats):
        # repeats on predefined folds: affects only stochastic samplers/classifiers
        rep_seed = cfg.random_state + rep

        for clf_name, clf0 in classifiers.items():
            for sampler_name, sampler0 in samplers.items():
                for fold_i in range(loader.get_n_splits()):
                    # Incremental caching: skip already computed folds
                    if store is not None and benchmark_id is not None and skip_existing:
                        if store.has_record(benchmark_id, dataset_name, protocol, clf_name, sampler_name, rep, fold_i):
                            continue

                    try:
                        X_train_df, y_train_s, X_test_df, y_test_s = loader.load_fold(fold_i)

                        X_tr = X_train_df.to_numpy(dtype=float)
                        y_tr = y_train_s.to_numpy()
                        X_te = X_test_df.to_numpy(dtype=float)
                        y_te = y_test_s.to_numpy()

                        # clone and re-seed if supported
                        clf = clone(clf0)
                        sampler = None if sampler0 is None else clone(sampler0)

                        # attempt to set random_state on sampler/clf if it exists (best-effort)
                        if hasattr(clf, "random_state"):
                            try:
                                clf.set_params(random_state=rep_seed)
                            except Exception:
                                pass
                        if sampler is not None and hasattr(sampler, "random_state"):
                            try:
                                sampler.set_params(random_state=rep_seed)
                            except Exception:
                                pass

                        md, rs_t, ft = evaluate_fold(
                            X_tr, y_tr, X_te, y_te,
                            sampler=sampler,
                            clf=clf,
                            metrics=cfg.metrics or [],
                            use_scaler=cfg.scalers,
                            scaler_before_sampler=cfg.scaler_before_sampler,
                            scaler_after_sampler=cfg.scaler_after_sampler,
                        )
                        fr = FoldResult(
                            dataset=dataset_name,
                            protocol=protocol,
                            classifier=clf_name,
                            sampler=sampler_name,
                            repeat=rep,
                            fold=fold_i,
                            metrics=md,
                            resample_time_sec=rs_t,
                            fit_time_sec=ft,
                            error="",
                        )
                        results.append(fr)
                        if store is not None and benchmark_id is not None:
                            store.upsert_fold_result(benchmark_id, fr, overwrite=overwrite_existing)
                    except Exception as e:
                        fr = FoldResult(
                            dataset=dataset_name,
                            protocol=protocol,
                            classifier=clf_name,
                            sampler=sampler_name,
                            repeat=rep,
                            fold=fold_i,
                            metrics={m.name: float("nan") for m in (cfg.metrics or [])},
                            resample_time_sec=float("nan"),
                            fit_time_sec=float("nan"),
                            error=f"{type(e).__name__}: {e}",
                        )
                        results.append(fr)
                        if store is not None and benchmark_id is not None:
                            store.upsert_fold_result(benchmark_id, fr, overwrite=overwrite_existing)
                        continue

    return results


# =============================================================================
# Dataset loaders
# =============================================================================

def load_keel_single_dat(dat_path: Union[str, Path], one_hot_encode: bool, normalize: bool) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    @brief Load a single KEEL .dat file.

    @param dat_path Path to KEEL .dat file.
    @param one_hot_encode One-hot encode categorical input features.
    @param normalize Normalize numeric features to [0,1] (KEEL ranges).

    @return (dataset_name, X, y)
    """
    parse_keel_file, _, _ = import_keel_utils()
    meta, df, input_cols, _ = parse_keel_file(str(dat_path), one_hot_encode=one_hot_encode, normalize=normalize)
    ds_name = meta.get("relation") or Path(dat_path).stem
    X = df[input_cols].to_numpy(dtype=float)
    y = df.drop(columns=input_cols).iloc[:, 0].to_numpy()
    return ds_name, X, y


def load_sklearn_dataset(name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """
    @brief Load a small sklearn dataset for quick benchmarking.

    Supported:
    - breast_cancer (binary)
    - wine (binary: class 2 vs rest)
    - iris (binary: setosa vs rest)
    - digits (binary: 0 vs rest)

    @return (dataset_name, X, y)
    """
    from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits

    key = name.lower().strip()
    if key == "breast_cancer":
        ds = load_breast_cancer()
        return f"sklearn:{key}", ds.data, ds.target

    if key == "wine":
        ds = load_wine()
        y = (ds.target == 2).astype(int)
        return f"sklearn:{key}", ds.data, y

    if key == "iris":
        ds = load_iris()
        y = (ds.target == 0).astype(int)
        return f"sklearn:{key}", ds.data, y

    if key == "digits":
        ds = load_digits()
        y = (ds.target == 0).astype(int)
        return f"sklearn:{key}", ds.data, y

    raise ValueError(f"Unknown sklearn dataset: {name}")


# =============================================================================
# Result conversion and summarization
# =============================================================================

def fold_results_to_dataframe(rows: List[FoldResult]) -> pd.DataFrame:
    """
    @brief Convert fold-level results to a flat DataFrame.

    Columns:
    - dataset, protocol, classifier, sampler, repeat, fold
    - metrics...
    - resample_time_sec, fit_time_sec, error

    @param rows FoldResult list.

    @return pandas DataFrame
    """
    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        d = {
            "dataset": r.dataset,
            "protocol": r.protocol,
            "classifier": r.classifier,
            "sampler": r.sampler,
            "repeat": r.repeat,
            "fold": r.fold,
            "resample_time_sec": r.resample_time_sec,
            "fit_time_sec": r.fit_time_sec,
            "error": r.error,
        }
        for k, v in r.metrics.items():
            d[k] = v
        out_rows.append(d)
    return pd.DataFrame(out_rows)


def aggregate_dataset_level(df_folds: pd.DataFrame, metric_names: Sequence[str]) -> pd.DataFrame:
    """
    @brief Aggregate fold-level results into dataset-level mean/std.

    Grouping keys:
      - dataset, protocol, classifier, sampler

    Output columns:
      <metric>_mean, <metric>_std
      resample_time_mean_sec, resample_time_std_sec
      fit_time_mean_sec, fit_time_std_sec
      error_rate

    @param df_folds Fold-level dataframe.
    @param metric_names metric column names to aggregate.
    """
    keys = ["dataset", "protocol", "classifier", "sampler"]

    def _agg(group: pd.DataFrame) -> pd.Series:
        # only successful folds for means/stats
        ok = group[group["error"].fillna("") == ""]
        n_all = len(group)
        n_ok = len(ok)

        data: Dict[str, Any] = {
            "n_folds_total": n_all,
            "n_folds_ok": n_ok,
            "error_rate": float((n_all - n_ok) / n_all) if n_all > 0 else float("nan"),
        }

        for m in metric_names:
            vals = ok[m].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            data[f"{m}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            data[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))

        for col, base in [("resample_time_sec", "resample_time"), ("fit_time_sec", "fit_time")]:
            vals = ok[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            data[f"{base}_mean_sec"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            data[f"{base}_std_sec"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))

        return pd.Series(data)

    agg = df_folds.groupby(keys, dropna=False).apply(_agg).reset_index()
    return agg


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    """
    @brief Format mean±std for paper tables.

    @param mean Mean value.
    @param std Std value.
    @param digits Decimal digits.
    """
    if not np.isfinite(mean):
        return "NA"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


# =============================================================================
# Ranking + win/tie/loss + Friedman/Nemenyi
# =============================================================================

def compute_rank_matrix(
    df_dataset: pd.DataFrame,
    primary_metric: str,
    higher_is_better: bool = True,
    block_cols: Sequence[str] = ("dataset", "classifier"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    @brief Compute per-block ranks for each sampler, using dataset-level aggregated means.

    @param df_dataset dataset-level aggregated results (output of aggregate_dataset_level).
    @param primary_metric metric base name (e.g., "balanced_acc") for ranking.
    @param higher_is_better If True, higher metric = better (rank 1).
    @param block_cols Columns that define a "block" (typically dataset + classifier).
    @return (rank_matrix, avg_ranks)

    rank_matrix:
      index = blocks
      columns = sampler
      values = rank (1 best)

    avg_ranks:
      average rank per sampler across blocks
    """
    mcol = f"{primary_metric}_mean"
    if mcol not in df_dataset.columns:
        raise KeyError(f"Primary metric column not found: {mcol}")

    blocks = df_dataset[list(block_cols)].drop_duplicates().reset_index(drop=True)
    samplers = sorted(df_dataset["sampler"].unique().tolist())

    rank_rows = []
    block_index = []

    for _, b in blocks.iterrows():
        cond = np.ones(len(df_dataset), dtype=bool)
        for c in block_cols:
            cond &= (df_dataset[c] == b[c]).to_numpy()
        sub = df_dataset.loc[cond, ["sampler", mcol]].copy()

        # align order
        sub = sub.set_index("sampler").reindex(samplers)

        values = sub[mcol].to_numpy(dtype=float)
        # treat nan as worst
        if higher_is_better:
            # higher better => rank by -value
            keyvals = np.where(np.isfinite(values), -values, np.inf)
        else:
            keyvals = np.where(np.isfinite(values), values, np.inf)

        # ranks with average tie handling
        order = np.argsort(keyvals, kind="mergesort")
        ranks = np.empty_like(keyvals, dtype=float)
        ranks[order] = np.arange(1, len(order) + 1, dtype=float)

        # tie averaging
        # group equal keyvals (including inf)
        for kv in np.unique(keyvals):
            idx = np.where(keyvals == kv)[0]
            if len(idx) > 1:
                ranks[idx] = float(np.mean(ranks[idx]))

        rank_rows.append(ranks.tolist())
        block_index.append(tuple(b[c] for c in block_cols))

    rank_matrix = pd.DataFrame(rank_rows, columns=samplers, index=pd.MultiIndex.from_tuples(block_index, names=list(block_cols)))
    avg_ranks = rank_matrix.mean(axis=0)
    return rank_matrix, avg_ranks


def win_tie_loss_vs_reference(
    df_dataset: pd.DataFrame,
    primary_metric: str,
    ref_sampler: str = "FRSMOTE",
    higher_is_better: bool = True,
    block_cols: Sequence[str] = ("dataset", "classifier"),
    tie_eps: float = 1e-12,
) -> pd.DataFrame:
    """
    @brief Compute win/tie/loss counts per sampler vs a reference sampler across blocks.

    A "block" is typically (dataset, classifier). For each block, compare sampler metric to reference.

    @param df_dataset dataset-level aggregated results.
    @param primary_metric base metric name (e.g., "balanced_acc").
    @param ref_sampler reference sampler name (default: FRSMOTE).
    @param higher_is_better If True, higher is better.
    @param block_cols block-defining columns.
    @param tie_eps threshold for tie.

    @return dataframe with columns: sampler, wins, ties, losses, n_blocks
    """
    mcol = f"{primary_metric}_mean"
    blocks = df_dataset[list(block_cols)].drop_duplicates()
    samplers = sorted(df_dataset["sampler"].unique().tolist())

    rows = []
    for s in samplers:
        wins = ties = losses = 0
        n = 0
        for _, b in blocks.iterrows():
            cond = np.ones(len(df_dataset), dtype=bool)
            for c in block_cols:
                cond &= (df_dataset[c] == b[c]).to_numpy()

            sub = df_dataset.loc[cond, ["sampler", mcol]].set_index("sampler")
            if ref_sampler not in sub.index or s not in sub.index:
                continue

            a = float(sub.loc[s, mcol])
            r = float(sub.loc[ref_sampler, mcol])
            if not (np.isfinite(a) and np.isfinite(r)):
                continue

            n += 1
            diff = a - r
            if abs(diff) <= tie_eps:
                ties += 1
            else:
                better = diff > 0 if higher_is_better else diff < 0
                if better:
                    wins += 1
                else:
                    losses += 1

        rows.append({"sampler": s, "wins": wins, "ties": ties, "losses": losses, "n_blocks": n})
    return pd.DataFrame(rows).sort_values(["wins", "ties"], ascending=False).reset_index(drop=True)


def friedman_test_from_ranks(rank_matrix: pd.DataFrame) -> Tuple[float, float]:
    """
    @brief Friedman test (Demsar-style) computed from rank matrix.

    Uses the Friedman chi-square approximation:
      chi2 = (12*N)/(k*(k+1)) * (sum(R_j^2) - k*(k+1)^2/4)

    where:
      N = number of blocks (datasets or dataset+classifier)
      k = number of algorithms (samplers)
      R_j = average rank of algorithm j

    @param rank_matrix rows=blocks, cols=samplers with per-block ranks.

    @return (chi2_stat, p_value) where p_value is chi-square survival with df=k-1
            If scipy not available, p_value is np.nan.
    """
    N = rank_matrix.shape[0]
    k = rank_matrix.shape[1]
    Rj = rank_matrix.mean(axis=0).to_numpy(dtype=float)

    chi2 = (12.0 * N) / (k * (k + 1.0)) * (np.sum(Rj ** 2) - (k * (k + 1.0) ** 2) / 4.0)

    p = float("nan")
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, df=k - 1))
    except Exception:
        p = float("nan")

    return float(chi2), p


def nemenyi_critical_difference(
    n_blocks: int,
    k_algorithms: int,
    alpha: float = 0.05,
) -> float:
    """
    @brief Compute Nemenyi Critical Difference (CD) for average ranks.

    CD = q_alpha * sqrt(k*(k+1)/(6*N))

    Per Demšar (2006), q_alpha is based on the studentized range statistic divided by sqrt(2).
    We approximate q_alpha using SciPy studentized_range if available.

    @param n_blocks N (datasets or dataset+classifier blocks)
    @param k_algorithms k (number of compared methods)
    @param alpha significance level (default 0.05)

    @return CD value or np.nan if scipy is unavailable.
    """
    try:
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(1.0 - alpha, k_algorithms, np.inf)) / np.sqrt(2.0)
        cd = q * np.sqrt((k_algorithms * (k_algorithms + 1.0)) / (6.0 * n_blocks))
        return float(cd)
    except Exception:
        return float("nan")




def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """
    @brief Holm-Bonferroni p-value adjustment.

    @param p_values List of raw p-values.
    @return adjusted p-values in original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    for i, idx in enumerate(order):
        adj[idx] = min(1.0, (m - i) * p[idx])
    # enforce monotonicity
    for i in range(1, m):
        prev = order[i - 1]
        cur = order[i]
        adj[cur] = max(adj[cur], adj[prev])
    return adj.tolist()


def wilcoxon_vs_reference(
    df_dataset: pd.DataFrame,
    primary_metric: str,
    ref_sampler: str = "FRSMOTE",
    block_cols: Sequence[str] = ("dataset", "classifier"),
) -> pd.DataFrame:
    """
    @brief Compute Wilcoxon signed-rank test vs a reference sampler across blocks.

    For each sampler S != ref:
      - collects paired primary_metric means over blocks where both S and ref exist
      - runs scipy.stats.wilcoxon if available (two-sided)
      - reports n_pairs, mean_diff, wins/ties/losses vs ref (block-level)

    @param df_dataset Dataset-level aggregated results.
    @param primary_metric Metric base name (e.g., 'balanced_acc').
    @param ref_sampler Reference sampler.
    @param block_cols Block definition (default dataset+classifier)

    @return DataFrame with columns:
      sampler, n_pairs, mean_diff, wins, ties, losses, wilcoxon_stat, p_value, p_value_holm
    """
    mcol = f"{primary_metric}_mean"
    blocks = df_dataset[list(block_cols)].drop_duplicates()
    samplers = sorted([s for s in df_dataset["sampler"].unique().tolist() if s != ref_sampler])

    results = []
    raw_p = []

    # best-effort wilcoxon
    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except Exception:
        _wilcoxon = None

    for s in samplers:
        diffs = []
        wins = ties = losses = 0

        for _, b in blocks.iterrows():
            cond = np.ones(len(df_dataset), dtype=bool)
            for c in block_cols:
                cond &= (df_dataset[c] == b[c]).to_numpy()
            sub = df_dataset.loc[cond, ["sampler", mcol]].set_index("sampler")

            if ref_sampler not in sub.index or s not in sub.index:
                continue

            a = float(sub.loc[s, mcol])
            r = float(sub.loc[ref_sampler, mcol])
            if not (np.isfinite(a) and np.isfinite(r)):
                continue

            d = a - r
            diffs.append(d)
            if abs(d) < 1e-12:
                ties += 1
            elif d > 0:
                wins += 1
            else:
                losses += 1

        diffs = np.asarray(diffs, dtype=float)
        n_pairs = int(len(diffs))
        mean_diff = float(np.mean(diffs)) if n_pairs else float("nan")

        stat = float("nan")
        pval = float("nan")
        if _wilcoxon is not None and n_pairs >= 1:
            # Wilcoxon requires at least one non-zero diff for a meaningful statistic; handle all-zero
            if np.all(np.abs(diffs) < 1e-12):
                stat = 0.0
                pval = 1.0
            else:
                try:
                    w = _wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode="auto")
                    stat = float(w.statistic)
                    pval = float(w.pvalue)
                except Exception:
                    stat = float("nan")
                    pval = float("nan")

        results.append({
            "sampler": s,
            "n_pairs": n_pairs,
            "mean_diff": mean_diff,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_stat": stat,
            "p_value": pval,
        })
        raw_p.append(pval)

    df = pd.DataFrame(results)
    # Holm adjustment on finite p-values; keep NaNs as NaN
    finite_mask = np.isfinite(df["p_value"].to_numpy(dtype=float))
    adj = [float("nan")] * len(df)
    if finite_mask.any():
        fin_p = df.loc[finite_mask, "p_value"].to_numpy(dtype=float).tolist()
        fin_adj = holm_adjust(fin_p)
        j = 0
        for i, ok in enumerate(finite_mask.tolist()):
            if ok:
                adj[i] = fin_adj[j]
                j += 1
    df["p_value_holm"] = adj
    return df.sort_values(["p_value", "sampler"], ascending=[True, True]).reset_index(drop=True)


# =============================================================================
# Paper-ready tables
# =============================================================================

def build_overall_summary(
    df_dataset: pd.DataFrame,
    cfg: BenchmarkConfig,
    ref_sampler: str = "FRSMOTE",
    block_cols: Sequence[str] = ("dataset", "classifier"),
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    @brief Build overall summary table for paper:
    - mean±std of primary metric across blocks
    - average rank (primary metric)
    - win/tie/loss vs reference sampler
    - Friedman test + Nemenyi CD

    @param df_dataset dataset-level aggregated results.
    @param cfg BenchmarkConfig (must include metrics).
    @param ref_sampler Reference sampler for W/T/L counts.
    @param block_cols block definition for ranking/statistics.

    @return (summary_df, stats_dict)
    """
    cfg = cfg.ensure_metrics()
    metric_specs = {m.name: m for m in (cfg.metrics or [])}
    if cfg.primary_metric not in metric_specs:
        raise KeyError(f"Primary metric not in metrics list: {cfg.primary_metric}")

    higher_is_better = metric_specs[cfg.primary_metric].higher_is_better
    mcol = f"{cfg.primary_metric}_mean"

    # primary metric mean±std across blocks (dataset+classifier)
    blocks = df_dataset[list(block_cols)].drop_duplicates()
    samplers = sorted(df_dataset["sampler"].unique().tolist())

    rows = []
    for s in samplers:
        vals = []
        for _, b in blocks.iterrows():
            cond = np.ones(len(df_dataset), dtype=bool)
            for c in block_cols:
                cond &= (df_dataset[c] == b[c]).to_numpy()
            sub = df_dataset.loc[cond, ["sampler", mcol]].set_index("sampler")
            if s not in sub.index:
                continue
            v = float(sub.loc[s, mcol])
            if np.isfinite(v):
                vals.append(v)
        mean = float(np.mean(vals)) if vals else float("nan")
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if len(vals) == 1 else float("nan"))
        rows.append({"sampler": s, f"{cfg.primary_metric}_overall_mean": mean, f"{cfg.primary_metric}_overall_std": std})

    df_overall = pd.DataFrame(rows)

    # ranks
    rank_matrix, avg_ranks = compute_rank_matrix(df_dataset, cfg.primary_metric, higher_is_better, block_cols=block_cols)
    df_ranks = avg_ranks.reset_index()
    df_ranks.columns = ["sampler", "avg_rank"]

    # W/T/L vs FRSMOTE
    df_wtl = win_tie_loss_vs_reference(df_dataset, cfg.primary_metric, ref_sampler=ref_sampler, higher_is_better=higher_is_better, block_cols=block_cols)

    # stats: Friedman + CD
    chi2, p = friedman_test_from_ranks(rank_matrix)
    cd = nemenyi_critical_difference(n_blocks=rank_matrix.shape[0], k_algorithms=rank_matrix.shape[1], alpha=0.05)

    stats = {
        "n_blocks": int(rank_matrix.shape[0]),
        "k_algorithms": int(rank_matrix.shape[1]),
        "friedman_chi2": chi2,
        "friedman_p_value": p,
        "nemenyi_cd_alpha_0.05": cd,
        "block_cols": list(block_cols),
        "primary_metric": cfg.primary_metric,
        "ref_sampler": ref_sampler,
    }

    summary = df_overall.merge(df_ranks, on="sampler", how="left").merge(df_wtl, on="sampler", how="left")
    # best first: avg_rank ascending
    summary = summary.sort_values(["avg_rank", f"{cfg.primary_metric}_overall_mean"], ascending=[True, False]).reset_index(drop=True)
    return summary, stats


def summary_to_markdown_table(
    summary_df: pd.DataFrame,
    cfg: BenchmarkConfig,
    digits: int = 4,
) -> str:
    """
    @brief Convert overall summary DataFrame to a markdown table.

    Includes:
      - primary metric mean±std
      - avg_rank
      - wins/ties/losses

    @param summary_df summary dataframe from build_overall_summary.
    @param cfg BenchmarkConfig
    @param digits formatting digits

    @return markdown string
    """
    pm = cfg.primary_metric
    mean_col = f"{pm}_overall_mean"
    std_col = f"{pm}_overall_std"

    rows = []
    header = ["Sampler", f"{pm} (mean±std)", "AvgRank", "W", "T", "L", "N"]
    rows.append("| " + " | ".join(header) + " |")
    rows.append("|" + "|".join(["---"] * len(header)) + "|")

    for _, r in summary_df.iterrows():
        metric_txt = format_mean_std(float(r.get(mean_col, np.nan)), float(r.get(std_col, np.nan)), digits=digits)
        rows.append("| " + " | ".join([
            str(r["sampler"]),
            metric_txt,
            f'{float(r.get("avg_rank", np.nan)):.3f}' if np.isfinite(r.get("avg_rank", np.nan)) else "NA",
            str(int(r.get("wins", 0))) if pd.notna(r.get("wins", np.nan)) else "NA",
            str(int(r.get("ties", 0))) if pd.notna(r.get("ties", np.nan)) else "NA",
            str(int(r.get("losses", 0))) if pd.notna(r.get("losses", np.nan)) else "NA",
            str(int(r.get("n_blocks", 0))) if pd.notna(r.get("n_blocks", np.nan)) else "NA",
        ]) + " |")

    return "\n".join(rows)


# =============================================================================
# End-to-end runners
# =============================================================================

def run_task(task: DatasetTask, samplers: Dict[str, Optional[Any]], classifiers: Dict[str, Any], cfg: BenchmarkConfig,
            store: Optional[ResultStore] = None, benchmark_id: Optional[str] = None,
            skip_existing: bool = True, overwrite_existing: bool = False) -> List[FoldResult]:
    """
    @brief Run a single DatasetTask and return fold-level results.

    @param task DatasetTask
    @param samplers Samplers dict
    @param classifiers Classifiers dict
    @param cfg BenchmarkConfig

    @return list of FoldResult
    """
    cfg = cfg.ensure_metrics()

    if task.kind == "keel_dat":
        ds_name, X, y = load_keel_single_dat(task.path, one_hot_encode=cfg.one_hot_encode_keel, normalize=cfg.normalize_keel)
        name = task.name or ds_name
        return evaluate_stratified_kfold(X, y, name, samplers, classifiers, cfg,
                                         store=store, benchmark_id=benchmark_id,
                                         skip_existing=skip_existing, overwrite_existing=overwrite_existing)

    if task.kind == "keel_cv":
        name = task.name or Path(task.path).name
        return evaluate_keel_predefined_folds(task.path, name, samplers, classifiers, cfg,
                                             store=store, benchmark_id=benchmark_id,
                                             skip_existing=skip_existing, overwrite_existing=overwrite_existing)

    if task.kind == "sklearn":
        name, X, y = load_sklearn_dataset(task.path)
        return evaluate_stratified_kfold(X, y, task.name or name, samplers, classifiers, cfg)

    raise ValueError(f"Unknown task kind: {task.kind}")


def export_reports_from_store(
    store: ResultStore,
    benchmark_id: str,
    out_dir: Union[str, Path],
    cfg: BenchmarkConfig,
) -> Dict[str, Any]:
    """
    @brief Export final reports (CSVs/tables/stats) from cached fold-level results.

    @param store ResultStore (SQLite cache).
    @param benchmark_id Benchmark id to export.
    @param out_dir Output directory.
    @param cfg BenchmarkConfig

    @return dict with keys: df_folds, df_dataset, summary_df, wilcoxon_df, stats, output_paths
    """
    cfg = cfg.ensure_metrics()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_folds = store.load_fold_dataframe(benchmark_id)

    # keep column order similar to fold_results_to_dataframe()
    # (database already stores flattened metric columns)
    metric_names = [m.name for m in (cfg.metrics or [])]
    df_dataset = aggregate_dataset_level(df_folds.rename(columns={"resample_time_sec": "resample_time_sec", "fit_time_sec": "fit_time_sec"}), metric_names)

    summary_df, stats = build_overall_summary(
        df_dataset,
        cfg=cfg,
        ref_sampler="FRSMOTE",
        block_cols=("dataset", "classifier"),
    )
    md_table = summary_to_markdown_table(summary_df, cfg=cfg)

    wilcoxon_df = wilcoxon_vs_reference(
        df_dataset=df_dataset,
        primary_metric=cfg.primary_metric,
        ref_sampler="FRSMOTE",
        block_cols=("dataset", "classifier"),
    )

    # save outputs
    paths = {
        "store_db": store.db_path,
        "folds_csv": str(out_dir / "results_folds.csv"),
        "dataset_csv": str(out_dir / "results_dataset.csv"),
        "summary_csv": str(out_dir / "summary_overall.csv"),
        "paper_md": str(out_dir / "paper_table.md"),
        "wilcoxon_csv": str(out_dir / "wilcoxon_vs_frsmote.csv"),
        "stats_json": str(out_dir / "stats.json"),
    }

    df_folds.to_csv(paths["folds_csv"], index=False)
    df_dataset.to_csv(paths["dataset_csv"], index=False)
    summary_df.to_csv(paths["summary_csv"], index=False)
    wilcoxon_df.to_csv(paths["wilcoxon_csv"], index=False)
    (out_dir / "paper_table.md").write_text(md_table, encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[BenchmarkID] {benchmark_id}  (store: {store.db_path})")
    print("\n=== Paper-ready overall table ===\n")
    print(md_table)
    print("\n=== Wilcoxon vs FRSMOTE (primary metric) ===\n")
    print(wilcoxon_df.head(20).to_string(index=False))
    print("\n=== Stats ===\n")
    print(json.dumps(stats, indent=2))

    return {
        "df_folds": df_folds,
        "df_dataset": df_dataset,
        "summary_df": summary_df,
        "wilcoxon_df": wilcoxon_df,
        "stats": stats,
        "output_paths": paths,
    }


def run_benchmark(
    tasks: List[DatasetTask],
    out_dir: Union[str, Path],
    cfg: BenchmarkConfig,
    frsmote_kwargs: Optional[Dict[str, Any]] = None,
    store_db: Optional[Union[str, Path]] = None,
    benchmark_id: Optional[str] = None,
    only_aggregate: bool = False,
    skip_existing: bool = True,
    overwrite_existing: bool = False,
) -> Dict[str, Any]:
    """
    @brief Run benchmark incrementally and export final reports.

    This function supports long-running benchmarks by caching fold-level results in a SQLite db.
    You can run the script multiple times (even on different days) and it will:
      - reuse cached folds (skip_existing=True)
      - compute only missing folds
      - regenerate final reports from the cache

    Outputs (inside out_dir):
      - benchmark_store.sqlite       SQLite cache (if store_db not provided)
      - results_folds.csv            fold-level results (exported from cache)
      - results_dataset.csv          dataset-level mean/std
      - summary_overall.csv          overall summary (ranks + WTL vs FRSMOTE)
      - paper_table.md               paper-ready markdown table
      - wilcoxon_vs_frsmote.csv      Wilcoxon signed-rank vs FRSMOTE (primary metric)
      - stats.json                   Friedman + Nemenyi CD report

    @param tasks list of DatasetTask
    @param out_dir output directory
    @param cfg BenchmarkConfig
    @param frsmote_kwargs optional FRSMOTE kwargs
    @param store_db SQLite db path (default: out_dir/benchmark_store.sqlite)
    @param benchmark_id benchmark id (default: computed from config+methods)
    @param only_aggregate if True, do not run evaluation; only export from cache
    @param skip_existing if True, skip folds already in cache
    @param overwrite_existing if True, overwrite cached folds (only if you want to rerun)
    """
    cfg = cfg.ensure_metrics()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samplers = build_samplers(random_state=cfg.random_state, frsmote_kwargs=frsmote_kwargs)
    classifiers = build_classifiers(random_state=cfg.random_state)

    sampler_names = list(samplers.keys())
    classifier_names = list(classifiers.keys())

    if benchmark_id is None:
        benchmark_id = compute_benchmark_id(cfg, sampler_names, classifier_names, frsmote_kwargs)

    if store_db is None:
        store_db = out_dir / "benchmark_store.sqlite"

    store = ResultStore(store_db)
    store.init_schema()
    store.ensure_metric_columns([m.name for m in (cfg.metrics or [])])

    try:
        if not only_aggregate:
            # cap datasets if requested
            if cfg.max_datasets is not None:
                tasks = tasks[: int(cfg.max_datasets)]

            for i, t in enumerate(tasks, start=1):
                print(f"[{i}/{len(tasks)}] Running task: {t.kind} | {t.name} | {t.path}")
                run_task(
                    t,
                    samplers,
                    classifiers,
                    cfg,
                    store=store,
                    benchmark_id=benchmark_id,
                    skip_existing=skip_existing,
                    overwrite_existing=overwrite_existing,
                )

        # Export final reports from store
        return export_reports_from_store(store, benchmark_id, out_dir, cfg)

    finally:
        store.close()
# =============================================================================
# CLI
# =============================================================================

def build_tasks_from_args(args: argparse.Namespace) -> List[DatasetTask]:
    """
    @brief Build dataset tasks list from CLI arguments.
    """
    tasks: List[DatasetTask] = []

    if args.keel_dat:
        tasks.append(DatasetTask(name=args.dataset_name or "", kind="keel_dat", path=args.keel_dat))
        return tasks

    if args.keel_dat_dir:
        files = discover_keel_dat_files(args.keel_dat_dir, recursive=True)
        for p in files:
            tasks.append(DatasetTask(name=p.stem, kind="keel_dat", path=str(p)))
        return tasks

    if args.keel_cv_dir:
        tasks.append(DatasetTask(name=args.dataset_name or Path(args.keel_cv_dir).name, kind="keel_cv", path=args.keel_cv_dir))
        return tasks

    if args.keel_cv_root:
        dirs = discover_keel_cv_dataset_dirs(args.keel_cv_root, recursive=False)
        for d in dirs:
            tasks.append(DatasetTask(name=d.name, kind="keel_cv", path=str(d)))
        return tasks

    if args.sklearn:
        tasks.append(DatasetTask(name=args.dataset_name or f"sklearn:{args.sklearn}", kind="sklearn", path=args.sklearn))
        return tasks

    raise ValueError("No dataset source provided.")


def main():
    """
    @brief CLI entry point.
    """
    ap = argparse.ArgumentParser(description="Benchmark FRSMOTE vs imblearn + smote-variants baselines.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory for results files.")
    ap.add_argument("--store-db", type=str, default="", help="Optional SQLite cache path. Default: <out-dir>/benchmark_store.sqlite")
    ap.add_argument("--benchmark-id", type=str, default="", help="Optional benchmark id. If empty, computed from config+methods.")
    ap.add_argument("--only-aggregate", action="store_true", help="Only export reports from cache (no new evaluation).")
    # ap.add_argument("--skip-existing", action="store_true", help="Skip folds already stored in cache (default behavior).")
    # ap.add_argument("--overwrite-existing", action="store_true", help="Overwrite cached folds (recompute even if present).")
    ap.add_argument("--list-benchmarks", action="store_true", help="List benchmark ids found in cache and exit.")

    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--keel-dat", type=str, help="Path to a single KEEL .dat file.")
    group.add_argument("--keel-dat-dir", type=str, help="Directory containing multiple KEEL .dat files (non-fold).")
    group.add_argument("--keel-cv-dir", type=str, help="Directory containing a single KEEL dataset folds (*_tra.dat/*_tst.dat).")
    group.add_argument("--keel-cv-root", type=str, help="Root directory containing multiple dataset-fold directories.")
    group.add_argument("--sklearn", type=str, help="Sklearn dataset: breast_cancer|wine|iris|digits")

    ap.add_argument("--dataset-name", type=str, default="", help="Optional display name for a single-dataset run.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1, help="Repeat CV multiple times with different seeds.")
    ap.add_argument("--primary-metric", type=str, default="balanced_acc", help="Primary metric for ranking.")
    ap.add_argument("--no-scaler", action="store_true", help="Disable all scaling (pre/post).")
    ap.add_argument("--scaler-before", type=str, default="minmax", choices=["none","minmax","standard"],
                    help="Scaling BEFORE sampler: none|minmax|standard. Default=minmax (good for FRSMOTE).")
    ap.add_argument("--scaler-after", type=str, default="standard", choices=["none","minmax","standard"],
                    help="Scaling AFTER sampler: none|minmax|standard. Default=standard (good for SVC/LogReg).")

    ap.add_argument("--keel-one-hot", action="store_true", help="Enable one-hot encoding for KEEL categorical features.")
    ap.add_argument("--keel-no-normalize", action="store_true", help="Disable KEEL numeric normalization to [0,1].")

    ap.add_argument("--max-datasets", type=int, default=None, help="Limit number of datasets in batch runs.")

    ap.add_argument(
        "--frsmote-config-json",
        type=str,
        default="",
        help="Optional JSON dict of FRSMOTE kwargs (merged into constructor). Example: '{\"type\":\"itfrs\",\"k_neighbors\":5}'",
    )

    # Incremental runs / resume support
    ap.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, skips folds that already exist in saved results (incremental runs).",
    )
    ap.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="If set, recompute and overwrite existing fold results (implies --no-skip-existing).",
    )

    args = ap.parse_args()

    # overwrite_existing = bool(getattr(args, "overwrite_existing", False))
    # skip_existing_flag = bool(getattr(args, "skip_existing", True))

   

    # Cache options
    store_db = args.store_db.strip() if getattr(args, 'store_db', '') else ''
    benchmark_id = args.benchmark_id.strip() if getattr(args, 'benchmark_id', '') else ''
    only_aggregate = bool(getattr(args, 'only_aggregate', False))
    skip_existing_flag = bool(getattr(args, 'skip_existing', False))
    # skip_existing = True if (not skip_existing_flag and not overwrite_existing) else skip_existing_flag
    overwrite_existing = bool(getattr(args, 'overwrite_existing', False))

    # If overwrite is requested, we must NOT skip existing
    skip_existing = skip_existing_flag and (not overwrite_existing)

    # List benchmarks and exit
    if bool(getattr(args, 'list_benchmarks', False)):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = store_db if store_db else str(out_dir / 'benchmark_store.sqlite')
        store = ResultStore(db_path)
        store.init_schema()
        ids = store.list_benchmark_ids()
        store.close()
        print('Benchmark IDs in cache:')
        for bid in ids:
            print('  -', bid)
        return


    # If not aggregating-only, enforce a dataset source is provided
    if (not only_aggregate) and (not bool(getattr(args, 'list_benchmarks', False))):
        if not any([
            getattr(args, "keel_dat", None),
            getattr(args, "keel_dat_dir", None),
            getattr(args, "keel_cv_dir", None),
            getattr(args, "keel_cv_root", None),
            getattr(args, "sklearn", None),
        ]):
            ap.error("You must provide one dataset source argument unless --only-aggregate or --list-benchmarks is used.")

    frsmote_kwargs: Optional[Dict[str, Any]] = None
    if args.frsmote_config_json.strip():
        frsmote_kwargs = json.loads(args.frsmote_config_json)

    cfg = BenchmarkConfig(
        random_state=args.random_state,
        n_splits=args.n_splits,
        repeats=args.repeats,
        metrics=None,
        primary_metric=args.primary_metric,
        scalers=(not args.no_scaler),
        scaler_before_sampler=getattr(args, "scaler_before", "minmax"),
        scaler_after_sampler=getattr(args, "scaler_after", "standard"),
        one_hot_encode_keel=args.keel_one_hot,
        normalize_keel=(not args.keel_no_normalize),
        max_datasets=args.max_datasets,
    ).ensure_metrics()

    tasks = [] if only_aggregate else build_tasks_from_args(args)
    run_benchmark(
        tasks,
        out_dir=args.out_dir,
        cfg=cfg,
        frsmote_kwargs=frsmote_kwargs,
        store_db=(store_db if store_db else None),
        benchmark_id=(benchmark_id if benchmark_id else None),
        only_aggregate=only_aggregate,
        skip_existing=skip_existing,
        overwrite_existing=overwrite_existing,
    )


if __name__ == "__main__":
    main()