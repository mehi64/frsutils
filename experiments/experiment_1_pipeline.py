# """
# @file
# @brief Run multiple oversampling methods on KEEL 5-fold CV datasets and evaluate classifiers using pipelines.
# """

# import os
# import numpy as np

# # print(np.__config__.show())

# import pandas as pd
# from typing import List, Dict, Any
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
#                              recall_score, f1_score, confusion_matrix)
# from sklearn.preprocessing import LabelEncoder

# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import SMOTE, ADASYN
# # from smote_variants import distance_SMOTE
# from fuzzy_rough_oversampling import FRSMOTE
# from FRsutils.core.models.itfrs import ITFRS
# import FRsutils.core.tnorms as tnorms
# import FRsutils.core.implicators as implicators
# import FRsutils.core.tnorms as tnorms
# import FRsutils.core.implicators as implicators
# from FRsutils.core.similarities import GaussianSimilarity, calculate_similarity_matrix
# from FRsutils.utils.dataset_utils.KEEL_CV_Utility import discover_keel_cv_folds, KeelCVLoader


# # ------------------------------
# # Configurable Parameters
# # ------------------------------
# CLASSIFIERS = {
#     'SVC': SVC(probability=True),
#     'KNN': KNeighborsClassifier(n_neighbors=5)
# }

# OVERSAMPLERS = {
#     'SMOTE': SMOTE(),
#     'ADASYN': ADASYN(),
#     # 'distance_SMOTE': distance_SMOTE(),
#     'FRSMOTE': 'FRSMOTE'  # Custom handling in code below
# }

# # User-defined datasets and root
# DATASET_ROOT = "./datasets/KEEL/imbalanced"
# DATASET_NAMES = ['ecoli3-5-fold', 'iris0-5-fold']  # Edit to include/exclude datasets

# # ------------------------------
# # Metric Calculation Function
# # ------------------------------
# def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
#     """
#     @brief Computes evaluation metrics from predictions.
#     @return Dictionary of evaluation scores.
#     """
#     labels = list(np.unique(y_true))
#     if len(labels) != 2:
#         raise ValueError("Binary classification expected. Found labels: {}".format(labels))

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
#     return {
#         'accuracy': accuracy_score(y_true, y_pred),
#         'precision': precision_score(y_true, y_pred, pos_label=labels[1], zero_division=0),
#         'recall': recall_score(y_true, y_pred, pos_label=labels[1]),
#         'f1_score': f1_score(y_true, y_pred, pos_label=labels[1]),
#         'auc': roc_auc_score(y_true, y_prob),
#         'gmean': np.sqrt(tp / (tp + fn) * tn / (tn + fp)),
#         'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
#     }

# # ------------------------------
# # Main Experiment Runner
# # ------------------------------
# def run_experiments():
#     """
#     @brief Main function to run all oversamplers, datasets, classifiers using pipelines.
#     @return Results DataFrame.
#     """
#     results = []

#     for dataset_name in DATASET_NAMES:
#         dataset_path = os.path.join(DATASET_ROOT, dataset_name)
#         fold_paths = discover_keel_cv_folds(dataset_path)
#         cv_loader = KeelCVLoader(fold_paths, 
#                                  one_hot_encode=False,
#                                  normalize=True)

#         for fold_idx in range(cv_loader.get_n_splits()):
#             X_train, y_train, X_test, y_test = cv_loader.load_fold_as_array(fold_idx)
#             # X_train = X_train_df.to_numpy()
#             # y_train = y_train_s.to_numpy()
            
#             # X_test = X_test_df.to_numpy()
#             # y_test = y_test_s.to_numpy()
            
#             # le = LabelEncoder()
#             # y_train = le.fit_transform(y_train)
#             # y_test = le.transform(y_test)

#             for sampler_name, sampler in OVERSAMPLERS.items():
#                 for clf_name, clf in CLASSIFIERS.items():
#                     try:
#                         if sampler_name == 'FRSMOTE':
#                             sampler = FRSMOTE(
#                                 fr_model=ITFRS(
#                                     similarity_matrix=calculate_similarity_matrix(
#                                         X_train, GaussianSimilarity(), tnorms.MinTNorm()),
#                                     labels=y_train,
#                                     tnorm=tnorms.MinTNorm(),
#                                     implicator=implicators.imp_gaines),
#                                 sampling_strategy='auto')

#                         pipeline = ImbPipeline([
#                             ('sampler', sampler),
#                             ('classifier', clf)
#                         ])

#                         pipeline.fit(X_train, y_train)
#                         y_pred = pipeline.predict(X_test)
#                         y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

#                         metrics = compute_metrics(y_test, y_pred, y_prob)
#                         results.append({
#                             'dataset': dataset_name,
#                             'fold': fold_idx,
#                             'oversampler': sampler_name,
#                             'classifier': clf_name,
#                             'set': 'test',
#                             **metrics
#                         })
#                     except Exception as e:
#                         print(f"[Error] {sampler_name} + {clf_name} on {dataset_name}, fold {fold_idx}: {str(e)}")
#                         continue

#     return pd.DataFrame(results)


# if __name__ == "__main__":
#     df_results = run_experiments()
#     df_results.to_csv("all_fold_results_pipeline.csv", index=False)
#     print("Experiment completed and saved to 'all_fold_results.csv'")
































###################################################################################################
#$
###################################################################################################

# """
# @file demo_small_oversampling_pipeline.py
# @brief Demo script to run FRSMOTE / SMOTE / ADASYN inside an imbalanced-learn Pipeline on a small toy dataset.

# ##############################################
# # ✅ What this script does
# # 1) Creates a small imbalanced binary classification dataset
# # 2) Builds 4 pipelines:
# #    - Baseline (no oversampling)
# #    - SMOTE
# #    - ADASYN
# #    - FRSMOTE (from this project)
# # 3) Fits each pipeline and prints:
# #    - class distribution before/after resampling (train only)
# #    - Balanced Accuracy + F1 (binary)
# #    - Classification report
# #
# # ✅ Notes
# # - FRSMOTE in this project validates that X must be float and in [0, 1],
# #   so MinMaxScaler is required before FRSMOTE.
# ##############################################

# @example
# # From project root:
# # python experiments/demo_small_oversampling_pipeline.py
# """

# from __future__ import annotations

# from collections import Counter
# from dataclasses import dataclass
# from typing import Dict, Tuple

# import numpy as np

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import SMOTE, ADASYN

# # ✅ Your project oversampler
# from fuzzy_rough_oversampling import FRSMOTE


# @dataclass(frozen=True)
# class ExperimentResult:
#     """@brief Container for evaluation results."""
#     name: str
#     before_counts: Counter
#     after_counts: Counter
#     bal_acc: float
#     f1: float


# def make_toy_imbalanced_dataset(
#     n_samples: int = 200,
#     minority_ratio: float = 0.10,
#     random_state: int = 42,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     @brief Create a small toy binary imbalanced dataset.

#     @param n_samples Number of samples
#     @param minority_ratio Fraction of minority class
#     @param random_state Random seed
#     @return (X, y) where X is float64 and y is int64
#     """
#     weights = [1.0 - minority_ratio, minority_ratio]
#     X, y = make_classification(
#         n_samples=n_samples,
#         n_features=10,
#         n_informative=6,
#         n_redundant=2,
#         n_repeated=0,
#         n_classes=2,
#         weights=weights,
#         class_sep=1.0,
#         flip_y=0.01,
#         random_state=random_state,
#     )
#     return X.astype(np.float64), y.astype(np.int64)


# def build_frsmote_sampler(random_state: int = 42) -> FRSMOTE:
#     """
#     @brief Build a ready-to-use FRSMOTE sampler by providing full config via set_params().

#     IMPORTANT:
#     FRSMOTE in this project uses LazyConstructibleMixin; it must receive a non-empty config
#     (including 'type') via set_params() before fit/fit_resample.

#     @param random_state Random seed
#     @return Configured FRSMOTE instance
#     """
#     config: Dict = {
#         # ---- fuzzy-rough model selection ----
#         "type": "itfrs",  # ITFRS model
#         "similarity": "gaussian",
#         "sigma": 0.25,
#         "similarity_tnorm": "minimum",

#         # ---- ITFRS model parameters ----
#         "ub_tnorm_name": "minimum",
#         "lb_implicator_name": "lukasiewicz",

#         # ---- FRSMOTE / oversampling parameters ----
#         "k_neighbors": 5,
#         "bias_interpolation": True,
#         "random_state": random_state,

#         # NOTE: in current FRSMOTE implementation, sampling_ratio drives the amount of generation.
#         # If sampling_ratio is None -> oversample to majority count.
#         "sampling_ratio": None,

#         # Which classes to oversample:
#         # 'pos' -> oversample all non-majority classes (good for binary)
#         "instance_ranking_strategy": "pos",

#         # Kept for compatibility with BaseOverSampler interface / metadata
#         "sampling_strategy": "auto",
#     }

#     # FRSMOTE() starts UNCONFIGURED; set_params() will call configure(...)
#     return FRSMOTE().set_params(**config)


# def build_pipeline(name: str, sampler) -> ImbPipeline:
#     """
#     @brief Build an imblearn Pipeline: MinMaxScaler -> sampler (optional) -> LogisticRegression

#     @param name Name of pipeline (for logging)
#     @param sampler Oversampler or None
#     @return ImbPipeline
#     """
#     clf = LogisticRegression(max_iter=2000, n_jobs=None)

#     if sampler is None:
#         return ImbPipeline(steps=[
#             ("scaler", MinMaxScaler()),
#             ("clf", clf),
#         ])

#     return ImbPipeline(steps=[
#         ("scaler", MinMaxScaler()),
#         ("sampler", sampler),
#         ("clf", clf),
#     ])


# def resample_train_counts(pipeline: ImbPipeline, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Counter, Counter]:
#     """
#     @brief Compute class counts before/after resampling on train set only.

#     @param pipeline A pipeline with (scaler, sampler, clf) or (scaler, clf)
#     @param X_train Train features
#     @param y_train Train labels
#     @return (before_counts, after_counts)
#     """
#     before = Counter(y_train.tolist())

#     # No sampler -> unchanged
#     if "sampler" not in pipeline.named_steps:
#         return before, before

#     scaler = pipeline.named_steps["scaler"]
#     sampler = pipeline.named_steps["sampler"]

#     Xs = scaler.fit_transform(X_train)
#     Xr, yr = sampler.fit_resample(Xs, y_train)
#     after = Counter(yr.tolist())
#     return before, after


# def fit_and_eval(name: str, pipeline: ImbPipeline, X_train, X_test, y_train, y_test) -> ExperimentResult:
#     """
#     @brief Fit pipeline and evaluate on test set.

#     @param name Experiment name
#     @param pipeline ImbPipeline
#     @param X_train Train features
#     @param X_test Test features
#     @param y_train Train labels
#     @param y_test Test labels
#     @return ExperimentResult
#     """
#     # For correct "after resampling" counts, use a fresh sampler object each time
#     before_counts, after_counts = resample_train_counts(pipeline, X_train, y_train)

#     pipeline.fit(X_train, y_train)
#     y_pred = pipeline.predict(X_test)

#     bal_acc = float(balanced_accuracy_score(y_test, y_pred))
#     f1 = float(f1_score(y_test, y_pred))

#     print("\n" + "=" * 70)
#     print(f"[{name}]")
#     print(f"Train counts: before={dict(before_counts)} after={dict(after_counts)}")
#     print(f"Balanced Accuracy: {bal_acc:.4f}")
#     print(f"F1 (binary):       {f1:.4f}")
#     print("\nClassification report:")
#     print(classification_report(y_test, y_pred, digits=4))

#     return ExperimentResult(
#         name=name,
#         before_counts=before_counts,
#         after_counts=after_counts,
#         bal_acc=bal_acc,
#         f1=f1,
#     )


# def main() -> None:
#     """@brief Entry point."""
#     X, y = make_toy_imbalanced_dataset(n_samples=200, minority_ratio=0.10, random_state=42)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.30, random_state=42, stratify=y
#     )

#     # Important: build NEW sampler objects (don’t reuse fitted FRSMOTE across runs)
#     experiments = [
#         ("Baseline", build_pipeline("Baseline", sampler=None)),
#         ("SMOTE", build_pipeline("SMOTE", sampler=SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42))),
#         ("ADASYN", build_pipeline("ADASYN", sampler=ADASYN(sampling_strategy="auto", n_neighbors=5, random_state=42))),
#         ("FRSMOTE", build_pipeline("FRSMOTE", sampler=build_frsmote_sampler(random_state=42))),
#     ]

#     results = []
#     for name, pipe in experiments:
#         results.append(fit_and_eval(name, pipe, X_train, X_test, y_train, y_test))

#     print("\n" + "=" * 70)
#     print("Summary (higher is better):")
#     for r in results:
#         print(f"- {r.name:8s} | bal_acc={r.bal_acc:.4f} | f1={r.f1:.4f} | train_after={dict(r.after_counts)}")


# if __name__ == "__main__":
#     main()

"""
@file demo_multi_dataset_oversampling_pipeline.py
@brief Run Baseline / SMOTE / ADASYN / FRSMOTE(ITFRS, OWAFRS) on 3 small datasets using imbalanced-learn Pipeline.

##############################################
# ✅ What this script does
# - Loads 3 datasets (toy, iris-binary, breast-cancer) and forces a stronger imbalance
# - Runs multiple pipelines on each dataset:
#   Baseline, SMOTE, ADASYN, FRSMOTE variants (ITFRS with different tnorms + OWAFRS with different OWA settings)
# - Prints train class counts before/after resampling (train only) + test metrics
#
# ✅ Why MinMaxScaler is before samplers?
# - Your fuzzy-rough validation requires X in [0,1] float, so scaling is mandatory for FRSMOTE
#
# ✅ How to customize FRSMOTE configs?
# - Edit FRSMOTE_VARIANTS dict below (recommended)
# - Or override via CLI:
#   --set key=value              (applies to ALL FRSMOTE variants)
#   --set-variant NAME:key=value (applies only to that variant)
#
# ✅ Example
# python experiments/demo_multi_dataset_oversampling_pipeline.py
# python experiments/demo_multi_dataset_oversampling_pipeline.py --datasets toy,iris,cancer
# python experiments/demo_multi_dataset_oversampling_pipeline.py --set k_neighbors=7 --set sigma=0.2
# python experiments/demo_multi_dataset_oversampling_pipeline.py --set-variant itfrs_yager_p2:p=3
##############################################
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np

# --- Make project importable when running as a script ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN

# ✅ Your project oversampler
from fuzzy_rough_oversampling import FRSMOTE


# =========================================================
# Dataset utilities
# =========================================================

@dataclass(frozen=True)
class DatasetSpec:
    """@brief A dataset specification (name + loader)."""
    name: str
    loader: Callable[[int], Tuple[np.ndarray, np.ndarray]]  # random_state -> (X, y)


def _force_binary_and_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    minority_ratio: float = 0.12,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Force a stronger binary imbalance by downsampling the minority class.

    Keeps the majority class as-is, and downsamples the minority class so that:
        minority_count ≈ minority_ratio / (1 - minority_ratio) * majority_count

    @param X Feature matrix
    @param y Label vector (binary)
    @param minority_ratio Desired minority ratio in final dataset (approx.)
    @param random_state RNG seed
    @return (X_new, y_new) imbalanced
    """
    rng = np.random.RandomState(random_state)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("This helper expects binary labels.")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    maj_idx = np.where(y == majority_class)[0]
    min_idx = np.where(y == minority_class)[0]

    maj_count = len(maj_idx)
    target_min_count = int(round((minority_ratio / max(1e-9, (1.0 - minority_ratio))) * maj_count))
    target_min_count = max(2, min(target_min_count, len(min_idx)))  # keep at least 2

    chosen_min = rng.choice(min_idx, size=target_min_count, replace=False)
    keep_idx = np.concatenate([maj_idx, chosen_min])
    rng.shuffle(keep_idx)

    return X[keep_idx].astype(np.float64), y[keep_idx].astype(np.int64)


def load_toy_dataset(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Small synthetic dataset with class imbalance.

    @param random_state RNG seed
    @return (X, y)
    """
    X, y = make_classification(
        n_samples=240,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.88, 0.12],
        class_sep=1.0,
        flip_y=0.01,
        random_state=random_state,
    )
    return X.astype(np.float64), y.astype(np.int64)


def load_iris_binary_imbalanced(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Iris -> binary task (setosa vs non-setosa) + stronger imbalance.

    @param random_state RNG seed
    @return (X, y) with labels in {0,1}
    """
    data = load_iris()
    X = data.data.astype(np.float64)
    y = (data.target == 0).astype(np.int64)  # setosa=1, others=0 (binary)
    X, y = _force_binary_and_imbalance(X, y, minority_ratio=0.12, random_state=random_state)
    return X, y


def load_breast_cancer_binary_imbalanced(random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    @brief Breast-cancer -> binary task + stronger imbalance.

    @param random_state RNG seed
    @return (X, y)
    """
    data = load_breast_cancer()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)  # already binary
    X, y = _force_binary_and_imbalance(X, y, minority_ratio=0.12, random_state=random_state)
    return X, y


# =========================================================
# FRSMOTE config variants (EDIT HERE)
# =========================================================

FRSMOTE_COMMON: Dict[str, Any] = {
    # --- Similarity matrix ---
    "similarity": "gaussian",
    "sigma": 0.25,
    "similarity_tnorm": "minimum",   # across features (not the model tnorm)

    # --- Oversampling knobs ---
    "k_neighbors": 5,
    "bias_interpolation": True,
    "random_state": 42,
    "sampling_strategy": "auto",
    "instance_ranking_strategy": "pos",
    "sampling_ratio": None,          # None -> up to majority count
}

FRSMOTE_VARIANTS: Dict[str, Dict[str, Any]] = {
    # --------------------------
    # ITFRS variants (different ub_tnorm)
    # --------------------------
    "itfrs_minimum": {
        "type": "itfrs",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
    },
    "itfrs_product": {
        "type": "itfrs",
        "ub_tnorm_name": "product",
        "lb_implicator_name": "lukasiewicz",
    },
    "itfrs_hamacher": {
        "type": "itfrs",
        "ub_tnorm_name": "hamacher",
        "lb_implicator_name": "lukasiewicz",
    },
    "itfrs_einstein": {
        "type": "itfrs",
        "ub_tnorm_name": "einstein",
        "lb_implicator_name": "lukasiewicz",
    },
    "itfrs_yager_p2": {
        "type": "itfrs",
        "ub_tnorm_name": "yager",
        "p": 2.0,  # used by Yager T-norm (ub_tnorm); safe because args are filtered per constructor
        "lb_implicator_name": "lukasiewicz",
    },

    # --------------------------
    # OWAFRS variants (different OWA methods + maybe different ub_tnorm)
    # --------------------------
    "owafrs_linear_linear": {
        "type": "owafrs",
        "ub_tnorm_name": "minimum",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "linear",
        "lb_owa_method_name": "linear",
    },
    "owafrs_exp_harm": {
        "type": "owafrs",
        "ub_tnorm_name": "product",
        "lb_implicator_name": "lukasiewicz",
        "ub_owa_method_name": "exponential",
        "lb_owa_method_name": "harmonic",
        "base": 2.0,  # used by exponential OWAWeights
    },
}


# =========================================================
# Pipeline / evaluation
# =========================================================

@dataclass(frozen=True)
class ExperimentResult:
    """@brief Container for evaluation results."""
    dataset: str
    method: str
    before_counts: Counter
    after_counts: Counter
    bal_acc: float
    f1: float


def _parse_scalar(s: str) -> Any:
    """
    @brief Parse a CLI scalar into bool/int/float/str.

    @param s Input string
    @return Parsed value
    """
    sl = s.strip().lower()
    if sl in {"true", "false"}:
        return sl == "true"
    try:
        if "." in sl or "e" in sl:
            return float(sl)
        return int(sl)
    except ValueError:
        return s


def build_frsmote_sampler(config: Dict[str, Any]) -> FRSMOTE:
    """
    @brief Build a configured FRSMOTE sampler from a flat config dictionary.

    @param config Flat config containing fuzzy-rough model parameters + oversampling knobs
    @return Configured FRSMOTE instance
    """
    # FRSMOTE starts UNCONFIGURED; set_params will call configure(...)
    return FRSMOTE().set_params(**config)


def build_pipeline(sampler) -> ImbPipeline:
    """
    @brief Build an imbalanced-learn Pipeline: MinMaxScaler -> sampler(optional) -> LogisticRegression

    @param sampler Oversampler instance or None
    @return ImbPipeline
    """
    clf = LogisticRegression(max_iter=2500)

    if sampler is None:
        return ImbPipeline(steps=[
            ("scaler", MinMaxScaler()),
            ("clf", clf),
        ])

    return ImbPipeline(steps=[
        ("scaler", MinMaxScaler()),
        ("sampler", sampler),
        ("clf", clf),
    ])


def resample_train_counts(pipeline: ImbPipeline, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Counter, Counter]:
    """
    @brief Compute class counts before/after resampling on TRAIN only.

    @param pipeline A pipeline with (scaler, sampler, clf) or (scaler, clf)
    @param X_train Train features
    @param y_train Train labels
    @return (before_counts, after_counts)
    """
    before = Counter(y_train.tolist())

    if "sampler" not in pipeline.named_steps:
        return before, before

    scaler = pipeline.named_steps["scaler"]
    sampler = pipeline.named_steps["sampler"]

    Xs = scaler.fit_transform(X_train)
    Xr, yr = sampler.fit_resample(Xs, y_train)
    after = Counter(yr.tolist())
    return before, after


def fit_and_eval(
    dataset_name: str,
    method_name: str,
    pipeline: ImbPipeline,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> ExperimentResult:
    """
    @brief Fit pipeline and evaluate on test set.

    @param dataset_name Dataset label
    @param method_name Method label
    @param pipeline Pipeline to run
    @param X_train Train features
    @param X_test Test features
    @param y_train Train labels
    @param y_test Test labels
    @return ExperimentResult
    """
    before_counts, after_counts = resample_train_counts(pipeline, X_train, y_train)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    print("\n" + "=" * 90)
    print(f"Dataset: {dataset_name} | Method: {method_name}")
    print(f"Train counts: before={dict(before_counts)} after={dict(after_counts)}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"F1 (binary):       {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    return ExperimentResult(
        dataset=dataset_name,
        method=method_name,
        before_counts=before_counts,
        after_counts=after_counts,
        bal_acc=bal_acc,
        f1=f1,
    )


def main() -> None:
    """@brief CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="toy,iris,cancer",
        help="Comma-separated: toy, iris, cancer (exactly 3 recommended).",
    )
    parser.add_argument(
        "--frsmote-variants",
        type=str,
        default="all",
        help="Comma-separated FRSMOTE variant names, or 'all'.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override ALL FRSMOTE variants: key=value (repeatable).",
    )
    parser.add_argument(
        "--set-variant",
        action="append",
        default=[],
        help="Override ONE variant: NAME:key=value (repeatable).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # ---- dataset registry ----
    ds_registry: Dict[str, DatasetSpec] = {
        "toy": DatasetSpec("toy", load_toy_dataset),
        "iris": DatasetSpec("iris", load_iris_binary_imbalanced),
        "cancer": DatasetSpec("cancer", load_breast_cancer_binary_imbalanced),
    }

    requested_ds = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if len(requested_ds) < 1:
        raise ValueError("No datasets requested.")
    # if user gives >3, still run them; you asked for 3, default is 3.

    # ---- choose frsmote variants ----
    if args.frsmote_variants.strip().lower() == "all":
        chosen_variants = list(FRSMOTE_VARIANTS.keys())
    else:
        chosen_variants = [x.strip() for x in args.frsmote_variants.split(",") if x.strip()]
        unknown = [v for v in chosen_variants if v not in FRSMOTE_VARIANTS]
        if unknown:
            raise ValueError(f"Unknown FRSMOTE variants: {unknown}. Available: {list(FRSMOTE_VARIANTS.keys())}")

    # ---- global overrides (all variants) ----
    global_overrides: Dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Bad --set '{item}', expected key=value")
        k, v = item.split("=", 1)
        global_overrides[k.strip()] = _parse_scalar(v)

    # ---- per-variant overrides ----
    per_variant_overrides: Dict[str, Dict[str, Any]] = {}
    for item in args.set_variant:
        # NAME:key=value
        if ":" not in item or "=" not in item:
            raise ValueError(f"Bad --set-variant '{item}', expected NAME:key=value")
        name, rest = item.split(":", 1)
        k, v = rest.split("=", 1)
        name = name.strip()
        if name not in FRSMOTE_VARIANTS:
            raise ValueError(f"Unknown variant in --set-variant: {name}")
        per_variant_overrides.setdefault(name, {})
        per_variant_overrides[name][k.strip()] = _parse_scalar(v)

    # ---- build samplers list (baseline + classic + FRSMOTE variants) ----
    results: List[ExperimentResult] = []

    for ds_key in requested_ds:
        if ds_key not in ds_registry:
            raise ValueError(f"Unknown dataset '{ds_key}'. Use: {list(ds_registry.keys())}")

        spec = ds_registry[ds_key]
        X, y = spec.loader(args.random_state)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=args.random_state, stratify=y
        )

        # Classic methods
        experiments: List[Tuple[str, ImbPipeline]] = [
            ("Baseline", build_pipeline(None)),
            ("SMOTE", build_pipeline(SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=args.random_state))),
            ("ADASYN", build_pipeline(ADASYN(sampling_strategy="auto", n_neighbors=5, random_state=args.random_state))),
        ]

        # FRSMOTE variants
        for vname in chosen_variants:
            cfg = dict(FRSMOTE_COMMON)
            cfg.update(FRSMOTE_VARIANTS[vname])
            cfg.update(global_overrides)
            cfg.update(per_variant_overrides.get(vname, {}))

            # keep deterministic unless user overrides
            cfg.setdefault("random_state", args.random_state)

            sampler = build_frsmote_sampler(cfg)
            experiments.append((f"FRSMOTE::{vname}", build_pipeline(sampler)))

        # Run all
        print("\n" + "#" * 90)
        print(f"RUNNING DATASET: {spec.name} | total={len(y)} | counts={dict(Counter(y.tolist()))}")
        print("#" * 90)

        for method_name, pipe in experiments:
            res = fit_and_eval(spec.name, method_name, pipe, X_train, X_test, y_train, y_test)
            results.append(res)

    # ---- final compact summary ----
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (per dataset):")
    for ds_name in sorted({r.dataset for r in results}):
        subset = [r for r in results if r.dataset == ds_name]
        subset = sorted(subset, key=lambda z: (z.bal_acc, z.f1), reverse=True)
        print("\n" + "-" * 90)
        print(f"Dataset: {ds_name}")
        for r in subset:
            print(f"{r.method:25s} | bal_acc={r.bal_acc:.4f} | f1={r.f1:.4f} | train_after={dict(r.after_counts)}")


if __name__ == "__main__":
    main()
