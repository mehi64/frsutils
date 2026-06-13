# SPDX-License-Identifier: BSD-3-Clause
"""Run Baseline / SMOTE / ADASYN / FRSMOTE(ITFRS, OWAFRS) on 3 small datasets using imbalanced-learn Pipeline.

This module supports reproducible experiments and is not part of the stable public API.
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
from frsampling import FRSMOTE


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
