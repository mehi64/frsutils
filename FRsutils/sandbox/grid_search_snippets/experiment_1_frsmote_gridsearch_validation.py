"""
@file validate_gridsearch_from_saved_splits.py
@brief Re-evaluates GridSearch configurations manually using saved StratifiedKFold splits.
       Adds per-fold scores and saves results in Excel and JSON.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.models import *  # Needed for registry population
import joblib
import json

# === Step 1: Load dataset and saved CV splits ===
data = np.load("datasets/temp_datasets/frsmote_ds/normalized_data_with_splits.npz")
X, y = data["X"], data["y"]
splits = joblib.load("datasets/temp_datasets/frsmote_ds/cv_splits.pkl")

# === Step 2: Load all parameter combinations ===
results_df = pd.read_excel("temp/gridsearch_results.xlsx")

# === Step 3: JSON-safe converter ===
def json_safe(obj):
    """Convert NumPy types to native Python types recursively for JSON export."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# === Step 4: Evaluation function with per-fold scores ===
def evaluate_config_with_fold_scores(param_dict, X, y, splits):
    frsmote_params = {k.replace("frsmote__", ""): v for k, v in param_dict.items() if k.startswith("frsmote__")}
    svc_params     = {k.replace("svc__", ""): v for k, v in param_dict.items() if k.startswith("svc__")}

    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test   = X[test_idx], y[test_idx]

        oversampler = FRSMOTE(**frsmote_params)
        X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)

        clf = SVC(**svc_params)
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        supp = int(np.sum(y_test == 1))

        fold_scores.append({
            "fold": fold_idx,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "support": supp
        })

    return fold_scores

# === Step 5: Evaluate all configurations ===
all_results = []
per_fold_rows = []  # For Excel: one row per config + fold

for idx, row in results_df.iterrows():
    param_dict = row["params"]
    if isinstance(param_dict, str):
        param_dict = eval(param_dict)

    print(f"Evaluating config {idx+1}/{len(results_df)}")

    try:
        fold_scores = evaluate_config_with_fold_scores(param_dict, X, y, splits)

        mean_f1 = np.mean([f["f1"] for f in fold_scores])
        mean_prec = np.mean([f["precision"] for f in fold_scores])
        mean_rec = np.mean([f["recall"] for f in fold_scores])
        mean_supp = np.mean([f["support"] for f in fold_scores])

        # Add to summary
        all_results.append({
            "index": idx,
            "mean_f1": mean_f1,
            "mean_precision": mean_prec,
            "mean_recall": mean_rec,
            "mean_support": mean_supp,
            "params": param_dict
        })

        # Add per-fold rows
        for fscore in fold_scores:
            per_fold_rows.append({
                "config_index": idx,
                "fold": fscore["fold"],
                "f1": fscore["f1"],
                "precision": fscore["precision"],
                "recall": fscore["recall"],
                "support": fscore["support"],
                "mean_f1": mean_f1,
                "mean_precision": mean_prec,
                "mean_recall": mean_rec,
                "mean_support": mean_supp,
                "params": param_dict
            })


    except Exception as e:
        print(f"Error in config {idx}: {e}")
        all_results.append({
            "index": idx,
            "mean_f1": None,
            "params": param_dict,
            "error": str(e)
        })

# === Step 6: Save summary to Excel ===
df_summary = pd.DataFrame(all_results)
df_summary.to_excel("temp/gridsearch_revalidation_summary.xlsx", index=False)

# === Step 7: Save per-fold metrics to Excel ===
df_folds = pd.DataFrame(per_fold_rows)
df_folds.to_excel("temp/gridsearch_revalidation_fold_scores.xlsx", index=False)

# === Step 8: (Optional) Save JSON version for structured access ===
with open("temp/gridsearch_revalidation_fold_metrics.json", "w") as f:
    json.dump([
        json_safe({"index": r["index"], "fold_metrics": r.get("fold_metrics", [])})
        for r in all_results if "mean_f1" in r and r["mean_f1"] is not None
    ], f, indent=2)

print("✅ Evaluation complete. Results saved to Excel and JSON.")
