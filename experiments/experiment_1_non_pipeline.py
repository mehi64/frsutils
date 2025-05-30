"""
@file
@brief Run multiple oversampling methods on KEEL 5-fold CV datasets and evaluate classifiers.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)

from imblearn.over_sampling import SMOTE, ADASYN
# from smote_variants import distance_SMOTE
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.models.itfrs import ITFRS
import FRsutils.core.tnorms as tnorms
import FRsutils.core.implicators as implicators
from FRsutils.core.similarities import GaussianSimilarity, calculate_similarity_matrix
from FRsutils.utils.dataset_utils.KEEL_CV_Utility import discover_keel_cv_folds, KeelCVLoader

# ------------------------------
# Configurable Parameters
# ------------------------------
CLASSIFIERS = {
    'SVC': SVC(probability=True),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

OVERSAMPLERS = {
    'SMOTE': SMOTE(),
    'ADASYN': ADASYN(),
    # 'distance_SMOTE': distance_SMOTE(),
    'FRSMOTE': lambda X, y: FRSMOTE(
        fr_model=ITFRS(
            similarity_matrix=calculate_similarity_matrix(
                X, GaussianSimilarity(), tnorms.MinTNorm()),
            labels=y,
            tnorm=tnorms.MinTNorm(),
            implicator=implicators.imp_gaines),
        sampling_strategy='auto').fit_resample(X, y)
}

# User-defined datasets and root
DATASET_ROOT = "./datasets/KEEL/imbalanced"
DATASET_NAMES = ['ecoli3-5-fold', 'iris0-5-fold']  # Edit to include/exclude datasets

# ------------------------------
# Metric Calculation Function
# ------------------------------
def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, Any]:
    """
    @brief Computes evaluation metrics from predictions.
    @return Dictionary of evaluation scores.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'gmean': np.sqrt(tp / (tp + fn) * tn / (tn + fp)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# ------------------------------
# Main Experiment Runner
# ------------------------------
def run_experiments():
    """
    @brief Main function to run all oversamplers, datasets, classifiers.
    @return Results DataFrame.
    """
    results = []

    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(DATASET_ROOT, dataset_name)
        fold_paths = discover_keel_cv_folds(dataset_path)
        cv_loader = KeelCVLoader(fold_paths, 
                                 one_hot_encode=True,
                                 normalize=True)

        for fold_idx in range(cv_loader.get_n_splits()):
            X_train, y_train, X_test, y_test = cv_loader.load_fold_as_array(fold_idx)


            for sampler_name, sampler in OVERSAMPLERS.items():
                try:
                    if callable(sampler):
                        X_resampled, y_resampled = sampler(X_train, y_train)
                    else:
                        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f"[Error] {sampler_name} on {dataset_name}, fold {fold_idx}: {str(e)}")
                    continue

                for clf_name, clf in CLASSIFIERS.items():
                    model = clf.fit(X_resampled, y_resampled)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

                    metrics = compute_metrics(y_test, y_pred, y_prob)
                    results.append({
                        'dataset': dataset_name,
                        'fold': fold_idx,
                        'oversampler': sampler_name,
                        'classifier': clf_name,
                        'set': 'test',
                        **metrics
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_results = run_experiments()
    df_results.to_csv("all_fold_results.csv", index=False)
    print("Experiment completed and saved to 'all_fold_results.csv'")
