"""
@file
@brief Run multiple oversampling methods on KEEL 5-fold CV datasets and evaluate classifiers using pipelines.
"""

import os
import numpy as np

# print(np.__config__.show())

import pandas as pd
from typing import List, Dict, Any
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
# from smote_variants import distance_SMOTE
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.models.itfrs import ITFRS
import FRsutils.core.tnorms as tnorms
import FRsutils.core.implicators as implicators
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
    'FRSMOTE': 'FRSMOTE'  # Custom handling in code below
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
    labels = list(np.unique(y_true))
    if len(labels) != 2:
        raise ValueError("Binary classification expected. Found labels: {}".format(labels))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=labels[1], zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=labels[1]),
        'f1_score': f1_score(y_true, y_pred, pos_label=labels[1]),
        'auc': roc_auc_score(y_true, y_prob),
        'gmean': np.sqrt(tp / (tp + fn) * tn / (tn + fp)),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

# ------------------------------
# Main Experiment Runner
# ------------------------------
def run_experiments():
    """
    @brief Main function to run all oversamplers, datasets, classifiers using pipelines.
    @return Results DataFrame.
    """
    results = []

    for dataset_name in DATASET_NAMES:
        dataset_path = os.path.join(DATASET_ROOT, dataset_name)
        fold_paths = discover_keel_cv_folds(dataset_path)
        cv_loader = KeelCVLoader(fold_paths, 
                                 one_hot_encode=False,
                                 normalize=True)

        for fold_idx in range(cv_loader.get_n_splits()):
            X_train, y_train, X_test, y_test = cv_loader.load_fold_as_array(fold_idx)
            # X_train = X_train_df.to_numpy()
            # y_train = y_train_s.to_numpy()
            
            # X_test = X_test_df.to_numpy()
            # y_test = y_test_s.to_numpy()
            
            # le = LabelEncoder()
            # y_train = le.fit_transform(y_train)
            # y_test = le.transform(y_test)

            for sampler_name, sampler in OVERSAMPLERS.items():
                for clf_name, clf in CLASSIFIERS.items():
                    try:
                        if sampler_name == 'FRSMOTE':
                            sampler = FRSMOTE(
                                fr_model=ITFRS(
                                    similarity_matrix=calculate_similarity_matrix(
                                        X_train, GaussianSimilarity(), tnorms.MinTNorm()),
                                    labels=y_train,
                                    tnorm=tnorms.MinTNorm(),
                                    implicator=implicators.imp_gaines),
                                sampling_strategy='auto')

                        pipeline = ImbPipeline([
                            ('sampler', sampler),
                            ('classifier', clf)
                        ])

                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else y_pred

                        metrics = compute_metrics(y_test, y_pred, y_prob)
                        results.append({
                            'dataset': dataset_name,
                            'fold': fold_idx,
                            'oversampler': sampler_name,
                            'classifier': clf_name,
                            'set': 'test',
                            **metrics
                        })
                    except Exception as e:
                        print(f"[Error] {sampler_name} + {clf_name} on {dataset_name}, fold {fold_idx}: {str(e)}")
                        continue

    return pd.DataFrame(results)


if __name__ == "__main__":
    df_results = run_experiments()
    df_results.to_csv("all_fold_results_pipeline.csv", index=False)
    print("Experiment completed and saved to 'all_fold_results.csv'")
