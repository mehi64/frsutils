"""
@file validate_gridsearch_results.py
@brief Utility to validate and analyze GridSearchCV results for FRSMOTE pipelines.

Provides:
- Re-ranking results by different metrics
- Filtering by hyperparameters
- Comparing top-k settings
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict

def load_gridsearch_results(path: str) -> pd.DataFrame:
    """
    @brief Load saved GridSearchCV results from Excel or CSV.
    @param path: Path to gridsearch results file
    @return: DataFrame of results
    """
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")

def rank_top_k(results_df: pd.DataFrame, metric: str = "mean_test_score", top_k: int = 5):
    """
    @brief Show top-k results by a specific metric.
    """
    ranked = results_df.sort_values(by=metric, ascending=False).head(top_k)
    print(f"Top {top_k} configurations by {metric}:")
    print(ranked[[metric, "params"]])
    return ranked

def filter_by_params(results_df: pd.DataFrame, **param_filters):
    """
    @brief Filter results by fixed parameter values.
    @param results_df: Full results DataFrame
    @param param_filters: Dict of param=value to filter
    @return: Filtered DataFrame
    """
    df = results_df.copy()
    for param, value in param_filters.items():
        df = df[df[f"param_{param}"] == value]
    return df

def re_evaluate_on_predictions(estimator, X, y, metric="f1"):
    """
    @brief Re-evaluate an estimator using cross_val_predict + metric
    @param estimator: fitted pipeline
    @param X: Input features
    @param y: Labels
    @param metric: f1, accuracy, etc.
    @return: score
    """
    y_pred = cross_val_predict(estimator, X, y, cv=3)
    if metric == "f1":
        return f1_score(y, y_pred)
    elif metric == "accuracy":
        return accuracy_score(y, y_pred)
    elif metric == "report":
        return classification_report(y, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
