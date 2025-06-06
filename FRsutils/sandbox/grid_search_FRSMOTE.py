import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import MinMaxScaler


from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE

def save_grid_search_results(grid_search, filename='grid_search_report.csv'):
    """
    Save GridSearchCV results to a CSV file.

    Parameters:
    - grid_search: fitted GridSearchCV object
    - filename: output CSV filename (default: 'grid_search_report.csv')
    """
    # Extract all results
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Optional: Reorder or select only relevant columns
    cols_to_save = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    split_cols = [col for col in results_df.columns if 'split' in col and 'test_score' in col]
    results_df = results_df[cols_to_save + split_cols]

    # Save to CSV with headers
    results_df.to_csv(filename, index=False)
    print(f"Grid search results saved to '{filename}'.")

# Example usage after fitting:
# grid_search.fit(X, y)
# save_grid_search_results(grid_search, 'frsmote_gridsearch_results.csv')


def main():
    # -------------------------
    # 1. Simulate imbalanced dataset
    # -------------------------
    X, y = make_classification(n_samples=50,
                               n_features=50,
                               n_classes=2,
                               n_informative=15,
                               n_redundant=5,
                               n_clusters_per_class=1,
                               weights=[0.85, 0.15],
                               random_state=42)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = np.clip(X, 0.0, 0.99)
    # -------------------------
    # 2. Define CV strategy
    # -------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # -------------------------
    # 3. Grid Search: KNN Classifier
    # -------------------------
    print("Running GridSearchCV with KNeighborsClassifier...")

    knn_pipeline = ImbPipeline([
        ('oversampler', FRSMOTE()),  # Instance of the oversampler
        ('classifier', KNeighborsClassifier())
    ])

    knn_param_grid = {
        'oversampler__k_neighbors': [3, 5],
        'oversampler__similarity_type': ['gaussian', 'linear'],
        'oversampler__lb_implicator_type': ['reichenbach', 'kleene_dienes'],
        'classifier__n_neighbors': [3, 5]
    }

    knn_search = GridSearchCV(
        estimator=knn_pipeline,
        param_grid=knn_param_grid,
        scoring='f1_macro',
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    knn_search.fit(X, y)


    # Extract results
    results_dict = knn_search.cv_results_

    # Optional: convert to DataFrame for easier analysis/sorting
    results_df = pd.DataFrame(results_dict)

    save_grid_search_results(knn_search, filename='grid_search_report.csv')

    # View or save
    print(results_df[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
    
    print("Best params (KNN):", knn_search.best_params_)
    print("Best F1 macro (KNN):", knn_search.best_score_)
    print()

    # # -------------------------
    # # 4. Grid Search: SVC Classifier
    # # -------------------------
    # print("Running GridSearchCV with SVC...")

    # svc_pipeline = ImbPipeline([
    #     ('oversampler', FRSMOTE()),
    #     ('classifier', SVC())
    # ])

    # svc_param_grid = {
    #     'oversampler__k_neighbors': [3, 5],
    #     'oversampler__similarity_type': ['gaussian', 'linear'],
    #     'oversampler__lb_implicator_type': ['reichenbach', 'KD'],
    #     'classifier__C': [0.1, 1],
    #     'classifier__kernel': ['linear', 'rbf']
    # }

    # svc_search = GridSearchCV(
    #     estimator=svc_pipeline,
    #     param_grid=svc_param_grid,
    #     scoring='f1_macro',
    #     cv=cv,
    #     verbose=1,
    #     n_jobs=-1
    # )

    # svc_search.fit(X, y)

    # print("Best params (SVC):", svc_search.best_params_)
    # print("Best F1 macro (SVC):", svc_search.best_score_)


if __name__ == "__main__":
    main()
