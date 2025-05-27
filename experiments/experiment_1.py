import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from FRsutils.utils.dataset_utils.KEEL_CV_Utility import discover_keel_cv_folds, KeelCVLoader
from FRsutils.utils.dataset_utils.KEEL_DS_loader_utility import create_X_y
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
from FRsutils.core.models import ITFRS  # Replace with actual model
import warnings
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp
import FRsutils.core.similarities as sim
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import smote_variants as sv

# # Dummy FRSMOTE (replace with real FRSMOTE if available)
# class DummyFRSMOTE(SMOTE):
#     def __init__(self): super().__init__()
    
# Generate synthetic datasets
def generate_datasets():
    datasets = {}
    scaler = MinMaxScaler()
    for i in range(3):
        X, y = make_classification(n_samples=200, n_features=4, n_informative=3, n_redundant=0,
                                   n_clusters_per_class=1, weights=[0.9, 0.1], flip_y=0,
                                   random_state=42 + i)
        
        
        X_normalized = scaler.fit_transform(X)
        datasets[f"synthetic_ds_{i+1}"] = (X_normalized, y)
    return datasets

def get_oversamplers(fr_model):
    return {
        "SMOTE": SMOTE(),
        "BorderlineSMOTE": BorderlineSMOTE(),
        "SVMSMOTE": SVMSMOTE(),
        "ADASYN": ADASYN(),
        "KMeansSMOTE": KMeansSMOTE(),
        "FRSMOTE": FRSMOTE(fr_model=fr_model, k_neighbors=5, bias_interpolation=False),
        # smote-variants samplers (wrapped to match API)
        "G-SMOTE": sv.G_SMOTE(),
        "MDO": sv.MDO(),
        "SMOTE_IPF": sv.SMOTE_IPF(),
        "ProWSyn": sv.ProWSyn(),
        "ROSE": sv.ROSE(),
        "geometirc_SMOTE": sv.ProWSyn()
    }

def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_pred),
        'gmean': np.sqrt((tp / (tp + fn + 1e-10)) * (tn / (tn + fp + 1e-10))),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }

def run_experiments(base_dir, output_csv):
    all_results = []
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for dataset in datasets:
        dataset_path = os.path.join(base_dir, dataset)
        folds = discover_keel_cv_folds(dataset_path)
        loader = KeelCVLoader(folds)

        for fold_idx in range(loader.get_n_splits()):
            X_train_df, y_train, X_test_df, y_test = loader.load_fold(fold_idx)
            X_train, y_train = create_X_y(X_train_df, list(X_train_df.columns), [y_train.name])
            X_test, y_test = create_X_y(X_test_df, list(X_test_df.columns), [y_test.name])

            tnrm = tn.ProductTNorm()
            impli=imp.imp_reichenbach
            sim_f = sim.LinearSimilarity()
            sim_matrix = sim.calculate_similarity_matrix(X_train, similarity_func=sim_f, tnorm=tnrm)
    
             
            fr_model = ITFRS(sim_matrix, 
                             y_train,
                             tnrm,
                             impli) 
            oversamplers = get_oversamplers(fr_model)

            for name, sampler in oversamplers.items():
                try:
                    X_res, y_res = sampler.fit_resample(X_train, y_train)
                    from sklearn.ensemble import RandomForestClassifier
                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X_test)

                    results = evaluate(y_test, y_pred)
                    results.update({
                        'dataset': dataset,
                        'fold': fold_idx,
                        'oversampler': name,
                        'set': 'test'
                    })
                    all_results.append(results)
                except Exception as e:
                    warnings.warn(f"Error in {name} on {dataset} fold {fold_idx}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
# Run experiments
def run_experiments_synthetic():
    results = []
    datasets = generate_datasets()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (X, y) in datasets.items():
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            tnrm = tn.ProductTNorm()
            impli=imp.imp_reichenbach
            sim_f = sim.LinearSimilarity()
            sim_matrix = sim.calculate_similarity_matrix(X_train, similarity_func=sim_f, tnorm=tnrm)
    
             
            fr_model = ITFRS(sim_matrix, 
                             y_train,
                             tnrm,
                             impli) 
            oversamplers = get_oversamplers(fr_model)
            
            for sampler_name, sampler in oversamplers.items():
                try:
                    X_res, y_res = sampler.fit_resample(X_train, y_train)
                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X_test)

                    metrics = evaluate(y_test, y_pred)
                    metrics.update({
                        'dataset': name,
                        'fold number': fold,
                        'oversampler': sampler_name,
                        'set': 'test'
                    })
                    results.append(metrics)
                except Exception as e:
                    warnings.warn(f"{sampler_name} failed on {name} fold {fold}: {e}")

    df = pd.DataFrame(results)
    df.to_csv("synthetic_oversampling_results.csv", index=False)
    print("Results saved to synthetic_oversampling_results.csv")


# run_experiments('C:/Users/Mehran Amiri/Documents/codes/FRsutils/datasets/KEEL/imbalanced', 'C:/Users/Mehran Amiri/Documents/codes/FRsutils/results.csv')
# run_experiments_synthetic()
print("done")

l=sv.get_all_oversamplers()

print("done")