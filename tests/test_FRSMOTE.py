import FRsutils.core.preprocess.oversampling.FRSMOTE as FRSMOTE_util
from FRsutils.core.models import ITFRS
import FRsutils.core.tnorms as tn
import FRsutils.core.implicators as imp
import FRsutils.core.similarities as sim_util 
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import heapq
import time # For timing comparison if needed
# import KEEL_DS_loader_utility as KEEL_DS_loader_util
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


def test1():
    # Generate imbalanced data
    X, y = make_classification(n_samples=200, n_features=10, n_informative=2, # Increased sample size
                               n_redundant=0, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.95, 0.05], # More imbalance
                               class_sep=0.7, random_state=42)
    # Normalize data
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)
    
    # file_path = '/hard/Fuzzy_Rough_SMOTE_old/Dataset/KEEL_DS/imbalanced/imb_IRlowerThan9/imb_IRlowerThan9/ecoli1/ecoli1-5-fold/ecoli1-5-1tra.dat'
    # file_path = '/hard/Fuzzy_Rough_SMOTE_old/Dataset/KEEL_DS/imbalanced/imb_IRhigherThan9p1/imb_IRhigherThan9p1/abalone19/abalone19.dat'

    # global_metadata, df, input_features, output_features = KEEL_DS_loader_util.parse_keel_file(file_path=file_path)
    # X, y = KEEL_DS_loader_util.create_X_y(df, input_features, output_features)
    
    print("Original dataset shape %s" % Counter(y))

    # --- FRSMOTE Example ---
    print("\nApplying FRSMOTE (Optimized)...")

#     imp_ = imp.imp_kleene_dienes
#     tnorm_ = tn.MinTNorm()
#     sim_func_ = sim_util.GaussianSimilarity(sigma=0.2)
#     sim_matrix_ = sim_util.calculate_similarity_matrix(X_norm,
#                                                     similarity_func=sim_func_,
#                                                     tnorm=tnorm_)
#     fr_model = ITFRS(similarity_matrix=sim_matrix_,
#                     labels=y,
#                     tnorm=tnorm_,
#                     implicator=imp_)

    ITFR_params = {'lb_tnorm' : 'minimum',
                   'ub_implicator' : 'reichenbach'}
    frsmote = FRSMOTE_util.FRSMOTE( k_neighbors=5,
                                    bias_interpolation=False,
                                    random_state=42,
                                    lb_tnorm='minimum',
                                    ub_implicator='reichenbach')
    
    pp = frsmote.set_params()
    X_res_smote, y_res_smote = frsmote.fit_resample(X_norm, y)
    # t_smote_end = time.time()
    print("Resampled dataset shape (FRSMOTE) %s" % Counter(y_res_smote))
    # print(f"FRSMOTE execution time: {t_smote_end - t_smote_start:.4f}s")


    # # --- FRS Undersampling Example ---
    # print("\nApplying FRS-Based Undersampling (Optimized)...")
    # # t_under_start = time.time()
    # frs_under = FRSMOTE_util.FRSBasedUndersamplerOptimized( # Use optimized class
    #     sampling_strategy=1.0, # Target 1:1 ratio
    #     # undersampling_method='remove_safest',
    #     undersampling_method='weighted_removal',
    #     # undersampling_method='preserve_boundary', boundary_threshold=0.4,
    #     similarity='gaussian', sigma=0.2,
    #     pos_region_type='standard', # 'standard' or 'vqrs_avg' are vectorized
    #     random_state=42
    # )
    # X_res_under, y_res_under = frs_under.fit_resample(X, y)
    # # t_under_end = time.time()
    # print("Resampled dataset shape (FRS Undersampler) %s" % Counter(y_res_under))
    # # print(f"FRS Undersampling execution time: {t_under_end - t_under_start:.4f}s")


    # --- Plotting (same as before) ---
    plt.figure(figsize=(15, 5))
    # ... (plotting code remains the same) ...
    plt.subplot(1, 3, 1)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0 (Maj)", alpha=0.6, s=10)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1 (Min)", alpha=0.6, s=10)
    plt.title("Original Data")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(X_res_smote[y_res_smote == 0][:, 0], X_res_smote[y_res_smote == 0][:, 1], label="Class 0", alpha=0.6, s=10)
    # plt.scatter(X_res_smote[y_res_smote == 1][:, 0], X_res_smote[y_res_smote == 1][:, 1], label="Class 1", alpha=0.6, s=10)
    # Highlight synthetic points (crude way)
    n_original_minority = Counter(y)[1] if 1 in Counter(y) else 0
    n_synthetic = Counter(y_res_smote)[1] - n_original_minority
    if n_synthetic > 0 and 1 in Counter(y_res_smote):
         synthetic_X = X_res_smote[y_res_smote==1][-n_synthetic:]
         plt.scatter(synthetic_X[:, 0], synthetic_X[:, 1], label="Synthetic", marker='x', c='yellow', s=30)
    plt.title("After FRSMOTE (Optimized)")
    plt.legend()

    # plt.subplot(1, 3, 3)
    # plt.scatter(X_res_under[y_res_under == 0][:, 0], X_res_under[y_res_under == 0][:, 1], label="Class 0", alpha=0.6, s=10)
    # plt.scatter(X_res_under[y_res_under == 1][:, 0], X_res_under[y_res_under == 1][:, 1], label="Class 1", alpha=0.6, s=10)
    # plt.title("After FRS Undersampling (Optimized)")
    # plt.legend()


    plt.tight_layout()
    plt.show()