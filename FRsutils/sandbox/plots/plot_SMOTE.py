from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from imblearn_extra.over_sampling import GeometricSMOTE



# Generate imbalanced data
X, y = make_classification(n_samples=150, n_features=10, n_informative=2, # Increased sample size
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
print("\nApplying SMOTE ...")

# 2. Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_norm, y)

g_smote = GeometricSMOTE(random_state=0,
                         truncation_factor=1.0,      # 1.0 = full sphere
                         deformation_factor=0.5,     # 0.0 spheroid → 1.0 line
                         selection_strategy="combined")
X_res, y_res = g_smote.fit_resample(X, y)

print(f"Resampled class distribution: {Counter(y_resampled)}")

# Optional: convert to DataFrame for inspection
df_resampled = pd.DataFrame(X_resampled)
df_resampled['label'] = y_resampled
print(df_resampled.head())


# --- Plotting (same as before) ---
plt.figure(figsize=(15, 5))
# ... (plotting code remains the same) ...
plt.subplot(1, 3, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0 (Maj)", alpha=0.6, s=10)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1 (Min)", alpha=0.6, s=10)
plt.title("Original Data")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X_resampled[y_resampled == 0][:, 0], X_resampled[y_resampled == 0][:, 1], label="Class 0", alpha=0.6, s=10)
# plt.scatter(X_res_smote[y_res_smote == 1][:, 0], X_res_smote[y_res_smote == 1][:, 1], label="Class 1", alpha=0.6, s=10)
# Highlight synthetic points (crude way)
n_original_minority = Counter(y)[1] if 1 in Counter(y) else 0
n_synthetic = Counter(y_resampled)[1] - n_original_minority
if n_synthetic > 0 and 1 in Counter(y_resampled):
        synthetic_X = X_resampled[y_resampled==1][-n_synthetic:]
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