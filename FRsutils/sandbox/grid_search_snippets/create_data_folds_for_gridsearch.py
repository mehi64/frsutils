from sklearn.model_selection import StratifiedKFold
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create imbalanced data
X, y = make_classification(n_samples=300, n_features=10, n_informative=6,
                           n_redundant=2, weights=[0.85, 0.15], random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = np.clip(X, 0.0, 0.99)

# Create stratified 3-fold split
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(skf.split(X, y))

# Save X, y, and splits
np.savez("datasets/temp_datasets/frsmote_ds/normalized_data_with_splits.npz", X=X, y=y)
joblib.dump(splits, "datasets/temp_datasets/frsmote_ds/cv_splits.pkl")
