import numpy as np
from FRsutils.core.preprocess.oversampling.FRSMOTE import FRSMOTE
# from FRsutils.core.models.itfrs import ITFRS
# from FRsutils.core.models.owafrs import OWAFRS
from FRsutils.core.models import *

# Example imbalanced dataset
X = np.array([
    [0.1, 0.2],
    [0.2, 0.1],
    [0.15, 0.18],  # Minority class
    [0.8, 0.9],
    [0.82, 0.88],
    [0.85, 0.91],
    [0.87, 0.93]
])
y = np.array([1, 1, 1, 0, 0, 0, 0])  # Class 1 is minority

smote1 = FRSMOTE.empty()

config = {
    'type': 'owafrs',
    'similarity_name': 'gaussian',
    'gaussian_similarity_sigma': 0.2,
    'similarity_tnorm_name': 'minimum',
    'lb_implicator_name': 'lukasiewicz',
    'ub_tnorm_name': 'product',
    'ub_owa_method_name': "linear",
    'lb_owa_method_name': "linear",
    'k_neighbors': 3,
    'random_state': 42
}

smote1.store_init_config(
    type= 'owafrs',
    similarity_name= 'gaussian',
    gaussian_similarity_sigma= 0.2,
    similarity_tnorm_name= 'minimum',
    lb_implicator_name= 'lukasiewicz',
    ub_tnorm_name= 'product',
    ub_owa_method_name= "linear",
    lb_owa_method_name= "linear",
    k_neighbors= 3,
    random_state= 42
)

smote1.store_init_config(**config)
X_resampled, y_resampled = smote1.fit(X, y)
print(X_resampled)
print(y_resampled)




# Initialize FRSMOTE with default or overridden parameters
smote = FRSMOTE(
    type='owafrs',
    similarity_name='gaussian',
    gaussian_similarity_sigma=0.2,
    similarity_tnorm_name='minimum',
    lb_implicator_name='lukasiewicz',
    ub_tnorm_name='product',
    ub_owa_method_name= "linear",
    lb_owa_method_name= "linear",
    k_neighbors=3,
    random_state=42
)

# Apply oversampling
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original class distribution:", np.bincount(y))
print("Resampled class distribution:", np.bincount(y_resampled))

params = smote.get_params()

smote.set_params(**params)
print("Parameters:", params)