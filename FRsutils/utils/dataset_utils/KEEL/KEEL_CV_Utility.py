"""
@file
@brief KEEL Cross-Validation Dataset Utilities

This module provides tools to:
1. Discover and pair KEEL dataset CV folds.
2. Parse KEEL .dat files into pandas DataFrames.
3. Integrate with scikit-learn as a custom cross-validation splitter.
"""

import os
import re
from typing import List, Tuple, Iterator
import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
import warnings

from FRsutils.utils.dataset_utils.KEEL_DS_loader_utility import parse_keel_file


# ------------------------------
# 📂 Discover KEEL CV Files
# ------------------------------
def discover_keel_cv_folds(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    @brief Scans a directory for KEEL .dat CV files and pairs train/test sets by fold.
    NOTE: This method expects the directory passed to it contains train/test folds
    of just one dataset.
        
    @param dataset_dir Path to the folder containing .dat files.
    @return List of (train_file, test_file) tuples.
    
    @exception ValueError If the train and test folds mismatch.
    """
    train_files = []
    test_files = []

    for file in os.listdir(dataset_dir):
        if not file.endswith(".dat"):
            continue
        path = os.path.join(dataset_dir, file)
        if 'tra' in file:
            train_files.append(path)
        elif 'tst' in file:
            test_files.append(path)

    def extract_base_name(path: str) -> str:
        return re.sub(r'(tra|tst)\.dat$', '', os.path.basename(path))

    train_dict = {extract_base_name(f): f for f in train_files}
    test_dict = {extract_base_name(f): f for f in test_files}
    
    common_keys = sorted(set(train_dict) & set(test_dict))
    fold_pairs = [(train_dict[k], test_dict[k]) for k in common_keys]
    
    len1 = len(train_dict)
    len2 = len(test_dict)
    len3 = len(common_keys)
    
    if not (len1 == len2 == len3):
        raise ValueError(
            f"numer od train and test datasets mismatch for datasets in: {dataset_dir} (train: {len1}, test: {len2}, common: {len3}). "
            "Proceeding with common keys only. Extra keys will be ignored.")

    return fold_pairs


# ------------------------------
# 🔄 Scikit-learn CV Wrapper
# ------------------------------
class KeelCVLoader(BaseCrossValidator):
    """
    @brief Custom cross-validator for KEEL dataset folds compatible with scikit-learn.
    This class acts as Scikit-learn CV Wrapper
    
    Loads one fold (train/test) at a time for memory efficiency.
    """

    def __init__(self, fold_paths: List[Tuple[str, str]], one_hot_encode=False, normalize=True):
        """
        @param fold_paths List of (train_file, test_file) pairs.
        """
        self.fold_paths = fold_paths
        self.one_hot_encode = one_hot_encode
        self.normalize = normalize

    def split(self, X=None, y=None, groups=None) -> Iterator[Tuple[List[int], List[int]]]:
        """
        @brief Placeholder method to satisfy scikit-learn's cross-validator API.

        @param X Not used.
        @param y Not used.
        @param groups Not used.
        @return Iterator of empty train/test index lists.
        """
        for i in range(len(self.fold_paths)):
            yield [], []

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        @brief Returns number of folds.

        @param X Not used.
        @param y Not used.
        @param groups Not used.
        @return Number of folds.
        """
        return len(self.fold_paths)

    def load_fold(self, fold_index: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        @brief Loads data for a specific fold from disk.

        @param fold_index Index of the fold to load.
        @return X_train, y_train, X_test, y_test: DataFrames and Series for model training/testing.
        NOTE: pandas dataframes match best to scikit learn pipelines. That's why
        we output dataframe instead of Numpy ndarray
        """
        train_file, test_file = self.fold_paths[fold_index]

        _, df_train, input_order, _ = parse_keel_file(train_file, one_hot_encode=self.one_hot_encode, normalize=self.normalize)
        _, df_test, _, _ = parse_keel_file(test_file, one_hot_encode=self.one_hot_encode, normalize=self.normalize)

        X_train = df_train[input_order]
        y_train = df_train.drop(columns=input_order).iloc[:, 0]

        X_test = df_test[input_order]
        y_test = df_test.drop(columns=input_order).iloc[:, 0]

        return X_train, y_train, X_test, y_test
   
    # TODO: Not tested
    def load_fold_as_array(self, fold_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
    @brief Loads data for a specific fold from disk and returns it as NumPy arrays.

    @details
    This function loads the train and test data for the specified fold index,
    converting both features and targets from pandas DataFrame and Series formats
    into NumPy arrays. This may be useful when working with frameworks or codebases
    that expect raw NumPy input rather than pandas objects.

    @param fold_index Index of the fold to load.

    @return X_train, y_train, X_test, y_test: NumPy arrays containing the training and 
    testing data (features and targets).

    @see load_fold()
    """
        X_train, y_train, X_test, y_test = self.load_fold(fold_index)
        
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy()

        return X_train, y_train, X_test, y_test