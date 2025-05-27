"""
KEEL Cross-Validation Dataset Utilities
----------------------------------------
This module provides tools to:
1. Discover and pair KEEL dataset CV folds.
2. Parse KEEL .dat files into pandas DataFrames.
3. Integrate with scikit-learn as a custom cross-validation splitter.
"""

import os
import re
from typing import List, Tuple, Dict, Iterator
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from FRsutils.utils.dataset_utils.KEEL_DS_loader_utility import parse_keel_file  # Assumes previous parser code is available


# ------------------------------
# 📂 Discover KEEL CV Files
# ------------------------------
def discover_keel_cv_folds(dataset_dir: str) -> List[Tuple[str, str]]:
    """
    Scans a directory for KEEL .dat CV files and pairs train/test sets by fold.

    Parameters:
        dataset_dir (str): Path to folder containing .dat files.

    Returns:
        List of (train_file, test_file) tuples.
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

    # Match train/test pairs based on file stem without 'tra' or 'tst'
    def extract_base_name(path: str) -> str:
        return re.sub(r'(tra|tst)\.dat$', '', os.path.basename(path))

    train_dict = {extract_base_name(f): f for f in train_files}
    test_dict = {extract_base_name(f): f for f in test_files}

    common_keys = sorted(set(train_dict) & set(test_dict))
    fold_pairs = [(train_dict[k], test_dict[k]) for k in common_keys]

    return fold_pairs


# ------------------------------
# 🔄 Scikit-learn CV Wrapper (Lazy Loading Version)
# ------------------------------
class KeelCVLoader(BaseCrossValidator):
    """
    Custom cross-validator for KEEL dataset folds compatible with scikit-learn.

    Loads only one fold (train/test) at a time for memory efficiency.
    """

    def __init__(self, fold_paths: List[Tuple[str, str]]):
        """
        Parameters:
            fold_paths (List[Tuple[str, str]]): List of (train_file, test_file) pairs.
        """
        self.fold_paths = fold_paths

    def split(self, X=None, y=None, groups=None) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Yields:
            train_idx, test_idx: Indices are relative to the current fold's local data.
            This method is a placeholder since scikit-learn requires split() even if not used.
        """
        for i in range(len(self.fold_paths)):
            yield [], []  # Placeholder to satisfy API, use `load_fold(i)` instead

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns number of folds"""
        return len(self.fold_paths)

    def load_fold(self, fold_index: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Loads data for a specific fold from disk.

        Parameters:
            fold_index (int): Index of the fold to load.

        Returns:
            X_train, y_train, X_test, y_test: DataFrames and Series ready for model training/testing.
        """
        train_file, test_file = self.fold_paths[fold_index]

        # Load train and test data separately
        _, df_train, input_order, _ = parse_keel_file(train_file)
        _, df_test, _, _ = parse_keel_file(test_file)

        # Assume output is the last attribute
        X_train = df_train[input_order]
        y_train = df_train.drop(columns=input_order).iloc[:, 0]

        X_test = df_test[input_order]
        y_test = df_test.drop(columns=input_order).iloc[:, 0]

        return X_train, y_train, X_test, y_test