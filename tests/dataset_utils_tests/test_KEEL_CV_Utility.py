import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import patch
from typing import Literal

from FRsutils.utils.dataset_utils.KEEL_CV_Utility import discover_keel_cv_folds, KeelCVLoader


# ─────────────────────────────────────────────────────────────
# UNIT TESTS (MOCKED parse_keel_file)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cv_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(1, 4):
            for suffix in ['tra', 'tst']:
                path = os.path.join(tmpdir, f"dataset-{i}{suffix}.dat")
                with open(path, 'w') as f:
                    f.write("% dummy")
        yield tmpdir

def test_discover_keel_cv_folds_pairs_files_correctly(mock_cv_directory: str):
    pairs = discover_keel_cv_folds(mock_cv_directory)
    assert len(pairs) == 3
    for train_path, test_path in pairs:
        assert train_path.endswith("tra.dat")
        assert test_path.endswith("tst.dat")
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)

@pytest.fixture
def mock_cv_directory_with_non_matched_train_test_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(1, 4):
            for suffix in ['tra', 'tst']:
                path = os.path.join(tmpdir, f"dataset-{i}{suffix}.dat")
                with open(path, 'w') as f:
                    f.write("% dummy")
        
        path = os.path.join(tmpdir, f"dataset-4tra.dat")
        with open(path, 'w') as f:
            f.write("% dummy")
            
        path = os.path.join(tmpdir, f"dataset-5tst.dat")
        with open(path, 'w') as f:
            f.write("% dummy")
        yield tmpdir

def test_discover_keel_cv_folds_pairs_files_correctly_for_Non_Matched_train_test(mock_cv_directory_with_non_matched_train_test_files: str):
    with pytest.raises(ValueError):
        discover_keel_cv_folds(mock_cv_directory_with_non_matched_train_test_files)

    assert True

@patch("FRsutils.utils.dataset_utils.KEEL_CV_Utility.parse_keel_file")
def test_keelcvloader_load_fold(mock_parse_keel_file):
    # Fake DataFrames returned by mock
    df_train = pd.DataFrame({
        "f1": [1, 2],
        "f2": [3, 4],
        "class": ["A", "B"]
    })
    df_test = pd.DataFrame({
        "f1": [5, 6],
        "f2": [7, 8],
        "class": ["A", "B"]
    })

    # Simulate parse_keel_file
    mock_parse_keel_file.side_effect = [
        (None, df_train, ["f1", "f2"], ["class"]),
        (None, df_test, ["f1", "f2"], ["class"]),
    ]

    loader = KeelCVLoader([("train1.dat", "test1.dat")])
    X_train, y_train, X_test, y_test = loader.load_fold(0)

    assert X_train.equals(df_train[["f1", "f2"]])
    assert y_train.equals(df_train["class"])
    assert X_test.equals(df_test[["f1", "f2"]])
    assert y_test.equals(df_test["class"])


def test_keelcvloader_n_splits_and_split():
    folds = [("f1tra.dat", "f1tst.dat"), ("f2tra.dat", "f2tst.dat")]
    loader = KeelCVLoader(folds)
    assert loader.get_n_splits() == 2
    splits = list(loader.split())
    assert len(splits) == 2
    for train, test in splits:
        assert train == []
        assert test == []



# ─────────────────────────────────────────────────────────────
# INTEGRATION TESTS (REAL FILES)
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def real_keel_cv_dir():
    return "./datasets/KEEL/imbalanced/ecoli3-5-fold"  # Folder with ecoli3-5-1tra.dat ... ecoli3-5-5tst.dat


def test_real_keel_folds_discovery(real_keel_cv_dir: Literal['./datasets/KEEL/imbalanced/ecoli3-5-fold']):
    folds = discover_keel_cv_folds(real_keel_cv_dir)
    assert len(folds) == 5, "Should discover 5 folds"

    for train_path, test_path in folds:
        assert train_path.endswith("tra.dat")
        assert test_path.endswith("tst.dat")
        base_train = os.path.basename(train_path).replace("tra.dat", "")
        base_test = os.path.basename(test_path).replace("tst.dat", "")
        assert base_train == base_test


def test_real_keel_fold_loading(real_keel_cv_dir: Literal['./datasets/KEEL/imbalanced/ecoli3-5-fold']):
    folds = discover_keel_cv_folds(real_keel_cv_dir)
    loader = KeelCVLoader(folds)

    for i in range(len(folds)):
        X_train, y_train, X_test, y_test = loader.load_fold(i)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert list(X_train.columns) == list(X_test.columns)
        assert y_train.nunique() >= 1
        assert y_test.nunique() >= 1


def test_real_keel_fold_loading_data_correctness(real_keel_cv_dir: Literal['./datasets/KEEL/imbalanced/ecoli3-5-fold']):
    folds = discover_keel_cv_folds(real_keel_cv_dir)
    loader = KeelCVLoader(folds,one_hot_encode=False, normalize=True)

    for i in range(len(folds)):
        X_train, y_train, X_test, y_test = loader.load_fold(i)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)

        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert list(X_train.columns) == list(X_test.columns)
        assert y_train.nunique() >= 1
        assert y_test.nunique() >= 1