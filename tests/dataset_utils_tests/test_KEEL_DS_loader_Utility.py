import pytest
import pandas as pd
import numpy as np
import tempfile
import math
from FRsutils.utils.dataset_utils.KEEL_DS_loader_utility import parse_keel_file, create_X_y

# TODO: NOTE: file _apply_one_hot_encoding is not tested

@pytest.fixture
def keel_temp_file():
    content = """% Sample KEEL dataset
@relation test-dataset

@attribute attr1 real [0,10]
@attribute attr2 {low,medium,high}
@attribute class {yes,no}

@inputs attr1, attr2
@outputs class

@data
1.0, low, yes
2.0, medium, no
3.0, high, yes
?, medium, no
"""
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.dat', delete=False) as f:
        f.write(content)
        f.seek(0)
        yield f.name


def test_parse_keel_file_basic(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file, one_hot_encode=False, normalize=False)
    assert metadata['relation'] == 'test-dataset'
    assert 'attr1' in input_features
    assert 'attr2' in input_features
    assert output_features == ['class']
    assert metadata['num_instances'] == 4
    assert metadata['missing_values_total'] == 1
    assert metadata['instances_with_missing'] == 1
    assert set(metadata['class_distribution'].keys()) == {'yes', 'no'}
    assert metadata['class_distribution']['yes'] == 2
    assert metadata['class_distribution']['no'] == 2
    assert metadata['imbalance_ratio'] == 1.0
    assert metadata['num_input_features'] == 2
    assert metadata['num_output_features'] == 1


def test_parse_with_one_hot_encoding(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file, one_hot_encode=True, normalize=False)
    assert all(col.startswith('attr2_') for col in input_features if 'attr2' in col)
    assert 'attr2' not in df.columns
    assert metadata['attributes']['attr2_low']['type'] == 'binary'


def test_parse_with_normalization(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file, one_hot_encode=False, normalize=True)
    attr1 = df['attr1'].dropna().astype(float)
    assert attr1.min() >= 0.0 and attr1.max() <= 1.0
    assert metadata['attributes']['attr1']['min_range'] == 0.0
    assert metadata['attributes']['attr1']['max_range'] == 1.0
    assert metadata['attributes']['attr1']['range_source'] == 'normalized'


def test_create_X_y_single_target(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file)
    X, y = create_X_y(df, input_features, output_features)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]


def test_create_X_y_no_output_raises_error(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file)
    with pytest.raises(ValueError, match="Output feature list is empty"):
        create_X_y(df, input_features, [])


def test_create_X_y_multiple_outputs_raises_error(keel_temp_file):
    metadata, df, input_features, output_features = parse_keel_file(keel_temp_file)
    with pytest.raises(ValueError, match="Multiple output features detected"):
        create_X_y(df, input_features, ['class', 'extra_output'])


# ──────────────────────────────────────────────────────
# NEW TESTS using real KEEL dataset: ecoli3-5-1tra.dat
# ──────────────────────────────────────────────────────

@pytest.fixture
def ecoli_dataset_file():
    return "./datasets/KEEL/imbalanced/ecoli3-5-fold/ecoli3-5-1tra.dat"


def test_ecoli_parse_metadata(ecoli_dataset_file):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file, one_hot_encode=False, normalize=False)
    # the name inside file
    assert metadata['relation'] == 'ecoli3'
    assert metadata['num_instances'] > 0
    assert isinstance(metadata['class_distribution'], dict)
    assert all(key is not None for key in metadata['class_distribution'].keys())
    
    assert 'Mcg' in input_features
    assert 'Gvh' in input_features
    assert 'Lip' in input_features
    assert 'Chg' in input_features
    assert 'Aac' in input_features
    assert 'Alm1' in input_features
    assert 'Alm2' in input_features
    
    
    expected_min_ranges =[0.0 , 0.16, 0.48, 0.5, 0.0, 0.03, 0.0]
    expected_max_ranges =[0.89, 1.0, 1.0, 1.0, 0.88, 1.0, 0.99]
    
    min_ranges = []
    max_ranges = []
    
    for i in range(len(input_features)):

        temp_min_range = metadata['attributes'][input_features[i]]['min_range']
        temp_max_range = metadata['attributes'][input_features[i]]['max_range']
        min_ranges.append(temp_min_range)
        max_ranges.append(temp_max_range)
    
    assert min_ranges == expected_min_ranges
    assert max_ranges == expected_max_ranges
        
    features = set(metadata['attributes'].keys())
    assert features == {'Mcg', 'Gvh', 'Lip', 'Chg', 'Aac', 'Alm1', 'Alm2', 'Class'}
    
    assert output_features == ['Class']
    assert metadata['num_instances'] == 268
    assert metadata['missing_values_total'] == 0
    assert metadata['instances_with_missing'] == 0
    assert set(metadata['class_distribution'].keys()) == {'positive', 'negative'}
    assert metadata['class_distribution']['positive'] == 28
    assert metadata['class_distribution']['negative'] == 240
    assert math.isclose(metadata['imbalance_ratio'], 8.5714285714,rel_tol=1e-6)
    assert metadata['num_input_features'] == 7
    assert metadata['num_output_features'] == 1


@pytest.mark.parametrize("row_idx, expected_non_decision_vector, expected_class",  [
    (0, np.array([0.68, 0.49, 1.0, 0.5,  0.62, 0.55, 0.28]), 'negative'),
    (1, np.array([0.75, 0.84, 0.48, 0.5, 0.35, 0.52, 0.33]), 'negative'),
    (2, np.array([0.52, 0.44, 0.48, 0.5, 0.37, 0.36, 0.42]), 'negative'),
    (3, np.array([0.87, 0.49, 0.48, 0.5, 0.61, 0.76, 0.79]), 'positive'),
    (4, np.array([0.41, 0.51, 0.48, 0.5, 0.58, 0.2, 0.31]), 'negative'),
    (5, np.array([0.2, 0.46, 0.48, 0.5, 0.57, 0.78, 0.81]), 'negative'),
    (267, np.array([0.74, 0.49, 0.48, 0.5, 0.42, 0.54, 0.36]), 'negative')
])
def test_ecoli_parse_data_correctness(ecoli_dataset_file, row_idx, expected_non_decision_vector, expected_class):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file, one_hot_encode=False, normalize=False)
    
    actual_class = df.iloc[row_idx, -1]
    actual_non_decision_vector = df.iloc[row_idx, :-1].astype(float).to_numpy()
    assert np.allclose(actual_non_decision_vector, expected_non_decision_vector), (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_non_decision_vector}\n"
        f"  expected = {expected_non_decision_vector}"
    )
    assert actual_class == expected_class, (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_class}\n"
        f"  expected = {expected_class}"
    )
    
    assert True
 

@pytest.mark.parametrize("row_idx, expected_normalized_vector, expected_class",  [
    (0, np.array([0.7640449438, 0.3928571429, 1.0, 0.0,  0.7045454545, 0.5360824742, 0.2828282828]), 'negative'),
    (3, np.array([0.9775280899, 0.3928571429, 0.0, 0.0, 0.6931818182, 0.7525773196, 0.797979798]), 'positive')
    # ,(267, np.array([0.8314606742, 0.49, 0.48, 0.5, 0.42, 0.54, 0.36]), 'negative')
])
def test_ecoli_normalized_data_correctness(ecoli_dataset_file, row_idx, expected_normalized_vector, expected_class):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file, one_hot_encode=False, normalize=True)
    
    actual_class = df.iloc[row_idx, -1]
    actual_normalized_vector = df.iloc[row_idx, :-1].astype(float).to_numpy()
    assert np.allclose(actual_normalized_vector, expected_normalized_vector), (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_normalized_vector}\n"
        f"  expected = {expected_normalized_vector}"
    )
    assert actual_class == expected_class, (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_class}\n"
        f"  expected = {expected_class}"
    )
    
    assert True 
    
def test_ecoli_data_integrity(ecoli_dataset_file):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file)
    assert df.shape[0] == metadata['num_instances']
    assert df.shape[1] == len(input_features) + len(output_features)
    # Just one class feature
    assert output_features[0] == df.columns[-1]
    assert df.isnull().sum().sum() == 0  # Should have handled missing values


def test_ecoli_normalization_bounds(ecoli_dataset_file):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file, one_hot_encode=False, normalize=True)
    for col in input_features:
        if metadata['attributes'][col]['type'] in ['real', 'integer']:
            # errors='coerce' option means any value which can’t be parsed as a number (e.g. '?' or other text) 
            # will be set to NaN instead of raising an error
            values = pd.to_numeric(df[col], errors='coerce')
            assert values.min() >= 0.0
            assert values.max() <= 1.0
            assert metadata['attributes'][col]['range_source'] == 'normalized'


def test_ecoli_create_X_y_shape(ecoli_dataset_file):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file)
    X, y = create_X_y(df, input_features, output_features)
    assert X.shape[0] == df.shape[0]
    assert y.shape[0] == df.shape[0]
    assert X.shape[1] == len(input_features)

@pytest.mark.parametrize("row_idx, expected_instance, expected_class",  [
    (0, np.array([0.68, 0.49, 1.0, 0.5, 0.62, 0.55, 0.28]), 'negative'),
    (1, np.array([0.75, 0.84, 0.48, 0.5, 0.35, 0.52, 0.33]), 'negative'),
    (2, np.array([0.52, 0.44, 0.48, 0.5, 0.37, 0.36, 0.42]), 'negative'),
    (3, np.array([0.87, 0.49, 0.48, 0.5, 0.61, 0.76, 0.79]), 'positive'),
    (4, np.array([0.41, 0.51, 0.48, 0.5, 0.58, 0.2, 0.31]), 'negative'),
    (5, np.array([0.2, 0.46, 0.48, 0.5, 0.57, 0.78, 0.81]), 'negative'),
    (6, np.array([0.41, 0.51, 0.48, 0.5, 0.53, 0.75, 0.78]), 'negative'),
    (7, np.array([0.35, 0.34, 0.48, 0.5, 0.46, 0.3, 0.27]), 'negative'),
    (8, np.array([0.44, 0.35, 0.48, 0.5, 0.55, 0.55, 0.61]), 'negative'),
    (9, np.array([0.76, 0.71, 0.48, 0.5, 0.5, 0.71, 0.75]), 'positive'),
    (10, np.array([0.76, 0.41, 0.48, 0.5, 0.5, 0.59, 0.62]), 'positive'),
    (11, np.array([0.3, 0.37, 0.48, 0.5, 0.43, 0.18, 0.3]), 'negative'),
    (12, np.array([0.18, 0.3, 0.48, 0.5, 0.46, 0.24, 0.35]), 'negative'),
    (13, np.array([0.81, 0.52, 0.48, 0.5, 0.57, 0.78, 0.8]), 'positive'),
    (14, np.array([0.25, 0.4, 0.48, 0.5, 0.46, 0.44, 0.52]), 'negative'),
    (15, np.array([0.78, 0.44, 0.48, 0.5, 0.45, 0.73, 0.68]), 'positive'),
    (16, np.array([0.69, 0.59, 0.48, 0.5, 0.46, 0.44, 0.52]), 'negative'),
    (17, np.array([0.64, 0.66, 0.48, 0.5, 0.41, 0.39, 0.2]), 'negative'),
    (18, np.array([0.63, 0.75, 0.48, 0.5, 0.64, 0.73, 0.66]), 'negative'),
    (19, np.array([0.69, 0.66, 0.48, 0.5, 0.41, 0.5, 0.25]), 'negative')
])
def test_ecoli_create_X_y_data_correctness(ecoli_dataset_file, row_idx, expected_instance, expected_class):
    metadata, df, input_features, output_features = parse_keel_file(ecoli_dataset_file, one_hot_encode=False, normalize=False)
    
    actual_class = df.iloc[row_idx, -1]
    actual_vector = df.iloc[row_idx, :-1].astype(float).to_numpy()
    assert np.allclose(actual_vector, expected_instance), (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_vector}\n"
        f"  expected = {expected_instance}"
    )
    assert actual_class == expected_class, (
        f"Row {row_idx} mismatch:\n"
        f"  actual   = {actual_class}\n"
        f"  expected = {expected_class}"
    )
    
    assert True 
    
