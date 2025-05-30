import pandas as pd
import numpy as np
import os
import re
import warnings
from sklearn.preprocessing import LabelEncoder
from typing import List, Tuple, Dict



def parse_keel_file(file_path, one_hot_encode=False, normalize=True):
    """
    @brief Parses a KEEL dataset file (.dat) into a structured DataFrame.
    NOTE: This method expects each keel dataset file has just one class attribute. NOt more than that

    @param file_path Path to the KEEL .dat file.
    @param one_hot_encode If True, perform one-hot encoding on categorical input features. (default: True)
    @param normalize If True, normalize numerical input features to [0, 1]. (default: True)

    @return
    Returns a tuple containing:
        - metadata (dict): Dataset metadata.
        - df (pd.DataFrame): The parsed dataset with the output feature(s) at the end.
        - input_features (list[str]): Names of input (non-class) features.
        - output_features (list[str]): Names of output (class) features.

    @exception ValueError If the file has a missing @data section or has unrecognized attribute definitions.
    @exception FileNotFoundError If the specified file does not exist.
    @exception pd.errors.ParserError If the data section cannot be parsed into a DataFrame.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('%')]

    relation = None
    attributes = {}
    input_features = []
    output_features = []
    in_data_section = False
    data = []
    range_source_info = {}

    for idx, line in enumerate(lines):
        if line.lower().startswith('@relation'):
            # NOTE: relation name might be different than the file name
            relation = line.split()[1]

        elif line.lower().startswith('@attribute'):
            parts = re.split(r'\s+', line, maxsplit=2)
            name = parts[1]
            definition = parts[2]

            if definition.startswith('integer') or definition.startswith('real'):
                attr_type = 'real' if 'real' in definition else 'integer'
                match = re.search(r'\[(.*),(.*)\]', definition)
                if match:
                    min_val = float(match.group(1))
                    max_val = float(match.group(2))
                    source = 'provided in dataset'
                else:
                    min_val = max_val = None
                    source = 'from data'
                attributes[name] = {
                    'type': attr_type,
                    'min_range': min_val,
                    'max_range': max_val,
                    'values': None,
                    'range_source': source
                }
            elif definition.startswith('{'):
                values = [v.strip() for v in definition.strip('{}').split(',')]
                attributes[name] = {
                    'type': 'nominal',
                    'min_range': None,
                    'max_range': None,
                    'values': set(values),
                    'range_source': 'provided in dataset'
                }
            else:
                raise ValueError(f"Unknown attribute type in line: {line}")

        elif line.lower().startswith('@inputs'):
            input_features = [x.strip() for x in line.split(' ', 1)[1].split(',')]

        elif line.lower().startswith('@outputs'):
            output_features = [x.strip() for x in line.split(' ', 1)[1].split(',')]

        elif line.lower() == '@data':
            in_data_section = True
            data_start_idx = idx + 1
            break

    if not in_data_section:
        raise ValueError("No data section found in file")

    # Read data
    data_lines = lines[data_start_idx:]
    data = [re.split(r'\s*,\s*', line.strip()) for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=list(attributes.keys()))

    # Default inputs/outputs if not defined
    if not input_features and not output_features:
        input_features = list(df.columns[:-1])
        output_features = [df.columns[-1]]

    # Reorder columns so output is at the end
    class_col = output_features[0]
    other_cols = [col for col in df.columns if col != class_col]
    df = df[other_cols + [class_col]]
    input_features = [col for col in input_features if col != class_col]
    input_features = [col for col in df.columns if col != class_col]
    output_features = [class_col]

    # Infer missing min/max if needed
    for col, meta in attributes.items():
        if meta['type'] in ['real', 'integer'] and meta['min_range'] is None:
            numeric = pd.to_numeric(df[col].replace('?', pd.NA), errors='coerce')
            attributes[col]['min_range'] = numeric.min()
            attributes[col]['max_range'] = numeric.max()
            attributes[col]['range_source'] = 'from data'

        if meta['type'] == 'nominal' and meta['values'] is None:
            attributes[col]['values'] = set(df[col].dropna().unique())
            attributes[col]['range_source'] = 'from data'

    # Handle missing values
    total_missing = df.isin(['?']).sum().sum()
    rows_with_missing = (df == '?').any(axis=1).sum()

    # Class distribution and imbalance
    class_counts = df[class_col].value_counts().to_dict()
    imbalance_ratio = None
    if len(class_counts) == 2:
        counts = list(class_counts.values())
        imbalance_ratio = max(counts) / min(counts)

    metadata = {
        'relation': relation,
        'file_path': file_path,
        'attributes': attributes,
        'num_input_features': len(input_features),
        'num_output_features': len(output_features),
        'num_instances': len(df),
        'class_distribution': class_counts,
        'imbalance_ratio': imbalance_ratio,
        'missing_values_total': total_missing,
        'instances_with_missing': rows_with_missing
    }

    if one_hot_encode:
        metadata, df, input_features, output_features = _apply_one_hot_encoding(
            metadata, df, input_features, output_features
        )

    if normalize:
        metadata, df = _normalize_data(metadata, df, input_features)

    df, label_encoders, mappings = _encode_class_columns(df, output_features)
    return metadata, df, input_features, output_features

# TODO: NOT TESTED. TEST IT
def _apply_one_hot_encoding(metadata, df, input_features, output_features):
    """
    @brief One-hot encodes all categorical input features and updates metadata.

    @param metadata Dictionary of dataset metadata.
    @param df The original DataFrame.
    @param input_features List of column names for input features.
    @param output_features List of column names for output features.

    @return A tuple containing:
        - updated_metadata (dict): Metadata updated for new one-hot columns.
        - new_df (pd.DataFrame): Transformed DataFrame with binary features.
        - new_input_features (list[str]): Updated input feature names.
        - output_features (list[str]): Same as input.

    @exception KeyError If a specified input feature is missing in metadata.
    """
    new_df = df.copy()
    updated_metadata = metadata.copy()
    updated_metadata['attributes'] = metadata['attributes'].copy()
    new_input_features = input_features.copy()
    class_col = output_features[0]

    for feature in input_features:
        attr_info = metadata['attributes'][feature]
        if attr_info['type'] == 'nominal':
            # Generate one-hot columns
            dummies = pd.get_dummies(new_df[feature], prefix=feature, dummy_na=False)
            new_df.drop(columns=[feature], inplace=True)
            new_df = pd.concat([new_df, dummies], axis=1)
            new_input_features.remove(feature)

            for col in dummies.columns:
                updated_metadata['attributes'][col] = {
                    'type': 'binary',
                    'min_range': 0,
                    'max_range': 1,
                    'values': {0, 1},
                    'range_source': 'generated by one-hot'
                }
                new_input_features.append(col)

            del updated_metadata['attributes'][feature]

    # Move class column to the end again
    class_series = new_df[class_col]
    new_df = new_df.drop(columns=[class_col])
    new_df[class_col] = class_series

    updated_metadata['num_input_features'] = len(new_input_features)
    updated_metadata['num_output_features'] = len(output_features)

    return updated_metadata, new_df, new_input_features, output_features


def _normalize_data(metadata, df, input_features):
    """
    @brief Normalizes all numerical input features to the range [0, 1].
    NOTE: Output features will not be normalized.
    NOTE: min and max are gotten from metadata in the KEEL dataset file header; not fromdata itself.
    NOTE: if min and max calculated from data are different than those in metadata, 
    an warning will be printed to the console and normalization will be performed 
    based on the metadata values.

    @param metadata Dictionary of dataset metadata (modified in-place).
    @param df DataFrame containing the dataset.
    @param input_features List of input feature names to normalize.

    @return A tuple:
        - updated_metadata (dict): Metadata with updated range values.
        - normalized_df (pd.DataFrame): DataFrame with normalized features.

    @exception KeyError If a feature is not found in metadata.
    @exception ValueError If a feature has invalid min/max values for normalization.
    """
    normalized_df = df.copy()
    for feature in input_features:
        attr_info = metadata['attributes'].get(feature)
        if attr_info and attr_info['type'] in ['real', 'integer']:
            # errors='coerce' option means any value which can’t be parsed as a number (e.g. '?' or other text) 
            # will be set to NaN instead of raising an error
            col_data = pd.to_numeric(df[feature], errors='coerce')
            min_val, max_val = col_data.min(), col_data.max()
            
            # TODO: Think about this warning
            # if ((min_val != attr_info['min_range']) or (max_val != attr_info['max_range'])):
            #     warnings.warn(f"calculated min/max values for feature '{feature}' in dataset '{metadata['file_path']}' are different than those in metadata (KEEL dataset file header). Normalized based on the information inside metadata.",
            #     category=UserWarning,
            #     stacklevel=2
            #     )
                
            min_val = attr_info['min_range']
            max_val = attr_info['max_range']
                
            if pd.notnull(min_val) and pd.notnull(max_val) and max_val > min_val:
                normalized_df[feature] = (col_data - min_val) / (max_val - min_val)
                metadata['attributes'][feature]['min_range'] = 0.0
                metadata['attributes'][feature]['max_range'] = 1.0
                metadata['attributes'][feature]['range_source'] = 'normalized'
    return metadata, normalized_df

# TODO: NOTE: Not tested. TEST it
def _encode_class_columns(df: pd.DataFrame, output_features: List[str]) -> Tuple[pd.DataFrame, List[LabelEncoder], Dict[str, Dict[str, int]]]:
    """
    @brief Encodes specified class columns in a DataFrame to integers using LabelEncoder.

    @param df: Input DataFrame with string class columns.
    @param output_features: List of column names to be encoded.

    @return: 
        - Transformed DataFrame with encoded class labels.
        - List of fitted LabelEncoders in order of output_features.
        - Dictionary of mappings {column_name: {original_label: int_value}}
    """
    df = df.copy()
    label_encoders = []
    mappings = {}

    for col in output_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders.append(le)
        mappings[col] = {original: int_val for int_val, original in enumerate(le.classes_)}

    return df, label_encoders, mappings

def create_X_y(df: pd.DataFrame, input_features: list[str], output_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Extracts input and output arrays from a KEEL dataset DataFrame.
    NOTE: This function assumes that there is just one target variable in the dataset.
    
    @param df The full parsed DataFrame returned by parse_keel_file().
    @param input_features List of column names for input features.
    @param output_features List of column names for output (target) features.

    @return A tuple:
        - X (np.ndarray): Numpy array of input features.
        - y (np.ndarray): Numpy array of target labels.

    @exception ValueError If output_features is empty or contains more than one column.
    """
    if not output_features:
        raise ValueError("Output feature list is empty. Cannot extract target variable.")
    if len(output_features) > 1:
        raise ValueError("Multiple output features detected. This function supports single-target datasets only.")

    X = df[input_features].to_numpy()
    y = df[output_features[0]].to_numpy()
    return X, y
