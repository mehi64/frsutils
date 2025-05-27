import pandas as pd
import numpy as np
import os
import re


def parse_keel_file(file_path, one_hot_encode=True, normalize=True):
    """
    Parses a KEEL dataset file.

    Parameters:
        file_path (str): Path to the KEEL .dat file.
        one_hot_encode (bool): If True, perform one-hot encoding on non-decision categorical features.
        normalize (bool): If True, normalize numerical input features to [0, 1].

    Returns:
        metadata (dict): Dataset metadata.
        df (pd.DataFrame): Parsed dataset with class feature as the last column.
        input_features (list): Names of input features.
        output_features (list): Names of output features.
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

    return metadata, df, input_features, output_features


def _apply_one_hot_encoding(metadata, df, input_features, output_features):
    """
    One-hot encodes all non-decision (input) categorical features in the dataset.

    Parameters:
        metadata (dict): Metadata dictionary.
        df (pd.DataFrame): Dataset.
        input_features (list): Names of input features.
        output_features (list): Names of output features.

    Returns:
        updated_metadata (dict): Metadata after encoding.
        new_df (pd.DataFrame): Encoded dataset with class feature as the last column.
        new_input_features (list): Updated input feature names.
        output_features (list): Output features (unchanged).
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
    Normalize numerical input features to the [0, 1] range and update metadata accordingly.

    Parameters:
        metadata (dict): Metadata dictionary.
        df (pd.DataFrame): Dataset.
        input_features (list): Names of input features.

    Returns:
        updated_metadata (dict): Metadata with updated range info.
        normalized_df (pd.DataFrame): Dataset with normalized features.
    """
    normalized_df = df.copy()
    for feature in input_features:
        attr_info = metadata['attributes'].get(feature)
        if attr_info and attr_info['type'] in ['real', 'integer']:
            col_data = pd.to_numeric(df[feature], errors='coerce')
            min_val, max_val = col_data.min(), col_data.max()
            if pd.notnull(min_val) and pd.notnull(max_val) and max_val > min_val:
                normalized_df[feature] = (col_data - min_val) / (max_val - min_val)
                metadata['attributes'][feature]['min_range'] = 0.0
                metadata['attributes'][feature]['max_range'] = 1.0
                metadata['attributes'][feature]['range_source'] = 'normalized'
    return metadata, normalized_df


def create_X_y(df: pd.DataFrame, input_features: list[str], output_features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts features (X) and target (y) as NumPy arrays from the KEEL dataset DataFrame.

    Args:
        df (pd.DataFrame): The full dataset returned by parse_keel_file.
        input_features (list[str]): List of column names used as input features.
        output_features (list[str]): List of column names used as output targets.

    Returns:
        tuple:
            X (np.ndarray): NumPy array of input features.
            y (np.ndarray): NumPy array of target values.
    """
    if not output_features:
        raise ValueError("Output feature list is empty. Cannot extract target variable.")
    if len(output_features) > 1:
        raise ValueError("Multiple output features detected. This function supports single-target datasets only.")

    X = df[input_features].to_numpy()
    y = df[output_features[0]].to_numpy()
    return X, y