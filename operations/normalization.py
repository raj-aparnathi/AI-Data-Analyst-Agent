"""Normalization operations."""

import pandas as pd


def normalize(df, column, method):
    """Normalize a numeric column using min-max or z-score."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not numeric and cannot be normalized.")

    if method == "minmax":
        # x' = (x - min) / (max - min)
        col_min = df[column].min()
        col_max = df[column].max()
        if col_max == col_min:
            df[column] = 0.0
        else:
            df[column] = (df[column] - col_min) / (col_max - col_min)
    elif method == "zscore":
        # z = (x - mean) / standard deviation
        std = df[column].std()
        if std == 0 or pd.isna(std):
            df[column] = 0.0
        else:
            df[column] = (df[column] - df[column].mean()) / std
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Use minmax or zscore.")

    return df
