"""Missing value operations."""

import pandas as pd


def drop_missing(df):
    """Remove all rows that contain any missing value."""
    return df.dropna()


def fill_missing(df, column, method):
    """Fill missing values in a column using mean, median, or mode."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")

    if method in ("mean", "median"):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numeric, so {method} cannot be used.")
        value = df[column].mean() if method == "mean" else df[column].median()
        df[column] = df[column].fillna(value)
    elif method == "mode":
        value = df[column].mode()
        fill_value = value.iloc[0] if not value.empty else None
        df[column] = df[column].fillna(fill_value)
    else:
        raise ValueError(f"Unknown fill method '{method}'. Use mean, median, or mode.")

    return df
