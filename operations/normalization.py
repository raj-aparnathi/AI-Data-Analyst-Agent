"""
operations/normalization.py - Normalization Operations (M3)
============================================================
Provides numerical normalization/standardization:
    - min_max_normalize : Scale column values to [0, 1].
    - z_score_normalize : Standardize via mean/std.

Design principles
-----------------
- df.copy() is always used; original is never modified.
- Categorical columns are rejected with a clear error.
- Constant columns (std=0) are handled without crashing.
"""

import warnings

import pandas as pd


def min_max_normalize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Scale a numerical column to the range [0, 1].

    Formula: (x - min) / (max - min)

    If all values are identical (min == max), results are 0.0
    to avoid division by zero.

    Args:
        df: Input DataFrame (not modified).
        column: Numerical column to normalize.

    Returns:
        DataFrame with the column replaced by its scaled values.

    Raises:
        ValueError: If column does not exist or is not numerical.
    """
    _validate_numerical_column(df, column)
    df_out = df.copy()
    col_min = df_out[column].min()
    col_max = df_out[column].max()
    col_range = col_max - col_min
    if col_range == 0:
        df_out[column] = 0.0
    else:
        df_out[column] = (df_out[column] - col_min) / col_range
    return df_out


def z_score_normalize(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Standardize a numerical column using Z-score normalization.

    Formula: (x - mean) / std

    If the column is constant (std=0), all values become 0.0 and
    a UserWarning is issued instead of raising ZeroDivisionError.

    Args:
        df: Input DataFrame (not modified).
        column: Numerical column to standardize.

    Returns:
        DataFrame with the column replaced by standardized values.

    Raises:
        ValueError: If column does not exist or is not numerical.
    """
    _validate_numerical_column(df, column)
    df_out = df.copy()
    col_mean = df_out[column].mean()
    col_std = df_out[column].std()
    if col_std == 0:
        warnings.warn(
            f"Column '{column}' is constant (std=0). "
            "Z-score normalization returns 0.0 for all values.",
            UserWarning,
            stacklevel=2,
        )
        df_out[column] = 0.0
    else:
        df_out[column] = (df_out[column] - col_mean) / col_std
    return df_out


def _validate_numerical_column(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if column is missing or non-numerical."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numerical (dtype={df[column].dtype}). "
            "Normalization requires a numerical column."
        )
