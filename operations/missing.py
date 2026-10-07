"""
Missing Value Operations Module (M2)

Provides reusable functions for detecting, dropping, and filling
missing values in Pandas DataFrames. All functions return new
DataFrames — the original is never modified.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_missing(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect missing values across all columns.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with:
            - total_missing: int — grand total of missing cells
            - columns: dict mapping column name → {count, percentage}
    """
    if df.empty:
        return {"total_missing": 0, "columns": {}}

    missing_counts = df.isnull().sum()
    total_rows = len(df)
    columns_info: Dict[str, Dict[str, Any]] = {}

    for col in df.columns:
        count = int(missing_counts[col])
        if count > 0:
            columns_info[col] = {
                "count": count,
                "percentage": round((count / total_rows) * 100, 2),
            }

    return {
        "total_missing": int(missing_counts.sum()),
        "columns": columns_info,
    }


# ---------------------------------------------------------------------------
# Drop helpers
# ---------------------------------------------------------------------------

def drop_missing_rows(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Drop rows containing missing values.

    Args:
        df: Input DataFrame.
        columns: If provided, only consider these columns when deciding
                 which rows to drop.  If *None*, consider all columns.

    Returns:
        New DataFrame with the relevant rows removed.

    Raises:
        ValueError: If any column in *columns* does not exist in *df*.
    """
    df = df.copy()

    if columns is not None:
        _validate_columns_exist(df, columns)
        df = df.dropna(subset=columns)
    else:
        df = df.dropna()

    return df.reset_index(drop=True)


def drop_missing_columns(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """Drop explicitly listed columns from the DataFrame.

    Only the columns the caller specifies are removed — this function
    will **not** automatically scan for columns with missing values.

    Args:
        df: Input DataFrame.
        columns: Column names to remove.

    Returns:
        New DataFrame without the specified columns.

    Raises:
        ValueError: If any column in *columns* does not exist in *df*.
    """
    df = df.copy()
    _validate_columns_exist(df, columns)
    df = df.drop(columns=columns)
    return df


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------

def fill_missing_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values in *column* with the column mean.

    Args:
        df: Input DataFrame.
        column: Target column name.

    Returns:
        New DataFrame with missing values in *column* filled.

    Raises:
        ValueError: If *column* is not numeric or does not exist.
    """
    df = df.copy()
    _validate_columns_exist(df, [column])
    _validate_numeric_column(df, column, "mean")

    mean_value = df[column].mean()
    df[column] = df[column].fillna(mean_value)
    return df


def fill_missing_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values in *column* with the column median.

    Args:
        df: Input DataFrame.
        column: Target column name.

    Returns:
        New DataFrame with missing values in *column* filled.

    Raises:
        ValueError: If *column* is not numeric or does not exist.
    """
    df = df.copy()
    _validate_columns_exist(df, [column])
    _validate_numeric_column(df, column, "median")

    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)
    return df


def fill_missing_mode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Fill missing values in *column* with the column mode.

    Works for both numerical and categorical columns.

    Args:
        df: Input DataFrame.
        column: Target column name.

    Returns:
        New DataFrame with missing values in *column* filled.

    Raises:
        ValueError: If *column* does not exist or the mode cannot be
                     computed (e.g. column is entirely NaN).
    """
    df = df.copy()
    _validate_columns_exist(df, [column])

    mode_series = df[column].mode()
    if mode_series.empty:
        raise ValueError(
            f"Cannot compute mode for column '{column}' — "
            "all values may be missing."
        )

    mode_value = mode_series.iloc[0]
    df[column] = df[column].fillna(mode_value)
    return df


def fill_missing_forward(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """Forward-fill (propagate last valid value) missing values.

    Args:
        df: Input DataFrame.
        column: If provided, only forward-fill this column.
                If *None*, forward-fill the entire DataFrame.

    Returns:
        New DataFrame with forward-filled values.

    Raises:
        ValueError: If *column* does not exist in *df*.
    """
    df = df.copy()

    if column is not None:
        _validate_columns_exist(df, [column])
        df[column] = df[column].ffill()
    else:
        df = df.ffill()

    return df


def fill_missing_backward(
    df: pd.DataFrame,
    column: Optional[str] = None,
) -> pd.DataFrame:
    """Backward-fill (propagate next valid value) missing values.

    Args:
        df: Input DataFrame.
        column: If provided, only backward-fill this column.
                If *None*, backward-fill the entire DataFrame.

    Returns:
        New DataFrame with backward-filled values.

    Raises:
        ValueError: If *column* does not exist in *df*.
    """
    df = df.copy()

    if column is not None:
        _validate_columns_exist(df, [column])
        df[column] = df[column].bfill()
    else:
        df = df.bfill()

    return df


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------

def _validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    """Raise *ValueError* if any column is not in *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}"
        )


def _validate_numeric_column(
    df: pd.DataFrame,
    column: str,
    method: str,
) -> None:
    """Raise *ValueError* if *column* is not numeric."""
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Cannot calculate {method} for column '{column}' "
            "because it is not numerical."
        )
