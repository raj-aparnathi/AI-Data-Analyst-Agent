"""
operations/statistics.py — Statistical Operations (M4)
=======================================================
Provides basic statistical functions that operate on individual
DataFrame columns.  Each function validates its input and returns
a structured result.

All functions are **read-only** — the source DataFrame is never modified.

Author : Member 4 (Analysis + Visualization)
"""

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if *column* is not in *df*."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_numeric_column(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if *column* is not numeric."""
    _validate_column_exists(df, column)
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            f"Column '{column}' is not numeric (dtype: {df[column].dtype}). "
            f"This operation requires a numerical column."
        )


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------

def calculate_mean(df: pd.DataFrame, column: str) -> float:
    """Calculate the arithmetic mean of a numeric column.

    Args:
        df: Input DataFrame.
        column: Name of the numeric column.

    Returns:
        The mean value.

    Raises:
        ValueError: If the column does not exist or is not numeric.
    """
    _validate_numeric_column(df, column)
    result = df[column].mean()
    return float(result)


def calculate_median(df: pd.DataFrame, column: str) -> float:
    """Calculate the median of a numeric column.

    Args:
        df: Input DataFrame.
        column: Name of the numeric column.

    Returns:
        The median value.

    Raises:
        ValueError: If the column does not exist or is not numeric.
    """
    _validate_numeric_column(df, column)
    result = df[column].median()
    return float(result)


def calculate_mode(df: pd.DataFrame, column: str) -> Union[Any, List[Any]]:
    """Return the most frequently occurring value(s) in a column.

    Works for both numerical and categorical columns.

    Args:
        df: Input DataFrame.
        column: Name of the column.

    Returns:
        A single value if there is one mode, or a list if there are
        multiple modes.

    Raises:
        ValueError: If the column does not exist.
    """
    _validate_column_exists(df, column)
    modes = df[column].mode()
    if len(modes) == 0:
        return None
    if len(modes) == 1:
        # Convert numpy types to Python native types
        val = modes.iloc[0]
        return val.item() if hasattr(val, "item") else val
    # Multiple modes
    return [v.item() if hasattr(v, "item") else v for v in modes]


def calculate_min(df: pd.DataFrame, column: str) -> Any:
    """Return the minimum value of a column.

    Works for numeric and orderable columns.

    Args:
        df: Input DataFrame.
        column: Name of the column.

    Returns:
        The minimum value.

    Raises:
        ValueError: If the column does not exist.
    """
    _validate_column_exists(df, column)
    result = df[column].min()
    return result.item() if hasattr(result, "item") else result


def calculate_max(df: pd.DataFrame, column: str) -> Any:
    """Return the maximum value of a column.

    Args:
        df: Input DataFrame.
        column: Name of the column.

    Returns:
        The maximum value.

    Raises:
        ValueError: If the column does not exist.
    """
    _validate_column_exists(df, column)
    result = df[column].max()
    return result.item() if hasattr(result, "item") else result


def calculate_count(df: pd.DataFrame, column: str) -> int:
    """Return the number of valid (non-null) values in a column.

    Args:
        df: Input DataFrame.
        column: Name of the column.

    Returns:
        The count of non-null values.

    Raises:
        ValueError: If the column does not exist.
    """
    _validate_column_exists(df, column)
    return int(df[column].count())


def calculate_variance(df: pd.DataFrame, column: str) -> float:
    """Calculate the variance of a numeric column.

    Uses sample variance (ddof=1) by default, consistent with Pandas.

    Args:
        df: Input DataFrame.
        column: Name of the numeric column.

    Returns:
        The variance.

    Raises:
        ValueError: If the column does not exist or is not numeric.
    """
    _validate_numeric_column(df, column)
    result = df[column].var()
    return float(result)


def calculate_std(df: pd.DataFrame, column: str) -> float:
    """Calculate the standard deviation of a numeric column.

    Uses sample standard deviation (ddof=1) by default.

    Args:
        df: Input DataFrame.
        column: Name of the numeric column.

    Returns:
        The standard deviation.

    Raises:
        ValueError: If the column does not exist or is not numeric.
    """
    _validate_numeric_column(df, column)
    result = df[column].std()
    return float(result)


def calculate_quartiles(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate Q1, Q2 (median), and Q3 for a numeric column.

    Args:
        df: Input DataFrame.
        column: Name of the numeric column.

    Returns:
        Dictionary with keys "Q1", "Q2", "Q3".

    Raises:
        ValueError: If the column does not exist or is not numeric.
    """
    _validate_numeric_column(df, column)
    q1 = float(df[column].quantile(0.25))
    q2 = float(df[column].quantile(0.50))
    q3 = float(df[column].quantile(0.75))
    return {"Q1": q1, "Q2": q2, "Q3": q3}


# ---------------------------------------------------------------------------
# Dispatcher — maps operation names to functions
# ---------------------------------------------------------------------------

STATISTICS_OPERATIONS = {
    "mean": calculate_mean,
    "median": calculate_median,
    "mode": calculate_mode,
    "min": calculate_min,
    "max": calculate_max,
    "count": calculate_count,
    "variance": calculate_variance,
    "std": calculate_std,
    "quartiles": calculate_quartiles,
}


def execute_statistic(
    df: pd.DataFrame,
    operation: str,
    column: str,
) -> Any:
    """Execute a named statistical operation on a column.

    Args:
        df: Input DataFrame.
        operation: One of the keys in STATISTICS_OPERATIONS.
        column: Target column name.

    Returns:
        The computed result.

    Raises:
        ValueError: If the operation or column is invalid.
    """
    if operation not in STATISTICS_OPERATIONS:
        raise ValueError(
            f"Unsupported statistical operation: '{operation}'. "
            f"Supported: {sorted(STATISTICS_OPERATIONS.keys())}"
        )
    return STATISTICS_OPERATIONS[operation](df, column)
