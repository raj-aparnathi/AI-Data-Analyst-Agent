"""
operations/analysis.py — Data Analysis Operations (M4)
=======================================================
Provides data-analysis functions: GroupBy, Sorting, Filtering,
Top-N, Correlation, Trend Analysis, and a central plan executor.

All functions are **read-only** — the source DataFrame is never modified.

Author : Member 4 (Analysis + Visualization)
"""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from operations.statistics import STATISTICS_OPERATIONS, execute_statistic


# ---------------------------------------------------------------------------
# Supported operations whitelist
# ---------------------------------------------------------------------------

SUPPORTED_ANALYSIS_OPERATIONS = {
    "mean",
    "median",
    "mode",
    "min",
    "max",
    "count",
    "variance",
    "std",
    "quartiles",
    "groupby",
    "sort",
    "filter",
    "top_n",
    "correlation",
    "trend",
}

SUPPORTED_AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count"}

SUPPORTED_FILTER_OPERATORS = {">", "<", ">=", "<=", "==", "!="}


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


def _validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    """Raise ValueError if any column in *columns* is not in *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# GroupBy
# ---------------------------------------------------------------------------

def group_by(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Group by a column and aggregate a value column.

    Args:
        df: Input DataFrame.
        group_column: Column to group by.
        value_column: Column to aggregate.
        aggregation: One of 'sum', 'mean', 'median', 'min', 'max', 'count'.

    Returns:
        A DataFrame with group_column and the aggregated value_column.

    Raises:
        ValueError: If columns or aggregation are invalid.
    """
    _validate_column_exists(df, group_column)
    _validate_column_exists(df, value_column)

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"Unsupported aggregation: '{aggregation}'. "
            f"Supported: {sorted(SUPPORTED_AGGREGATIONS)}"
        )

    working = df[[group_column, value_column]].copy()

    if aggregation == "median":
        result = working.groupby(group_column, as_index=False)[value_column].median()
    else:
        result = working.groupby(group_column, as_index=False)[value_column].agg(aggregation)

    return result


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def sort_data(
    df: pd.DataFrame,
    column: str,
    ascending: bool = True,
) -> pd.DataFrame:
    """Sort the DataFrame by a column.

    Args:
        df: Input DataFrame.
        column: Column to sort by.
        ascending: True for ascending, False for descending.

    Returns:
        A sorted copy of the DataFrame.

    Raises:
        ValueError: If the column does not exist.
    """
    _validate_column_exists(df, column)
    return df.copy().sort_values(by=column, ascending=ascending).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_data(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: Any,
) -> pd.DataFrame:
    """Filter rows based on a column comparison.

    Only safe comparison operators are supported — NO eval() / exec().

    Args:
        df: Input DataFrame.
        column: Column to filter on.
        operator: One of '>', '<', '>=', '<=', '==', '!='.
        value: Value to compare against.

    Returns:
        A filtered copy of the DataFrame.

    Raises:
        ValueError: If the column or operator is invalid.
    """
    _validate_column_exists(df, column)

    if operator not in SUPPORTED_FILTER_OPERATORS:
        raise ValueError(
            f"Unsupported filter operator: '{operator}'. "
            f"Supported: {sorted(SUPPORTED_FILTER_OPERATORS)}"
        )

    series = df[column]

    # Attempt to cast value to the column's dtype for proper comparison
    try:
        if pd.api.types.is_numeric_dtype(series):
            value = float(value)
        elif pd.api.types.is_datetime64_any_dtype(series):
            value = pd.to_datetime(value)
    except (ValueError, TypeError):
        pass  # Keep value as-is for string comparisons

    if operator == ">":
        mask = series > value
    elif operator == "<":
        mask = series < value
    elif operator == ">=":
        mask = series >= value
    elif operator == "<=":
        mask = series <= value
    elif operator == "==":
        mask = series == value
    elif operator == "!=":
        mask = series != value
    else:
        raise ValueError(f"Unsupported operator: '{operator}'")

    return df[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-N
# ---------------------------------------------------------------------------

def top_n(
    df: pd.DataFrame,
    column: str,
    n: int,
    ascending: bool = False,
) -> pd.DataFrame:
    """Return the top-N rows based on a column.

    Args:
        df: Input DataFrame.
        column: Column to rank by.
        n: Number of rows to return.
        ascending: If False (default), return highest values first.

    Returns:
        A DataFrame with the top-N rows.

    Raises:
        ValueError: If the column does not exist or n is invalid.
    """
    _validate_column_exists(df, column)

    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"'n' must be a positive integer, got: {n}")

    # Clamp n to available rows
    n = min(n, len(df))

    return (
        df.copy()
        .sort_values(by=column, ascending=ascending)
        .head(n)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def calculate_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate a Pearson correlation matrix.

    Args:
        df: Input DataFrame.
        columns: Specific columns to include.  If None, all numeric
                 columns are used.

    Returns:
        A correlation matrix as a DataFrame.

    Raises:
        ValueError: If specified columns don't exist or are not numeric.
    """
    if columns is not None:
        _validate_columns_exist(df, columns)
        numeric_cols = [
            c for c in columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric_cols) < 2:
            raise ValueError(
                "Correlation requires at least 2 numeric columns. "
                f"Found numeric: {numeric_cols}"
            )
        working = df[numeric_cols].copy()
    else:
        working = df.select_dtypes(include=[np.number]).copy()
        if working.shape[1] < 2:
            raise ValueError(
                "Correlation requires at least 2 numeric columns in the dataset."
            )

    return working.corr()


# ---------------------------------------------------------------------------
# Trend Analysis
# ---------------------------------------------------------------------------

def analyze_trend(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    frequency: str = "monthly",
) -> Dict[str, Any]:
    """Perform basic trend analysis on a time series.

    Args:
        df: Input DataFrame.
        date_column: Column containing date/datetime values.
        value_column: Numeric column to analyse.
        frequency: One of 'daily', 'monthly', 'yearly'.

    Returns:
        A dictionary with trend information:
        - direction: "increasing", "decreasing", or "stable"
        - start_value: first aggregated value
        - end_value: last aggregated value
        - change: absolute change (end - start)
        - change_percent: percentage change
        - trend_data: aggregated DataFrame

    Raises:
        ValueError: If columns are invalid.
    """
    _validate_column_exists(df, date_column)
    _validate_column_exists(df, value_column)

    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError(
            f"Value column '{value_column}' must be numeric for trend analysis."
        )

    working = df[[date_column, value_column]].copy()

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(working[date_column]):
        try:
            working[date_column] = pd.to_datetime(working[date_column])
        except Exception:
            raise ValueError(
                f"Column '{date_column}' could not be converted to datetime."
            )

    working = working.dropna(subset=[date_column, value_column])
    working = working.sort_values(date_column)

    # Determine aggregation frequency
    freq_map = {
        "daily": "D",
        "monthly": "MS",
        "yearly": "YS",
    }
    if frequency not in freq_map:
        raise ValueError(
            f"Unsupported frequency: '{frequency}'. "
            f"Supported: {list(freq_map.keys())}"
        )

    working = working.set_index(date_column)
    aggregated = working[value_column].resample(freq_map[frequency]).sum().reset_index()
    aggregated.columns = [date_column, value_column]

    if len(aggregated) == 0:
        return {
            "direction": "unknown",
            "start_value": None,
            "end_value": None,
            "change": None,
            "change_percent": None,
            "trend_data": aggregated,
        }

    start_value = float(aggregated[value_column].iloc[0])
    end_value = float(aggregated[value_column].iloc[-1])
    change = end_value - start_value

    if start_value != 0:
        change_percent = round((change / abs(start_value)) * 100, 2)
    else:
        change_percent = None

    # Determine direction using simple linear comparison
    if change > 0:
        direction = "increasing"
    elif change < 0:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "start_value": start_value,
        "end_value": end_value,
        "change": change,
        "change_percent": change_percent,
        "trend_data": aggregated,
    }


# ---------------------------------------------------------------------------
# Central Analysis Plan Executor
# ---------------------------------------------------------------------------

def apply_analysis_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a structured analysis plan on *df*.

    The plan is a dictionary produced by the AI planner.  Examples::

        {"operation": "mean", "column": "salary"}

        {"operation": "groupby", "group_column": "product",
         "value_column": "sales", "aggregation": "sum",
         "sort": "descending", "limit": 5}

    Args:
        df: Input DataFrame (will **not** be modified).
        plan: Structured analysis plan dictionary.

    Returns:
        A result dictionary with keys:
        - operation: the operation performed
        - result: the computed result (scalar, DataFrame, dict, etc.)
        - summary: metadata about the result
        - visualization: recommended chart type (or None)
        - result_type: "scalar", "dataframe", "dict", "matrix"

    Raises:
        ValueError: If the plan is invalid.
    """
    operation = plan.get("operation")
    if not operation:
        raise ValueError("Analysis plan must include an 'operation' key.")

    if operation not in SUPPORTED_ANALYSIS_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: '{operation}'. "
            f"Supported: {sorted(SUPPORTED_ANALYSIS_OPERATIONS)}"
        )

    visualization = plan.get("visualization")

    # ----- Statistical operations (single column) -------------------------
    if operation in STATISTICS_OPERATIONS:
        column = plan.get("column")
        if not column:
            raise ValueError(
                f"Operation '{operation}' requires a 'column' parameter."
            )
        _validate_column_exists(df, column)

        result_value = execute_statistic(df, operation, column)

        result_type = "dict" if isinstance(result_value, dict) else "scalar"

        return {
            "operation": operation,
            "result": result_value,
            "summary": {
                "column": column,
                "operation": operation,
            },
            "visualization": visualization,
            "result_type": result_type,
        }

    # ----- GroupBy --------------------------------------------------------
    if operation == "groupby":
        group_column = plan.get("group_column")
        value_column = plan.get("value_column")
        aggregation = plan.get("aggregation", "sum")

        if not group_column or not value_column:
            raise ValueError(
                "GroupBy requires 'group_column' and 'value_column'."
            )

        result_df = group_by(df, group_column, value_column, aggregation)

        # Optional sorting
        sort_order = plan.get("sort")
        if sort_order:
            ascending = sort_order.lower() != "descending"
            result_df = result_df.sort_values(
                by=value_column, ascending=ascending
            ).reset_index(drop=True)

        # Optional limit
        limit = plan.get("limit")
        if limit and isinstance(limit, int) and limit > 0:
            result_df = result_df.head(limit).reset_index(drop=True)

        # Default visualization for groupby
        if visualization is None:
            visualization = "bar"

        return {
            "operation": "groupby",
            "result": result_df,
            "summary": {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "group_column": group_column,
                "value_column": value_column,
                "aggregation": aggregation,
            },
            "visualization": visualization,
            "result_type": "dataframe",
        }

    # ----- Sort -----------------------------------------------------------
    if operation == "sort":
        column = plan.get("column")
        if not column:
            raise ValueError("Sort requires a 'column' parameter.")

        ascending = plan.get("ascending", True)
        if isinstance(ascending, str):
            ascending = ascending.lower() != "descending"

        sort_order_str = plan.get("sort")
        if sort_order_str and isinstance(sort_order_str, str):
            ascending = sort_order_str.lower() != "descending"

        result_df = sort_data(df, column, ascending=ascending)

        limit = plan.get("limit")
        if limit and isinstance(limit, int) and limit > 0:
            result_df = result_df.head(limit).reset_index(drop=True)

        return {
            "operation": "sort",
            "result": result_df,
            "summary": {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "sort_column": column,
                "ascending": ascending,
            },
            "visualization": visualization,
            "result_type": "dataframe",
        }

    # ----- Filter ---------------------------------------------------------
    if operation == "filter":
        column = plan.get("column")
        operator = plan.get("operator")
        value = plan.get("value")

        if not column or not operator:
            raise ValueError(
                "Filter requires 'column', 'operator', and 'value'."
            )

        result_df = filter_data(df, column, operator, value)

        return {
            "operation": "filter",
            "result": result_df,
            "summary": {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "filter": f"{column} {operator} {value}",
            },
            "visualization": visualization,
            "result_type": "dataframe",
        }

    # ----- Top-N ----------------------------------------------------------
    if operation == "top_n":
        column = plan.get("column")
        n = plan.get("n", 5)

        if not column:
            raise ValueError("Top-N requires a 'column' parameter.")

        ascending = plan.get("ascending", False)

        result_df = top_n(df, column, int(n), ascending=ascending)

        if visualization is None:
            visualization = "bar"

        return {
            "operation": "top_n",
            "result": result_df,
            "summary": {
                "rows": len(result_df),
                "columns": len(result_df.columns),
                "column": column,
                "n": int(n),
            },
            "visualization": visualization,
            "result_type": "dataframe",
        }

    # ----- Correlation ----------------------------------------------------
    if operation == "correlation":
        columns = plan.get("columns")

        corr_matrix = calculate_correlation(df, columns=columns)

        if visualization is None:
            visualization = "heatmap"

        return {
            "operation": "correlation",
            "result": corr_matrix,
            "summary": {
                "rows": corr_matrix.shape[0],
                "columns": corr_matrix.shape[1],
                "variables": list(corr_matrix.columns),
            },
            "visualization": visualization,
            "result_type": "matrix",
        }

    # ----- Trend Analysis -------------------------------------------------
    if operation == "trend":
        date_column = plan.get("date_column")
        value_column = plan.get("value_column")
        frequency = plan.get("frequency", "monthly")

        if not date_column or not value_column:
            raise ValueError(
                "Trend analysis requires 'date_column' and 'value_column'."
            )

        trend_result = analyze_trend(df, date_column, value_column, frequency)

        if visualization is None:
            visualization = "line"

        return {
            "operation": "trend",
            "result": trend_result,
            "summary": {
                "direction": trend_result["direction"],
                "start_value": trend_result["start_value"],
                "end_value": trend_result["end_value"],
                "change": trend_result["change"],
                "change_percent": trend_result["change_percent"],
            },
            "visualization": visualization,
            "result_type": "dict",
        }

    raise ValueError(f"Operation '{operation}' is not implemented.")
