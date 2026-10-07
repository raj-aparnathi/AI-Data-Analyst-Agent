"""
General Cleaning Operations Module (M2)

Provides reusable functions for common data-cleaning tasks
(deduplication, whitespace handling, case conversion) and a central
*apply_cleaning_plan* executor that turns a structured JSON plan
into actual DataFrame transformations.

All functions return new DataFrames — the original is never modified.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from operations.missing import (
    detect_missing,
    drop_missing_columns,
    drop_missing_rows,
    fill_missing_backward,
    fill_missing_forward,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
)


# ---------------------------------------------------------------------------
# Whitelisted operations — anything else is rejected
# ---------------------------------------------------------------------------

SUPPORTED_OPERATIONS = {
    "remove_duplicates",
    "drop_missing_rows",
    "drop_missing_columns",
    "fill_missing",
    "strip_spaces",
    "remove_extra_spaces",
    "lowercase",
    "uppercase",
}

SUPPORTED_FILL_METHODS = {
    "mean",
    "median",
    "mode",
    "forward_fill",
    "backward_fill",
}


# ---------------------------------------------------------------------------
# Individual cleaning functions
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate rows.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, number of removed rows).
    """
    df = df.copy()
    rows_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = rows_before - len(df)
    return df, removed


def strip_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from all string columns.

    Args:
        df: Input DataFrame.

    Returns:
        New DataFrame with stripped string columns.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def remove_extra_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple consecutive spaces into a single space.

    Applies only to string/object columns.

    Args:
        df: Input DataFrame.

    Returns:
        New DataFrame with extra internal whitespace removed.
    """
    df = df.copy()
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].apply(
            lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v
        )
    return df


def to_lowercase(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert text to lowercase.

    Args:
        df: Input DataFrame.
        columns: Target columns.  If *None*, apply to all string columns.

    Returns:
        New DataFrame with lowered text.

    Raises:
        ValueError: If a specified column does not exist.
    """
    df = df.copy()

    if columns is not None:
        _validate_columns_exist(df, columns)
        target_cols = columns
    else:
        target_cols = list(
            df.select_dtypes(include=["object", "string"]).columns
        )

    for col in target_cols:
        df[col] = df[col].apply(lambda v: v.lower() if isinstance(v, str) else v)
    return df


def to_uppercase(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Convert text to uppercase.

    Args:
        df: Input DataFrame.
        columns: Target columns.  If *None*, apply to all string columns.

    Returns:
        New DataFrame with uppercased text.

    Raises:
        ValueError: If a specified column does not exist.
    """
    df = df.copy()

    if columns is not None:
        _validate_columns_exist(df, columns)
        target_cols = columns
    else:
        target_cols = list(
            df.select_dtypes(include=["object", "string"]).columns
        )

    for col in target_cols:
        df[col] = df[col].apply(lambda v: v.upper() if isinstance(v, str) else v)
    return df


# ---------------------------------------------------------------------------
# Plan Validation
# ---------------------------------------------------------------------------

def validate_cleaning_plan(
    plan: Dict[str, Any],
    df: pd.DataFrame,
) -> List[str]:
    """Validate a cleaning plan against the DataFrame.

    Returns a list of error messages.  An empty list means the plan
    is valid.
    """
    errors: List[str] = []

    if not isinstance(plan, dict):
        errors.append("Plan must be a dictionary.")
        return errors

    operations = plan.get("operations")
    if operations is None:
        errors.append("Plan is missing the 'operations' key.")
        return errors

    if not isinstance(operations, list):
        errors.append("'operations' must be a list.")
        return errors

    for idx, op in enumerate(operations):
        prefix = f"Operation #{idx + 1}"

        if not isinstance(op, dict):
            errors.append(f"{prefix}: each operation must be a dictionary.")
            continue

        op_type = op.get("type")
        if op_type is None:
            errors.append(f"{prefix}: missing 'type' key.")
            continue

        if op_type not in SUPPORTED_OPERATIONS:
            errors.append(
                f"{prefix}: unsupported operation type '{op_type}'."
            )
            continue

        # -- fill_missing requires column + method --------------------------
        if op_type == "fill_missing":
            if "column" not in op:
                errors.append(
                    f"{prefix} (fill_missing): missing 'column' parameter."
                )
            elif op["column"] not in df.columns:
                errors.append(
                    f"{prefix} (fill_missing): column '{op['column']}' "
                    "does not exist in the DataFrame."
                )

            method = op.get("method")
            if method is None:
                errors.append(
                    f"{prefix} (fill_missing): missing 'method' parameter."
                )
            elif method not in SUPPORTED_FILL_METHODS:
                errors.append(
                    f"{prefix} (fill_missing): unsupported method "
                    f"'{method}'."
                )
            elif method in ("mean", "median") and "column" in op and op["column"] in df.columns:
                if not pd.api.types.is_numeric_dtype(df[op["column"]]):
                    errors.append(
                        f"{prefix} (fill_missing): cannot calculate {method} for "
                        f"column '{op['column']}' because it is not numerical."
                    )

        # -- drop_missing_rows may specify columns --------------------------
        if op_type == "drop_missing_rows":
            cols = op.get("columns")
            if cols is not None:
                if not isinstance(cols, list):
                    errors.append(
                        f"{prefix} (drop_missing_rows): "
                        "'columns' must be a list."
                    )
                else:
                    for c in cols:
                        if c not in df.columns:
                            errors.append(
                                f"{prefix} (drop_missing_rows): "
                                f"column '{c}' does not exist."
                            )

        # -- drop_missing_columns requires columns --------------------------
        if op_type == "drop_missing_columns":
            cols = op.get("columns")
            if cols is None:
                errors.append(
                    f"{prefix} (drop_missing_columns): "
                    "missing 'columns' parameter."
                )
            elif not isinstance(cols, list):
                errors.append(
                    f"{prefix} (drop_missing_columns): "
                    "'columns' must be a list."
                )
            else:
                for c in cols:
                    if c not in df.columns:
                        errors.append(
                            f"{prefix} (drop_missing_columns): "
                            f"column '{c}' does not exist."
                        )

        # -- lowercase / uppercase may specify columns ----------------------
        if op_type in ("lowercase", "uppercase"):
            cols = op.get("columns")
            if cols is not None:
                if not isinstance(cols, list):
                    errors.append(
                        f"{prefix} ({op_type}): 'columns' must be a list."
                    )
                else:
                    for c in cols:
                        if c not in df.columns:
                            errors.append(
                                f"{prefix} ({op_type}): column '{c}' "
                                "does not exist."
                            )

    return errors


# ---------------------------------------------------------------------------
# Central Executor
# ---------------------------------------------------------------------------

def apply_cleaning_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute a structured cleaning plan on *df*.

    The plan is a dictionary that looks like::

        {
            "operations": [
                {"type": "remove_duplicates"},
                {"type": "fill_missing", "column": "salary", "method": "median"}
            ]
        }

    Args:
        df: Input DataFrame (will **not** be modified).
        plan: Structured cleaning plan.

    Returns:
        Tuple of (cleaned DataFrame, cleaning report dict).

    Raises:
        ValueError: If the plan fails validation.
    """
    # --- 1. Validate ---------------------------------------------------------
    validation_errors = validate_cleaning_plan(plan, df)
    if validation_errors:
        raise ValueError(
            "Cleaning plan validation failed:\n"
            + "\n".join(f"  • {e}" for e in validation_errors)
        )

    # --- 2. Snapshot "before" ------------------------------------------------
    cleaned = df.copy()
    rows_before = len(cleaned)
    cols_before = list(cleaned.columns)
    missing_before = int(cleaned.isnull().sum().sum())

    duplicates_removed = 0
    missing_filled = 0
    operations_applied: List[str] = []
    operations_details: List[str] = []
    columns_removed: List[str] = []
    columns_modified: List[str] = []

    # --- 3. Execute operations in order --------------------------------------
    for op in plan["operations"]:
        op_type = op["type"]
        operations_applied.append(op_type)

        if op_type == "remove_duplicates":
            cleaned, removed = remove_duplicates(cleaned)
            duplicates_removed += removed
            operations_details.append("Removed duplicate rows")

        elif op_type == "strip_spaces":
            cleaned = strip_spaces(cleaned)
            operations_details.append("Stripped leading/trailing spaces")

        elif op_type == "remove_extra_spaces":
            cleaned = remove_extra_spaces(cleaned)
            operations_details.append("Removed extra internal spaces")

        elif op_type == "lowercase":
            cols = op.get("columns")
            cleaned = to_lowercase(cleaned, columns=cols)
            if cols:
                columns_modified.extend(cols)
                operations_details.append(
                    f"Converted {', '.join(cols)} to lowercase"
                )
            else:
                operations_details.append("Converted text to lowercase")

        elif op_type == "uppercase":
            cols = op.get("columns")
            cleaned = to_uppercase(cleaned, columns=cols)
            if cols:
                columns_modified.extend(cols)
                operations_details.append(
                    f"Converted {', '.join(cols)} to UPPERCASE"
                )
            else:
                operations_details.append("Converted text to UPPERCASE")

        elif op_type == "drop_missing_rows":
            cols = op.get("columns")
            cleaned = drop_missing_rows(cleaned, columns=cols)
            if cols:
                operations_details.append(
                    f"Dropped rows with missing values in {', '.join(cols)}"
                )
            else:
                operations_details.append("Dropped rows with missing values")

        elif op_type == "drop_missing_columns":
            cols = op["columns"]
            cleaned = drop_missing_columns(cleaned, columns=cols)
            columns_removed.extend(cols)
            operations_details.append(
                f"Dropped column(s): {', '.join(cols)}"
            )

        elif op_type == "fill_missing":
            column = op["column"]
            method = op["method"]
            missing_in_col_before = int(cleaned[column].isnull().sum())

            if method == "mean":
                cleaned = fill_missing_mean(cleaned, column)
            elif method == "median":
                cleaned = fill_missing_median(cleaned, column)
            elif method == "mode":
                cleaned = fill_missing_mode(cleaned, column)
            elif method == "forward_fill":
                cleaned = fill_missing_forward(cleaned, column)
            elif method == "backward_fill":
                cleaned = fill_missing_backward(cleaned, column)

            missing_in_col_after = int(cleaned[column].isnull().sum())
            missing_filled += missing_in_col_before - missing_in_col_after

            if column not in columns_modified:
                columns_modified.append(column)

            operations_details.append(
                f"Filled missing {column} using {method}"
            )

    # --- 4. Snapshot "after" -------------------------------------------------
    rows_after = len(cleaned)
    cols_after = list(cleaned.columns)
    missing_after = int(cleaned.isnull().sum().sum())

    report: Dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "columns_before": cols_before,
        "columns_after": cols_after,
        "duplicates_removed": duplicates_removed,
        "missing_values_filled": missing_filled,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "columns_removed": columns_removed,
        "columns_modified": list(set(columns_modified)),
        "operations_applied": operations_applied,
        "operations_details": operations_details,
    }

    return cleaned, report


# ---------------------------------------------------------------------------
# Internal helpers (private)
# ---------------------------------------------------------------------------

def _validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> None:
    """Raise *ValueError* if any column is not in *df*."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) not found in DataFrame: {missing}"
        )
