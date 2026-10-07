"""
operations/transformation.py - Transformation Operations (M3)
==============================================================
Provides general-purpose dataset transformation functions:
    - convert_datatype          : Cast a column to a new dtype.
    - rename_column             : Rename a single column safely.
    - drop_column               : Drop one column explicitly.
    - create_calculated_column  : Derive a column via +, -, *, /.
    - extract_date_part         : Extract year/month/day/dayofweek.

Safety
------
- eval() and exec() are NEVER used.
- Calculated columns only allow +, -, *, / between validated columns.
- df.copy() is always used; the original DataFrame is never modified.
"""

import pandas as pd


# Whitelisted operators for calculated columns
SAFE_OPERATORS = {"+", "-", "*", "/"}

# Supported date parts
SUPPORTED_DATE_PARTS = {"year", "month", "day", "dayofweek"}


# ---------------------------------------------------------------------------
# Datatype Conversion
# ---------------------------------------------------------------------------

def convert_datatype(
    df: pd.DataFrame,
    column: str,
    dtype: str,
) -> pd.DataFrame:
    """Cast *column* to the requested *dtype*.

    Supported dtype strings: int, float, string, bool, datetime.

    Args:
        df: Input DataFrame (not modified).
        column: Column to convert.
        dtype: Target type string.

    Returns:
        DataFrame with the column cast to the requested type.

    Raises:
        ValueError: If column does not exist, dtype is unsupported,
                    or the conversion fails.
    """
    _validate_column_exists(df, column)
    df_out = df.copy()
    try:
        if dtype == "int":
            df_out[column] = pd.to_numeric(df_out[column], errors="raise").astype(int)
        elif dtype == "float":
            df_out[column] = pd.to_numeric(df_out[column], errors="raise").astype(float)
        elif dtype == "string":
            df_out[column] = df_out[column].astype(str)
        elif dtype == "bool":
            df_out[column] = df_out[column].astype(bool)
        elif dtype == "datetime":
            df_out[column] = pd.to_datetime(df_out[column], errors="coerce")
        else:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. "
                "Supported: int, float, string, bool, datetime."
            )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Failed to convert column '{column}' to '{dtype}': {exc}"
        ) from exc
    return df_out


# ---------------------------------------------------------------------------
# Rename Column
# ---------------------------------------------------------------------------

def rename_column(
    df: pd.DataFrame,
    old_name: str,
    new_name: str,
) -> pd.DataFrame:
    """Rename *old_name* to *new_name*.

    Args:
        df: Input DataFrame (not modified).
        old_name: Existing column name.
        new_name: Desired new column name.

    Returns:
        DataFrame with the column renamed.

    Raises:
        ValueError: If old_name does not exist, new_name is invalid,
                    or new_name already exists.
    """
    _validate_column_exists(df, old_name)
    if not new_name or not isinstance(new_name, str):
        raise ValueError("new_name must be a non-empty string.")
    if new_name in df.columns and new_name != old_name:
        raise ValueError(
            f"Column '{new_name}' already exists in the DataFrame."
        )
    df_out = df.copy()
    df_out = df_out.rename(columns={old_name: new_name})
    return df_out


# ---------------------------------------------------------------------------
# Drop Column
# ---------------------------------------------------------------------------

def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop a single column explicitly.

    Args:
        df: Input DataFrame (not modified).
        column: Column to drop.

    Returns:
        DataFrame without the specified column.

    Raises:
        ValueError: If column does not exist.
    """
    _validate_column_exists(df, column)
    df_out = df.copy()
    df_out = df_out.drop(columns=[column])
    return df_out


# ---------------------------------------------------------------------------
# Create Calculated Column
# ---------------------------------------------------------------------------

def create_calculated_column(
    df: pd.DataFrame,
    new_column: str,
    left_column: str,
    operator: str,
    right_column: str,
) -> pd.DataFrame:
    """Derive a new column from two existing numerical columns.

    SAFETY: Only +, -, *, / operators are allowed.
    eval() and exec() are NEVER used.

    Args:
        df: Input DataFrame (not modified).
        new_column: Name of the new derived column.
        left_column: Left operand column (must be numerical).
        operator: One of +, -, *, /.
        right_column: Right operand column (must be numerical).

    Returns:
        DataFrame with the new calculated column appended.

    Raises:
        ValueError: If columns are missing, not numerical, operator is
                    not whitelisted, or division by zero occurs.
    """
    _validate_column_exists(df, left_column)
    _validate_column_exists(df, right_column)

    if operator not in SAFE_OPERATORS:
        raise ValueError(
            f"Operator '{operator}' is not allowed. "
            f"Supported operators: {sorted(SAFE_OPERATORS)}"
        )

    if not pd.api.types.is_numeric_dtype(df[left_column]):
        raise ValueError(
            f"Left column '{left_column}' is not numerical."
        )
    if not pd.api.types.is_numeric_dtype(df[right_column]):
        raise ValueError(
            f"Right column '{right_column}' is not numerical."
        )

    df_out = df.copy()
    left = df_out[left_column]
    right = df_out[right_column]

    if operator == "+":
        result = left + right
    elif operator == "-":
        result = left - right
    elif operator == "*":
        result = left * right
    elif operator == "/":
        if (right == 0).any():
            import warnings
            warnings.warn(
                f"Division by zero detected in column '{right_column}'. "
                "Result will contain NaN / inf values.",
                RuntimeWarning,
                stacklevel=2,
            )
        result = left / right

    df_out[new_column] = result
    return df_out


# ---------------------------------------------------------------------------
# Date Extraction
# ---------------------------------------------------------------------------

def extract_date_part(
    df: pd.DataFrame,
    column: str,
    part: str,
) -> pd.DataFrame:
    """Extract a date component from a datetime column.

    Supported parts: year, month, day, dayofweek.

    A new column named ``<column>_<part>`` is added to the DataFrame.
    The original datetime column is preserved.

    Args:
        df: Input DataFrame (not modified).
        column: Datetime (or parseable string) column.
        part: Date component to extract.

    Returns:
        DataFrame with a new ``<column>_<part>`` column appended.

    Raises:
        ValueError: If column does not exist or part is unsupported.
    """
    _validate_column_exists(df, column)
    if part not in SUPPORTED_DATE_PARTS:
        raise ValueError(
            f"Unsupported date part '{part}'. "
            f"Supported parts: {sorted(SUPPORTED_DATE_PARTS)}"
        )

    df_out = df.copy()

    # Convert to datetime if not already; coerce invalid dates to NaT
    if not pd.api.types.is_datetime64_any_dtype(df_out[column]):
        df_out[column] = pd.to_datetime(df_out[column], errors="coerce")

    new_col = f"{column}_{part}"
    if part == "year":
        df_out[new_col] = df_out[column].dt.year
    elif part == "month":
        df_out[new_col] = df_out[column].dt.month
    elif part == "day":
        df_out[new_col] = df_out[column].dt.day
    elif part == "dayofweek":
        df_out[new_col] = df_out[column].dt.dayofweek

    return df_out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if column is not in df."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )


# ===========================================================================
# Transformation Executor (M3)
# ===========================================================================

from typing import Any, Dict, List, Tuple

SUPPORTED_TRANSFORMATIONS = {
    "label_encode",
    "one_hot_encode",
    "min_max_normalize",
    "z_score_normalize",
    "convert_datatype",
    "rename_column",
    "drop_column",
    "create_calculated_column",
    "extract_date_part",
}


def apply_transformation_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute a structured transformation plan on *df*.

    The plan should look like::

        {
            "operations": [
                {"operation": "one_hot_encode", "column": "gender"},
                {"operation": "min_max_normalize", "column": "salary"}
            ]
        }

    Args:
        df: Input DataFrame (not modified).
        plan: Structured transformation plan dict with an "operations" list.

    Returns:
        Tuple of (transformed DataFrame, transformation report dict).

    Raises:
        ValueError: If the plan fails validation.
    """
    # Lazy imports to avoid circular deps at module import time
    from operations.encoding import label_encode, one_hot_encode
    from operations.normalization import min_max_normalize, z_score_normalize

    # ---- 1. Validate plan structure ----------------------------------------
    errors = validate_transformation_plan(plan, df)
    if errors:
        raise ValueError(
            "Transformation plan validation failed:\n"
            + "\n".join(f"  * {e}" for e in errors)
        )

    # ---- 2. Work on a copy --------------------------------------------------
    transformed = df.copy()
    rows_before = len(transformed)

    operations_applied: List[str] = []
    columns_created: List[str] = []
    columns_removed: List[str] = []
    columns_modified: List[str] = []
    errors_encountered: List[str] = []

    # ---- 3. Execute in order ------------------------------------------------
    for op in plan["operations"]:
        op_name = op["operation"]
        try:
            cols_before = set(transformed.columns)

            if op_name == "label_encode":
                transformed, _ = label_encode(transformed, op["column"])
                if op["column"] not in columns_modified:
                    columns_modified.append(op["column"])

            elif op_name == "one_hot_encode":
                orig_col = op["column"]
                transformed = one_hot_encode(transformed, orig_col)
                cols_after = set(transformed.columns)
                new_cols = [c for c in cols_after if c not in cols_before]
                columns_created.extend(new_cols)
                if orig_col not in columns_removed:
                    columns_removed.append(orig_col)

            elif op_name == "min_max_normalize":
                transformed = min_max_normalize(transformed, op["column"])
                if op["column"] not in columns_modified:
                    columns_modified.append(op["column"])

            elif op_name == "z_score_normalize":
                transformed = z_score_normalize(transformed, op["column"])
                if op["column"] not in columns_modified:
                    columns_modified.append(op["column"])

            elif op_name == "convert_datatype":
                transformed = convert_datatype(
                    transformed, op["column"], op["dtype"]
                )
                if op["column"] not in columns_modified:
                    columns_modified.append(op["column"])

            elif op_name == "rename_column":
                old, new = op["old_name"], op["new_name"]
                transformed = rename_column(transformed, old, new)
                if old not in columns_removed:
                    columns_removed.append(old)
                if new not in columns_created:
                    columns_created.append(new)

            elif op_name == "drop_column":
                col = op["column"]
                transformed = drop_column(transformed, col)
                if col not in columns_removed:
                    columns_removed.append(col)

            elif op_name == "create_calculated_column":
                transformed = create_calculated_column(
                    transformed,
                    op["new_column"],
                    op["left_column"],
                    op["operator"],
                    op["right_column"],
                )
                if op["new_column"] not in columns_created:
                    columns_created.append(op["new_column"])

            elif op_name == "extract_date_part":
                transformed = extract_date_part(
                    transformed, op["column"], op["part"]
                )
                new_col = f"{op['column']}_{op['part']}"
                if new_col not in columns_created:
                    columns_created.append(new_col)

            operations_applied.append(op_name)

        except Exception as exc:  # noqa: BLE001
            errors_encountered.append(f"{op_name}: {exc}")

    # ---- 4. Build report ----------------------------------------------------
    report: Dict[str, Any] = {
        "operations_applied": operations_applied,
        "columns_created": columns_created,
        "columns_removed": columns_removed,
        "columns_modified": [
            c for c in columns_modified
            if c not in columns_removed and c in transformed.columns
        ],
        "rows_before": rows_before,
        "rows_after": len(transformed),
        "errors": errors_encountered,
    }

    return transformed, report


# ---------------------------------------------------------------------------
# Plan validation helper
# ---------------------------------------------------------------------------

def validate_transformation_plan(
    plan: Dict[str, Any],
    df: pd.DataFrame,
) -> List[str]:
    """Validate a transformation plan.

    Returns a list of error strings (empty list = valid).
    """
    errs: List[str] = []

    if not isinstance(plan, dict):
        errs.append("Plan must be a dictionary.")
        return errs

    operations = plan.get("operations")
    if operations is None:
        errs.append("Plan is missing the 'operations' key.")
        return errs
    if not isinstance(operations, list):
        errs.append("'operations' must be a list.")
        return errs

    for idx, op in enumerate(operations):
        prefix = f"Operation #{idx + 1}"
        if not isinstance(op, dict):
            errs.append(f"{prefix}: each operation must be a dict.")
            continue

        op_name = op.get("operation")
        if not op_name:
            errs.append(f"{prefix}: missing 'operation' key.")
            continue
        if op_name not in SUPPORTED_TRANSFORMATIONS:
            errs.append(
                f"{prefix}: unsupported operation '{op_name}'. "
                f"Supported: {sorted(SUPPORTED_TRANSFORMATIONS)}"
            )
            continue

        # Per-operation parameter checks
        if op_name in ("label_encode", "one_hot_encode",
                       "min_max_normalize", "z_score_normalize",
                       "drop_column"):
            col = op.get("column")
            if not col:
                errs.append(f"{prefix} ({op_name}): missing 'column'.")
            elif col not in df.columns:
                errs.append(
                    f"{prefix} ({op_name}): column '{col}' "
                    "does not exist in the DataFrame."
                )

        if op_name in ("min_max_normalize", "z_score_normalize"):
            col = op.get("column")
            if col and col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errs.append(
                        f"{prefix} ({op_name}): column '{col}' "
                        "is not numerical."
                    )

        if op_name == "convert_datatype":
            for key in ("column", "dtype"):
                if key not in op:
                    errs.append(f"{prefix} (convert_datatype): missing '{key}'.")
            col = op.get("column")
            if col and col not in df.columns:
                errs.append(
                    f"{prefix} (convert_datatype): column '{col}' "
                    "does not exist."
                )

        if op_name == "rename_column":
            for key in ("old_name", "new_name"):
                if key not in op:
                    errs.append(f"{prefix} (rename_column): missing '{key}'.")
            old = op.get("old_name")
            if old and old not in df.columns:
                errs.append(
                    f"{prefix} (rename_column): column '{old}' "
                    "does not exist."
                )

        if op_name == "create_calculated_column":
            for key in ("new_column", "left_column", "operator", "right_column"):
                if key not in op:
                    errs.append(
                        f"{prefix} (create_calculated_column): missing '{key}'."
                    )
            op_sym = op.get("operator")
            if op_sym and op_sym not in SAFE_OPERATORS:
                errs.append(
                    f"{prefix} (create_calculated_column): "
                    f"operator '{op_sym}' is not allowed."
                )
            for col_key in ("left_column", "right_column"):
                col = op.get(col_key)
                if col and col not in df.columns:
                    errs.append(
                        f"{prefix} (create_calculated_column): "
                        f"column '{col}' does not exist."
                    )

        if op_name == "extract_date_part":
            for key in ("column", "part"):
                if key not in op:
                    errs.append(f"{prefix} (extract_date_part): missing '{key}'.")
            col = op.get("column")
            if col and col not in df.columns:
                errs.append(
                    f"{prefix} (extract_date_part): column '{col}' "
                    "does not exist."
                )
            part = op.get("part")
            if part and part not in SUPPORTED_DATE_PARTS:
                errs.append(
                    f"{prefix} (extract_date_part): part '{part}' "
                    f"is not supported. Use: {sorted(SUPPORTED_DATE_PARTS)}"
                )

    return errs
