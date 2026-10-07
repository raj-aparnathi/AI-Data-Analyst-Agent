"""
utils/dataset_profiler.py — Dataset Profiling (M1)
===================================================
Analyses a Pandas DataFrame and returns structured metadata about its
shape, column types, missing values, duplicates, and more.

The output is designed to be:
  • displayed directly in the Streamlit UI (upload.py),
  • passed to the AI agent as context (via agent/prompts.py).

Author : Member 1 (Dataset Understanding & File Handling)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from utils.helpers import (
    calculate_missing_percentage,
    calculate_memory_usage,
    looks_like_id_column,
)


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive profile of the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile.

    Returns
    -------
    dict
        A structured dictionary with the following keys:

        rows               : int        — Total number of rows.
        columns            : int        — Total number of columns.
        column_names       : list[str]  — Ordered list of column names.
        numerical_columns  : list[str]  — Columns classified as numerical
                                          features (excludes likely IDs).
        categorical_columns: list[str]  — Columns classified as categorical.
        datetime_columns   : list[str]  — Columns with datetime dtype.
        id_columns         : list[str]  — Columns detected as identifiers.
        missing_values     : dict       — {col_name: missing_count} for
                                          columns with at least one missing.
        missing_percentages: dict       — {col_name: pct} matching above.
        total_missing      : int        — Sum of all missing values.
        duplicate_rows     : int        — Number of fully duplicated rows.
        data_types         : dict       — {col_name: dtype_string}.
        unique_values      : dict       — {col_name: n_unique}.
        memory_usage       : str        — Human-readable memory footprint.
    """

    n_rows, n_cols = df.shape

    # ── Column classification ─────────────────────────────────────────────
    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    id_cols = []

    for col in df.columns:
        dtype = df[col].dtype

        # Datetime check first
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)

        # Numeric check (int / float) — but filter out likely identifiers
        elif pd.api.types.is_numeric_dtype(dtype):
            if looks_like_id_column(col, df[col]):
                id_cols.append(col)
            else:
                numerical_cols.append(col)

        # Everything else is treated as categorical
        else:
            categorical_cols.append(col)

    # ── Missing values ────────────────────────────────────────────────────
    missing_series = df.isnull().sum()
    # Only keep columns that actually have missing values
    missing_values = {
        col: int(count)
        for col, count in missing_series.items()
        if count > 0
    }

    missing_percentages = {
        col: calculate_missing_percentage(count, n_rows)
        for col, count in missing_values.items()
    }

    total_missing = int(missing_series.sum())

    # ── Duplicates ────────────────────────────────────────────────────────
    duplicate_rows = int(df.duplicated().sum())

    # ── Data types & unique counts ────────────────────────────────────────
    data_types = {col: str(df[col].dtype) for col in df.columns}
    unique_values = {col: int(df[col].nunique()) for col in df.columns}

    # ── Memory usage ──────────────────────────────────────────────────────
    memory = calculate_memory_usage(df)

    # ── Build and return the profile dictionary ───────────────────────────
    profile = {
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "id_columns": id_cols,
        "missing_values": missing_values,
        "missing_percentages": missing_percentages,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "data_types": data_types,
        "unique_values": unique_values,
        "memory_usage": memory,
    }

    return profile


# ---------------------------------------------------------------------------
# Human-readable description
# ---------------------------------------------------------------------------

def generate_dataset_description(df: pd.DataFrame) -> str:
    """
    Produce a concise, human-readable paragraph summarising the dataset.

    The text is dynamically generated from the actual data and is suitable
    for display in the UI or as context in an AI prompt.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    str
        Multi-line description string.

    Example output
    --------------
    This dataset contains 5,000 rows and 8 columns.
    It has 3 numerical columns, 3 categorical columns, and 0 datetime columns.
    There are 31 total missing values across 2 columns, and 14 duplicate rows.
    The dataset uses approximately 312.50 KB of memory.
    """

    profile = profile_dataset(df)

    lines = []

    # Shape
    lines.append(
        f"This dataset contains **{profile['rows']:,}** rows "
        f"and **{profile['columns']}** columns."
    )

    # Column types
    n_num = len(profile["numerical_columns"])
    n_cat = len(profile["categorical_columns"])
    n_dt  = len(profile["datetime_columns"])
    n_id  = len(profile["id_columns"])

    type_parts = []
    if n_num:
        type_parts.append(f"**{n_num}** numerical")
    if n_cat:
        type_parts.append(f"**{n_cat}** categorical")
    if n_dt:
        type_parts.append(f"**{n_dt}** datetime")
    if n_id:
        type_parts.append(f"**{n_id}** identifier")

    if type_parts:
        lines.append(
            "It has " + ", ".join(type_parts) + " column(s)."
        )

    # Missing values
    if profile["total_missing"] > 0:
        n_cols_with_missing = len(profile["missing_values"])
        lines.append(
            f"There are **{profile['total_missing']:,}** total missing values "
            f"across **{n_cols_with_missing}** column(s)."
        )
    else:
        lines.append("There are **no missing values**.")

    # Duplicates
    if profile["duplicate_rows"] > 0:
        lines.append(f"There are **{profile['duplicate_rows']:,}** duplicate rows.")
    else:
        lines.append("There are **no duplicate rows**.")

    # Memory
    lines.append(
        f"The dataset uses approximately **{profile['memory_usage']}** of memory."
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/dataset_profiler.py independently")
    print("=" * 60)

    # Create a realistic test DataFrame
    test_df = pd.DataFrame({
        "Customer_ID": [1001, 1002, 1003, 1004, 1005],
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "Age": [29, 34, None, 41, 36],
        "Salary": [55000.0, 62000.0, 71000.0, None, 68000.0],
        "City": ["New York", "Chicago", "New York", "Austin", "Chicago"],
        "Join_Date": pd.to_datetime(["2021-01-15", "2020-03-22", "2019-07-10", "2022-11-05", "2021-08-30"])
    })

    print(f"\nSample DataFrame:\n{test_df}\n")

    profile = profile_dataset(test_df)
    print("Dataset Profile Output:")
    for k, v in profile.items():
        print(f"  {k}: {v}")

    print("\nGenerated Human-Readable Description:")
    print(generate_dataset_description(test_df))

    print("\n[SUCCESS] utils/dataset_profiler.py passed standalone checks.")
