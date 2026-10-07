"""
utils/helpers.py — Reusable Helper Functions (M1)
=================================================
Small, focused utility functions used across the M1 modules
(file_handler, dataset_profiler, upload UI).

Author : Member 1 (Dataset Understanding & File Handling)
"""

import pandas as pd


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def format_file_size(size_in_bytes: int) -> str:
    """
    Convert a raw byte count into a human-readable string.

    Examples
    --------
    >>> format_file_size(0)
    '0 B'
    >>> format_file_size(1024)
    '1.00 KB'
    >>> format_file_size(1_500_000)
    '1.43 MB'
    """
    if size_in_bytes < 0:
        return "0 B"

    # Define size thresholds and labels
    units = [
        (1 << 30, "GB"),
        (1 << 20, "MB"),
        (1 << 10, "KB"),
    ]

    for threshold, label in units:
        if size_in_bytes >= threshold:
            return f"{size_in_bytes / threshold:.2f} {label}"

    return f"{size_in_bytes} B"


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

def safe_column_name(name: str) -> str:
    """
    Sanitise a column name so it is safe to use as a Python identifier or
    in file paths.

    Rules applied:
      • Strip leading/trailing whitespace.
      • Replace spaces and hyphens with underscores.
      • Collapse consecutive underscores.
      • Convert to lowercase.

    Parameters
    ----------
    name : str
        The raw column name.

    Returns
    -------
    str
        A cleaned, lowercase, underscore-separated name.

    Examples
    --------
    >>> safe_column_name("  Customer  ID ")
    'customer_id'
    >>> safe_column_name("Total-Sales (USD)")
    'total_sales_(usd)'
    """
    import re

    cleaned = name.strip()
    cleaned = cleaned.replace(" ", "_").replace("-", "_")
    cleaned = re.sub(r"_+", "_", cleaned)  # collapse multiple underscores
    cleaned = cleaned.lower()
    return cleaned


# ---------------------------------------------------------------------------
# Missing-value helpers
# ---------------------------------------------------------------------------

def calculate_missing_percentage(missing_count: int, total_rows: int) -> float:
    """
    Calculate the percentage of missing values for a given count.

    Parameters
    ----------
    missing_count : int
        Number of missing entries.
    total_rows : int
        Total number of rows in the dataset.

    Returns
    -------
    float
        Missing percentage rounded to 2 decimal places, or 0.0 if total_rows
        is zero (avoids ZeroDivisionError).

    Examples
    --------
    >>> calculate_missing_percentage(25, 5000)
    0.5
    >>> calculate_missing_percentage(0, 100)
    0.0
    """
    if total_rows == 0:
        return 0.0
    return round((missing_count / total_rows) * 100, 2)


def calculate_memory_usage(df: pd.DataFrame) -> str:
    """
    Return the total memory consumption of a DataFrame in a readable format.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.

    Returns
    -------
    str
        e.g. "1.43 MB"
    """
    total_bytes = df.memory_usage(deep=True).sum()
    return format_file_size(total_bytes)


# ---------------------------------------------------------------------------
# Identifier-column heuristic
# ---------------------------------------------------------------------------

# Column names that strongly suggest the column is an identifier,
# not a meaningful numerical feature.
_ID_KEYWORDS = {
    "id", "code", "number", "num", "no", "key", "index",
    "serial", "sno", "s_no", "srno", "sr_no",
    "zip", "zipcode", "zip_code", "postal", "pincode", "pin",
    "phone", "telephone", "mobile", "fax",
    "ssn", "social_security",
}


def looks_like_id_column(col_name: str, series: pd.Series) -> bool:
    """
    Heuristic check to decide whether a numeric-looking column is really
    an **identifier** rather than a meaningful numerical feature.

    Heuristics (any one is enough to flag the column):
      1. The lowercased, sanitised column name *ends with* or *equals* a
         known ID keyword (e.g. "customer_id", "order_no").
      2. Every non-null value is unique (cardinality == row count), AND the
         column has integer dtype.

    Parameters
    ----------
    col_name : str
        The column name (raw, as it appears in the DataFrame).
    series : pd.Series
        The actual data.

    Returns
    -------
    bool
        True if the column is *likely* an identifier.

    Notes
    -----
    This is a **simple heuristic** and can produce false positives/negatives.
    It intentionally errs on the side of caution — if unsure, it returns
    False (i.e. treats the column as numerical).
    """
    clean = safe_column_name(col_name)

    # Check 1: name-based — does the clean name end with an ID keyword?
    for keyword in _ID_KEYWORDS:
        if clean == keyword or clean.endswith(f"_{keyword}"):
            return True

    # Check 2: uniqueness-based — all values unique + integer dtype
    if pd.api.types.is_integer_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0 and non_null.nunique() == len(non_null):
            return True

    return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing utils/helpers.py independently")
    print("=" * 60)

    # 1. format_file_size
    print("\n1. format_file_size:")
    for size in [512, 1024, 1_500_000, 2_000_000_000]:
        print(f"   {size} bytes -> {format_file_size(size)}")

    # 2. safe_column_name
    print("\n2. safe_column_name:")
    for raw in ["  Customer ID ", "Total-Sales (USD)", "First_Name", "Age"]:
        print(f"   '{raw}' -> '{safe_column_name(raw)}'")

    # 3. calculate_missing_percentage
    print("\n3. calculate_missing_percentage:")
    print(f"   25 / 5000 -> {calculate_missing_percentage(25, 5000)}%")
    print(f"   0 / 100   -> {calculate_missing_percentage(0, 100)}%")
    print(f"   5 / 0     -> {calculate_missing_percentage(5, 0)}%")

    # 4. looks_like_id_column
    print("\n4. looks_like_id_column:")
    s_id = pd.Series([101, 102, 103, 104])
    s_age = pd.Series([25, 30, 25, 40])
    print(f"   Customer_ID: {looks_like_id_column('Customer_ID', s_id)}")
    print(f"   Age:         {looks_like_id_column('Age', s_age)}")
    print(f"   Unique Ints: {looks_like_id_column('custom_col', s_id)}")

    print("\n[SUCCESS] utils/helpers.py passed standalone checks.")
