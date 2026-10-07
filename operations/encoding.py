"""
operations/encoding.py — Encoding Operations (M3)
==================================================
Provides categorical encoding functions:
    - label_encode   : Maps categories to integer codes.
    - one_hot_encode : Creates binary indicator columns.

Design principles
-----------------
* The original DataFrame is never modified (df.copy() is always used).
* Only the explicitly requested column is altered.
* Missing values (NaN) are handled consistently without silent corruption.
* Useful metadata / error messages are returned or raised.
"""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def label_encode(
    df: pd.DataFrame,
    column: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Encode a categorical column as integer labels.

    Each unique category is mapped to a non-negative integer.
    NaN values remain NaN (they are not assigned a numeric code).

    Args:
        df: Input DataFrame (not modified).
        column: Name of the column to encode.

    Returns:
        Tuple of:
            - Transformed DataFrame with the column encoded in-place.
            - Metadata dict containing the column name and category-to-int
              mapping (``{"column": ..., "mapping": {...}}``).

    Raises:
        ValueError: If *column* does not exist in *df*.
    """
    _validate_column_exists(df, column)

    df_out = df.copy()
    series = df_out[column]

    # Separate NaN mask so they survive encoding unchanged
    nan_mask = series.isna()

    # Fit encoder on non-null values only
    non_null_values = series.dropna().astype(str)
    encoder = LabelEncoder()
    encoder.fit(non_null_values)

    # Apply encoding; leave NaN positions as NaN
    encoded = series.copy().astype(object)
    encoded[~nan_mask] = encoder.transform(
        series[~nan_mask].astype(str)
    )
    encoded[nan_mask] = float("nan")

    # Attempt to use a numeric dtype, falling back to object if NaN present
    try:
        df_out[column] = pd.to_numeric(encoded)
    except (ValueError, TypeError):
        df_out[column] = encoded

    mapping = {
        label: int(code)
        for code, label in enumerate(encoder.classes_)
    }

    metadata: Dict[str, Any] = {
        "column": column,
        "mapping": mapping,
    }
    return df_out, metadata


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------

def one_hot_encode(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """One-hot encode a categorical column.

    The original *column* is dropped and replaced by binary indicator
    columns named ``<column>_<category>`` (e.g. ``gender_Male``).

    NaN values produce all-zero indicator rows (consistent behaviour).

    Args:
        df: Input DataFrame (not modified).
        column: Name of the column to encode.

    Returns:
        Transformed DataFrame with indicator columns appended and the
        original column removed.

    Raises:
        ValueError: If *column* does not exist in *df*.
    """
    _validate_column_exists(df, column)

    df_out = df.copy()

    dummies = pd.get_dummies(
        df_out[column],
        prefix=column,
        prefix_sep="_",
        dummy_na=False,        # NaN rows -> all-zero rows
        dtype=int,
    )

    # Insert new columns at the position of the original column, then drop it
    col_pos = df_out.columns.get_loc(column)
    df_out = df_out.drop(columns=[column])
    for i, dummy_col in enumerate(dummies.columns):
        df_out.insert(col_pos + i, dummy_col, dummies[dummy_col])

    return df_out


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_column_exists(df: pd.DataFrame, column: str) -> None:
    """Raise ValueError if *column* is not in *df*."""
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' does not exist in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
