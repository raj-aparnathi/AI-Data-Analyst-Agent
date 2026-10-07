"""
Cleaning Results UI Component (M2)

Streamlit component that displays a before/after cleaning summary,
the list of operations applied, and the cleaned DataFrame.

Usage (from the main app)::

    from ui.results import display_cleaning_results

    display_cleaning_results(original_df, cleaned_df, report)
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st


def display_cleaning_results(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    report: Dict[str, Any],
) -> None:
    """Render the cleaning results inside a Streamlit app.

    Args:
        original_df: The DataFrame before cleaning.
        cleaned_df: The DataFrame after cleaning.
        report: The cleaning report dictionary produced by
                :func:`operations.cleaning.apply_cleaning_plan`.
    """
    st.header("🧹 Cleaning Results")

    # ----- Before / After summary columns -----------------------------------
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("Before Cleaning")
        st.metric("Rows", report.get("rows_before", len(original_df)))
        st.metric(
            "Columns",
            len(report.get("columns_before", original_df.columns)),
        )
        st.metric(
            "Missing Values",
            report.get("missing_values_before", int(original_df.isnull().sum().sum())),
        )
        duplicates_before = (
            report.get("duplicates_removed", 0) + int(cleaned_df.duplicated().sum())
        )
        st.metric("Duplicates", duplicates_before)

    with col_after:
        st.subheader("After Cleaning")
        st.metric("Rows", report.get("rows_after", len(cleaned_df)))
        st.metric(
            "Columns",
            len(report.get("columns_after", cleaned_df.columns)),
        )
        st.metric("Missing Values", report.get("missing_values_after", 0))
        st.metric("Duplicates", int(cleaned_df.duplicated().sum()))

    st.divider()

    # ----- Operations Applied -----------------------------------------------
    op_details = report.get("operations_details")
    if op_details:
        st.subheader("✅ Operations Applied")
        for detail in op_details:
            st.markdown(f"✓ {detail}")
    else:
        operations = report.get("operations_applied", [])
        if operations:
            st.subheader("✅ Operations Applied")
            for op_name in operations:
                label = _operation_label(op_name)
                st.markdown(f"✓ {label}")

    st.divider()

    # ----- Changes summary --------------------------------------------------
    st.subheader("📊 Changes")

    changes_col1, changes_col2 = st.columns(2)

    with changes_col1:
        dup_removed = report.get("duplicates_removed", 0)
        if dup_removed > 0:
            st.info(f"**Duplicates removed:** {dup_removed}")

        missing_filled = report.get("missing_values_filled", 0)
        if missing_filled > 0:
            st.info(f"**Missing values filled:** {missing_filled}")

    with changes_col2:
        cols_removed = report.get("columns_removed", [])
        if cols_removed:
            st.info(f"**Columns removed:** {', '.join(cols_removed)}")

        cols_modified = report.get("columns_modified", [])
        if cols_modified:
            st.info(f"**Columns modified:** {', '.join(cols_modified)}")

    rows_diff = report.get("rows_before", 0) - report.get("rows_after", 0)
    if rows_diff > 0:
        st.info(f"**Rows removed (total):** {rows_diff}")

    st.divider()

    # ----- Cleaned DataFrame preview ----------------------------------------
    st.subheader("📋 Cleaned Data Preview")
    st.dataframe(cleaned_df, use_container_width=True)

    st.caption(
        f"Showing {len(cleaned_df)} rows × "
        f"{len(cleaned_df.columns)} columns."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OPERATION_LABELS = {
    "remove_duplicates": "Removed duplicate rows",
    "drop_missing_rows": "Dropped rows with missing values",
    "drop_missing_columns": "Dropped specified columns",
    "fill_missing": "Filled missing values",
    "strip_spaces": "Stripped leading/trailing spaces",
    "remove_extra_spaces": "Removed extra internal spaces",
    "lowercase": "Converted text to lowercase",
    "uppercase": "Converted text to UPPERCASE",
}


def _operation_label(op_name: str) -> str:
    """Return a human-friendly label for an operation name."""
    return _OPERATION_LABELS.get(op_name, op_name)
