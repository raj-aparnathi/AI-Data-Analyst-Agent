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


# ===========================================================================
# MEMBER 3 CONTRIBUTION — TRANSFORMATION RESULTS UI
# ===========================================================================

def display_transformation_results(
    original_df,
    transformed_df,
    report: dict,
) -> None:
    """Render the transformation results inside a Streamlit app.

    M2's ``display_cleaning_results`` function is NOT modified.
    This function adds a separate transformation-results section.

    Args:
        original_df: The DataFrame before transformation.
        transformed_df: The DataFrame after transformation.
        report: The transformation report produced by
                :func:`operations.transformation.apply_transformation_plan`.
    """
    st.header("🔄 Transformation Results")

    # ----- Before / After summary -------------------------------------------
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("Before Transformation")
        st.metric("Rows", report.get("rows_before", len(original_df)))
        st.metric("Columns", len(original_df.columns))

    with col_after:
        st.subheader("After Transformation")
        st.metric("Rows", report.get("rows_after", len(transformed_df)))
        st.metric("Columns", len(transformed_df.columns))

    st.divider()

    # ----- Operations Applied -----------------------------------------------
    ops_applied = report.get("operations_applied", [])
    if ops_applied:
        st.subheader("✅ Operations Applied")
        for op_name in ops_applied:
            label = _transformation_label(op_name)
            st.markdown(f"✓ {label}")
    else:
        st.info("No transformation operations were applied.")

    st.divider()

    # ----- Changes summary --------------------------------------------------
    st.subheader("📊 Changes")

    changes_col1, changes_col2 = st.columns(2)

    with changes_col1:
        cols_created = report.get("columns_created", [])
        if cols_created:
            st.info(
                "**Columns created:**\n"
                + "\n".join(f"  - {c}" for c in cols_created)
            )

        cols_modified = report.get("columns_modified", [])
        if cols_modified:
            st.info(
                "**Columns modified:**\n"
                + "\n".join(f"  - {c}" for c in cols_modified)
            )

    with changes_col2:
        cols_removed = report.get("columns_removed", [])
        if cols_removed:
            st.info(
                "**Columns removed:**\n"
                + "\n".join(f"  - {c}" for c in cols_removed)
            )

        errors = report.get("errors", [])
        if errors:
            st.warning(
                "**Errors encountered:**\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    rows_diff = report.get("rows_before", 0) - report.get("rows_after", 0)
    if rows_diff != 0:
        st.info(f"**Row count change:** {rows_diff:+d}")

    st.divider()

    # ----- Transformed DataFrame preview ------------------------------------
    st.subheader("📋 Transformed Data Preview")
    st.dataframe(transformed_df, use_container_width=True)
    st.caption(
        f"Showing {len(transformed_df)} rows × "
        f"{len(transformed_df.columns)} columns."
    )


# ---------------------------------------------------------------------------
# Transformation label helper (M3)
# ---------------------------------------------------------------------------

_TRANSFORMATION_LABELS = {
    "label_encode": "Label encoded column",
    "one_hot_encode": "One-Hot encoded column",
    "min_max_normalize": "Min-Max normalized column",
    "z_score_normalize": "Z-Score standardized column",
    "convert_datatype": "Converted column datatype",
    "rename_column": "Renamed column",
    "drop_column": "Dropped column",
    "create_calculated_column": "Created calculated column",
    "extract_date_part": "Extracted date part",
}


def _transformation_label(op_name: str) -> str:
    """Return a human-friendly label for a transformation operation name."""
    return _TRANSFORMATION_LABELS.get(op_name, op_name)
