"""
ui/upload.py — Streamlit Upload & Dataset Overview UI (M1)
==========================================================
Renders the file-upload widget, loads the dataset via file_handler,
profiles it via dataset_profiler, and displays a rich overview in
the Streamlit sidebar and main area.

This module exposes a single entry-point:

    render_upload_section()  → pd.DataFrame or None

Other team members can call this function from app.py to get the
loaded DataFrame and continue with their own processing.

Author : Member 1 (Dataset Understanding & File Handling)
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd

from utils.file_handler import load_dataset
from utils.dataset_profiler import profile_dataset, generate_dataset_description
from utils.helpers import format_file_size


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def render_upload_section() -> pd.DataFrame | None:
    """
    Render the full upload + overview UI and return the loaded DataFrame.

    Returns
    -------
    pd.DataFrame or None
        The loaded dataset, or None if nothing has been uploaded yet
        (or if loading failed).
    """

    st.header("📂 Upload Your Dataset")
    st.markdown(
        "Upload a dataset to get started. "
        "Supported formats: **CSV**, **Excel** (.xlsx / .xls), **JSON**, **TXT**."
    )

    # ── File uploader widget ──────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls", "json", "txt"],
        help="Maximum file size is determined by your Streamlit server settings.",
    )

    if uploaded_file is None:
        st.info("👆 Upload a file to begin.")
        return None

    # ── Show basic file metadata ──────────────────────────────────────────
    file_size = format_file_size(uploaded_file.size)
    st.caption(f"📄 **{uploaded_file.name}**  •  {file_size}")

    # ── Load the dataset ──────────────────────────────────────────────────
    df = load_dataset(uploaded_file)

    if df is None:
        return None

    st.success(f"✅ Dataset loaded successfully — **{df.shape[0]:,}** rows, **{df.shape[1]}** columns.")

    # ── Store in session state so other modules can access ────────────────
    st.session_state["df"] = df

    # ── Profile the dataset ───────────────────────────────────────────────
    profile = profile_dataset(df)
    st.session_state["profile"] = profile

    # ── Render all overview sections ──────────────────────────────────────
    _show_dataset_preview(df)
    _show_dataset_overview(profile)
    _show_column_info(df, profile)
    _show_column_categories(profile)
    _show_data_quality(df, profile)
    _show_dataset_description(df)

    return df


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------

def _show_dataset_preview(df: pd.DataFrame) -> None:
    """Display the first rows of the dataset."""

    st.subheader("🔍 Dataset Preview")

    # Let the user choose how many rows to preview
    n_rows = st.slider(
        "Rows to preview",
        min_value=5,
        max_value=min(50, len(df)),
        value=min(10, len(df)),
        key="preview_slider",
    )

    st.dataframe(df.head(n_rows), use_container_width=True)


def _show_dataset_overview(profile: dict) -> None:
    """Display high-level metrics in a 4-column layout."""

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{profile['rows']:,}")
    col2.metric("Columns", profile["columns"])
    col3.metric("Missing Values", f"{profile['total_missing']:,}")
    col4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")

    # Extra detail row
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Numerical", len(profile["numerical_columns"]))
    col6.metric("Categorical", len(profile["categorical_columns"]))
    col7.metric("Datetime", len(profile["datetime_columns"]))
    col8.metric("Memory", profile["memory_usage"])


def _show_column_info(df: pd.DataFrame, profile: dict) -> None:
    """Display a table of column metadata: type, missing count, unique count."""

    st.subheader("📋 Column Information")

    # Build a summary DataFrame
    info_rows = []
    for col in profile["column_names"]:
        missing = profile["missing_values"].get(col, 0)
        pct = profile["missing_percentages"].get(col, 0.0)
        info_rows.append({
            "Column": col,
            "Data Type": profile["data_types"][col],
            "Missing": missing,
            "Missing %": f"{pct}%",
            "Unique": profile["unique_values"][col],
        })

    info_df = pd.DataFrame(info_rows)
    st.dataframe(info_df, use_container_width=True, hide_index=True)


def _show_column_categories(profile: dict) -> None:
    """Show the detected column classifications side by side."""

    st.subheader("🏷️ Column Classification")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Numerical Columns**")
        if profile["numerical_columns"]:
            for c in profile["numerical_columns"]:
                st.markdown(f"- `{c}`")
        else:
            st.caption("None detected.")

    with col_right:
        st.markdown("**Categorical Columns**")
        if profile["categorical_columns"]:
            for c in profile["categorical_columns"]:
                st.markdown(f"- `{c}`")
        else:
            st.caption("None detected.")

    # Extra sections for datetime and identifier columns (if any)
    if profile["datetime_columns"] or profile["id_columns"]:
        col_left2, col_right2 = st.columns(2)

        with col_left2:
            if profile["datetime_columns"]:
                st.markdown("**Datetime Columns**")
                for c in profile["datetime_columns"]:
                    st.markdown(f"- `{c}`")

        with col_right2:
            if profile["id_columns"]:
                st.markdown("**Identifier Columns** _(excluded from numerical)_")
                for c in profile["id_columns"]:
                    st.markdown(f"- `{c}`")


def _show_data_quality(df: pd.DataFrame, profile: dict) -> None:
    """Show data quality details: missing values breakdown, duplicates."""

    st.subheader("🩺 Data Quality")

    # ── Missing values breakdown ──────────────────────────────────────────
    st.markdown("**Missing Values**")

    if profile["missing_values"]:
        missing_df = pd.DataFrame([
            {
                "Column": col,
                "Missing Count": count,
                "Missing %": f"{profile['missing_percentages'][col]}%",
            }
            for col, count in profile["missing_values"].items()
        ])
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("No missing values found! 🎉")

    # ── Duplicates ────────────────────────────────────────────────────────
    st.markdown("**Duplicate Rows**")

    if profile["duplicate_rows"] > 0:
        st.warning(f"⚠️ {profile['duplicate_rows']:,} duplicate rows detected.")
    else:
        st.success("No duplicate rows found! 🎉")

    # ── Data types distribution ───────────────────────────────────────────
    st.markdown("**Data Types**")

    dtype_counts = {}
    for dtype_str in profile["data_types"].values():
        dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1

    dtype_df = pd.DataFrame([
        {"Data Type": dtype, "Count": count}
        for dtype, count in sorted(dtype_counts.items())
    ])
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)


def _show_dataset_description(df: pd.DataFrame) -> None:
    """Show the auto-generated natural-language description."""

    st.subheader("📝 Dataset Description")
    description = generate_dataset_description(df)
    st.markdown(description)


if __name__ == "__main__":
    import streamlit.runtime

    if streamlit.runtime.exists():
        # Running inside Streamlit server: e.g. streamlit run ui/upload.py
        st.set_page_config(page_title="Dataset Upload & Profiler", layout="wide")
        render_upload_section()
    else:
        # Running via direct python command: python ui/upload.py
        print("=" * 60)
        print("ui/upload.py: Streamlit Upload Module Checked Successfully")
        print("=" * 60)
        print("To launch the interactive UI in your browser, run:")
        print("    streamlit run ui/upload.py")
        print("=" * 60)
