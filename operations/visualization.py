"""
operations/visualization.py — Chart Generation (M4)
=====================================================
Creates interactive Plotly charts for analysis results.

Each function receives data and column names and returns a Plotly
figure object.  Chart creation is separated from UI rendering.

Author : Member 4 (Analysis + Visualization)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Supported chart types whitelist
# ---------------------------------------------------------------------------

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "histogram",
    "scatter",
    "pie",
    "box",
    "heatmap",
}


# ---------------------------------------------------------------------------
# Chart creation functions
# ---------------------------------------------------------------------------

def create_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
) -> go.Figure:
    """Create an interactive bar chart.

    Args:
        df: Data to plot.
        x_column: Column for the x-axis (categories).
        y_column: Column for the y-axis (values).
        title: Optional chart title.
        color: Optional column for colour encoding.

    Returns:
        A Plotly Figure object.
    """
    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        title=title or f"{y_column} by {x_column}",
        color=color,
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
    )
    return fig


def create_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Create an interactive line chart.

    Args:
        df: Data to plot.
        x_column: Column for the x-axis (typically dates).
        y_column: Column for the y-axis (values).
        title: Optional chart title.

    Returns:
        A Plotly Figure object.
    """
    fig = px.line(
        df,
        x=x_column,
        y=y_column,
        title=title or f"{y_column} over {x_column}",
        markers=True,
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
    )
    return fig


def create_histogram(
    df: pd.DataFrame,
    column: str,
    bins: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a histogram for a numeric column.

    Args:
        df: Data to plot.
        column: Column to create the histogram for.
        bins: Optional number of bins.
        title: Optional chart title.

    Returns:
        A Plotly Figure object.
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=title or f"Distribution of {column}",
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
    )
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: Optional[str] = None,
    color: Optional[str] = None,
    trendline: Optional[str] = None,
) -> go.Figure:
    """Create an interactive scatter plot.

    Args:
        df: Data to plot.
        x_column: Column for the x-axis.
        y_column: Column for the y-axis.
        title: Optional chart title.
        color: Optional column for colour encoding.
        trendline: Optional trendline type (e.g. "ols").

    Returns:
        A Plotly Figure object.
    """
    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        title=title or f"{y_column} vs {x_column}",
        color=color,
        trendline=trendline,
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white",
    )
    return fig


def create_pie_chart(
    df: pd.DataFrame,
    names_column: str,
    values_column: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a pie chart.

    Args:
        df: Data to plot.
        names_column: Column for slice labels.
        values_column: Column for slice sizes.
        title: Optional chart title.

    Returns:
        A Plotly Figure object.
    """
    fig = px.pie(
        df,
        names=names_column,
        values=values_column,
        title=title or f"{values_column} by {names_column}",
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_box_plot(
    df: pd.DataFrame,
    column: str,
    group_column: Optional[str] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a box plot.

    Args:
        df: Data to plot.
        column: Numeric column to visualise.
        group_column: Optional column to group boxes by.
        title: Optional chart title.

    Returns:
        A Plotly Figure object.
    """
    fig = px.box(
        df,
        x=group_column,
        y=column,
        title=title or f"Box Plot of {column}",
    )
    fig.update_layout(
        yaxis_title=column,
        template="plotly_white",
    )
    if group_column:
        fig.update_layout(xaxis_title=group_column)
    return fig


def create_heatmap(
    data: pd.DataFrame,
    title: Optional[str] = None,
) -> go.Figure:
    """Create a heatmap (typically for correlation matrices).

    Args:
        data: A square DataFrame (e.g. correlation matrix).
        title: Optional chart title.

    Returns:
        A Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=data.values,
            x=list(data.columns),
            y=list(data.index),
            colorscale="RdBu_r",
            zmid=0,
            text=np.round(data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12},
        )
    )
    fig.update_layout(
        title=title or "Correlation Heatmap",
        template="plotly_white",
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Automatic chart selection
# ---------------------------------------------------------------------------

def recommend_chart(
    analysis_result: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """Recommend an appropriate chart type based on the analysis result.

    Args:
        analysis_result: The result dict from apply_analysis_plan.
        df: The original DataFrame (for context).

    Returns:
        A chart type string or None if no chart is appropriate.
    """
    operation = analysis_result.get("operation", "")
    result = analysis_result.get("result")
    result_type = analysis_result.get("result_type", "")

    # Explicit visualization in the plan takes priority
    explicit = analysis_result.get("visualization")
    if explicit and explicit in SUPPORTED_CHART_TYPES:
        return explicit

    # Scalar results generally don't need charts
    if result_type == "scalar":
        return None

    # Correlation → heatmap
    if operation == "correlation":
        return "heatmap"

    # Trend → line
    if operation == "trend":
        return "line"

    # GroupBy / Top-N → bar
    if operation in ("groupby", "top_n"):
        return "bar"

    # DataFrame results with exactly 2 columns
    if result_type == "dataframe" and isinstance(result, pd.DataFrame):
        if result.shape[1] == 2:
            cols = result.columns
            col1_numeric = pd.api.types.is_numeric_dtype(result[cols[0]])
            col2_numeric = pd.api.types.is_numeric_dtype(result[cols[1]])

            if col1_numeric and col2_numeric:
                return "scatter"
            elif (col1_numeric and not col2_numeric) or (
                not col1_numeric and col2_numeric
            ):
                return "bar"

    return None


# ---------------------------------------------------------------------------
# Build chart from analysis result
# ---------------------------------------------------------------------------

def build_chart_from_result(
    analysis_result: Dict[str, Any],
    chart_type: Optional[str] = None,
) -> Optional[go.Figure]:
    """Build a Plotly chart from an analysis result dictionary.

    Args:
        analysis_result: The result dict from apply_analysis_plan.
        chart_type: Override chart type.  If None, uses the recommended
                    type from the result or auto-detection.

    Returns:
        A Plotly Figure, or None if no chart is appropriate.
    """
    if chart_type is None:
        chart_type = analysis_result.get("visualization")
    if chart_type is None:
        chart_type = recommend_chart(analysis_result)
    if chart_type is None or chart_type not in SUPPORTED_CHART_TYPES:
        return None

    operation = analysis_result.get("operation", "")
    result = analysis_result.get("result")
    summary = analysis_result.get("summary", {})
    result_type = analysis_result.get("result_type", "")

    try:
        # ----- Heatmap (correlation matrix) -----------------------------------
        if chart_type == "heatmap":
            if isinstance(result, pd.DataFrame):
                return create_heatmap(result, title="Correlation Matrix")
            return None

        # ----- Trend line chart -----------------------------------------------
        if operation == "trend" and isinstance(result, dict):
            trend_data = result.get("trend_data")
            if isinstance(trend_data, pd.DataFrame) and len(trend_data.columns) >= 2:
                return create_line_chart(
                    trend_data,
                    x_column=trend_data.columns[0],
                    y_column=trend_data.columns[1],
                    title=f"Trend: {trend_data.columns[1]} over time",
                )
            return None

        # ----- DataFrame results (groupby, top_n, filter, sort) ---------------
        if isinstance(result, pd.DataFrame) and len(result) > 0:
            # Determine x and y columns
            if operation in ("groupby", "top_n"):
                x_col = summary.get("group_column") or (
                    result.columns[0] if len(result.columns) >= 2 else None
                )
                y_col = summary.get("value_column") or summary.get("column") or (
                    result.columns[1] if len(result.columns) >= 2 else None
                )
            else:
                x_col = result.columns[0] if len(result.columns) >= 1 else None
                y_col = result.columns[1] if len(result.columns) >= 2 else None

            if x_col is None:
                return None

            if chart_type == "bar" and y_col:
                return create_bar_chart(result, x_column=x_col, y_column=y_col)
            elif chart_type == "line" and y_col:
                return create_line_chart(result, x_column=x_col, y_column=y_col)
            elif chart_type == "scatter" and y_col:
                return create_scatter_plot(result, x_column=x_col, y_column=y_col)
            elif chart_type == "pie" and y_col:
                return create_pie_chart(
                    result, names_column=x_col, values_column=y_col
                )
            elif chart_type == "histogram":
                col = y_col or x_col
                return create_histogram(result, column=col)
            elif chart_type == "box":
                col = y_col or x_col
                return create_box_plot(result, column=col, group_column=x_col if y_col else None)

    except Exception:
        # If chart building fails, return None gracefully
        return None

    return None
