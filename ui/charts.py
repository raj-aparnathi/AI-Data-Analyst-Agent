"""
ui/charts.py — Streamlit Chart Rendering Layer (M4)
=====================================================
Renders Plotly charts within the Streamlit UI.

This module is responsible ONLY for UI rendering — no data-analysis
logic belongs here.

Author : Member 4 (Analysis + Visualization)
"""

from typing import Optional

import streamlit as st

try:
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def display_chart(
    chart: Optional["go.Figure"],
    use_container_width: bool = True,
    key: Optional[str] = None,
) -> None:
    """Render a Plotly chart in the Streamlit app.

    Args:
        chart: A Plotly Figure object to display.  If None, nothing
               is rendered.
        use_container_width: Whether to stretch the chart to fill the
                             container width.
        key: Optional unique key for the Streamlit element.
    """
    if chart is None:
        return

    if not _HAS_PLOTLY:
        st.warning(
            "📊 Plotly is not installed. "
            "Install it with: `pip install plotly`"
        )
        return

    st.plotly_chart(
        chart,
        use_container_width=use_container_width,
        key=key,
    )


def display_chart_with_header(
    chart: Optional["go.Figure"],
    title: Optional[str] = None,
    use_container_width: bool = True,
    key: Optional[str] = None,
) -> None:
    """Render a chart with an optional section header.

    Args:
        chart: A Plotly Figure object to display.
        title: Optional header text shown above the chart.
        use_container_width: Whether to stretch the chart.
        key: Optional unique key for the Streamlit element.
    """
    if chart is None:
        return

    if title:
        st.subheader(f"📊 {title}")

    display_chart(chart, use_container_width=use_container_width, key=key)
