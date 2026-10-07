"""
ui/chat.py — AI Data Analyst Chat Interface (M4)
==================================================
Provides the Streamlit chat UI where users can ask natural-language
questions about their dataset and receive analysis results, charts,
and AI-generated explanations.

Author : Member 4 (Analysis + Visualization)
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from agent.planner import generate_analysis_plan
from agent.explainer import explain_result_local
from operations.analysis import apply_analysis_plan, SUPPORTED_ANALYSIS_OPERATIONS
from operations.visualization import build_chart_from_result
from ui.charts import display_chart


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_chat_state() -> None:
    """Initialise chat-related session state keys."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def render_chat_section(df: Optional[pd.DataFrame] = None) -> None:
    """Render the AI Data Analyst chat interface.

    Args:
        df: The DataFrame to analyse.  If None, attempts to retrieve
            it from ``st.session_state["df"]``.
    """
    _init_chat_state()

    st.header("💬 AI Data Analyst")
    st.markdown(
        "Ask any question about your dataset — the AI will plan the "
        "analysis, run it with Python, show a chart, and explain the result."
    )

    # Resolve DataFrame
    if df is None:
        df = st.session_state.get("df")

    if df is None:
        st.info(
            "👆 Please upload a dataset first, then come here to ask questions."
        )
        return

    # Show column info for reference
    with st.expander("📋 Available columns", expanded=False):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            col_info.append(f"- **{col}** ({dtype})")
        st.markdown("\n".join(col_info))

    st.divider()

    # ----- Display chat history -------------------------------------------
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Re-display chart if stored
            if msg.get("chart") is not None:
                display_chart(msg["chart"], key=f"chart_{msg.get('idx', 0)}")

            # Re-display result table if stored
            if msg.get("result_df") is not None:
                with st.expander("📊 View data", expanded=False):
                    st.dataframe(msg["result_df"], use_container_width=True)

    # ----- Chat input -----------------------------------------------------
    user_question = st.chat_input("Ask your dataset anything...")

    if user_question:
        # Add user message to history
        msg_idx = len(st.session_state.chat_messages)
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_question,
            "idx": msg_idx,
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Process the question
        with st.chat_message("assistant"):
            _process_question(user_question, df, msg_idx + 1)


def _process_question(
    user_question: str,
    df: pd.DataFrame,
    msg_idx: int,
) -> None:
    """Process a user question: plan → execute → visualise → explain.

    Args:
        user_question: The natural-language question.
        df: The DataFrame to analyse.
        msg_idx: Message index for unique chart keys.
    """
    response_parts = []
    chart_fig = None
    result_df = None

    try:
        # ----- Step 1: Generate analysis plan ---------------------------------
        with st.spinner("🧠 Planning analysis..."):
            plan = generate_analysis_plan(user_question, df)

        # Check if we need an AI model (rule-based couldn't handle it)
        if plan.get("status") == "awaiting_ai_response":
            st.warning(
                "⚠️ I couldn't determine the analysis from your question. "
                "Please try rephrasing with specific column names.\n\n"
                f"**Available columns:** {', '.join(df.columns)}"
            )
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    "I couldn't determine the analysis from your question. "
                    "Please try rephrasing with specific column names."
                ),
                "idx": msg_idx,
            })
            return

        # Show the plan
        with st.expander("📋 Analysis Plan", expanded=False):
            st.json(plan)

        response_parts.append(f"**Operation:** `{plan.get('operation', 'unknown')}`\n")

        # ----- Step 2: Execute the plan ---------------------------------------
        with st.spinner("⚙️ Running analysis..."):
            analysis_result = apply_analysis_plan(df, plan)

        result = analysis_result.get("result")
        result_type = analysis_result.get("result_type", "")

        # Display result
        if result_type == "scalar":
            if isinstance(result, float):
                formatted = f"{result:,.2f}"
            else:
                formatted = str(result)
            response_parts.append(f"**Result:** {formatted}\n")

        elif result_type == "dict":
            if isinstance(result, dict):
                # Handle quartiles and trend results
                for k, v in result.items():
                    if not isinstance(v, pd.DataFrame):
                        if isinstance(v, float):
                            response_parts.append(f"- **{k}:** {v:,.2f}")
                        else:
                            response_parts.append(f"- **{k}:** {v}")
                response_parts.append("")

        elif result_type in ("dataframe", "matrix"):
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                result_df = result
                st.dataframe(result.head(20), use_container_width=True)
                if len(result) > 20:
                    st.caption(f"Showing 20 of {len(result)} rows.")

        # ----- Step 3: Build and display chart --------------------------------
        with st.spinner("📊 Generating chart..."):
            chart_fig = build_chart_from_result(analysis_result)

        if chart_fig is not None:
            display_chart(chart_fig, key=f"chart_{msg_idx}")

        # ----- Step 4: Generate explanation -----------------------------------
        with st.spinner("💡 Generating explanation..."):
            explanation = explain_result_local(user_question, analysis_result)

        st.markdown(f"\n{explanation}")
        response_parts.append(explanation)

    except ValueError as e:
        error_msg = f"❌ **Error:** {str(e)}"
        st.error(error_msg)
        response_parts.append(error_msg)

    except Exception as e:
        error_msg = f"❌ **Unexpected error:** {str(e)}"
        st.error(error_msg)
        response_parts.append(error_msg)

    # Store assistant message in history
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": "\n".join(response_parts),
        "chart": chart_fig,
        "result_df": result_df,
        "idx": msg_idx,
    })


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import streamlit.runtime

    if streamlit.runtime.exists():
        st.set_page_config(
            page_title="AI Data Analyst",
            page_icon="💬",
            layout="wide",
        )
        render_chat_section()
    else:
        print("=" * 60)
        print("ui/chat.py: Chat Module Checked Successfully")
        print("=" * 60)
        print("To launch the interactive UI, run:")
        print("    streamlit run ui/chat.py")
        print("=" * 60)
