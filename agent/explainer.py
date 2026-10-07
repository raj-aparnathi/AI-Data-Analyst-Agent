"""
agent/explainer.py — AI Result Explainer (M4)
================================================
Generates natural-language explanations of analysis results.

The explainer receives the actual Python/Pandas result and converts it
into a clear, concise natural-language answer.

It NEVER invents information — it only explains values present in the
analysis result.

Author : Member 4 (Analysis + Visualization)
"""

import json
from typing import Any, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

EXPLAIN_RESULT_PROMPT = """You are a data analyst explaining an analysis result to a user.

User question:
{user_question}

Analysis performed:
{analysis_plan}

Actual result:
{result}

Explain the result clearly and concisely.

Rules:
1. Use only information present in the actual result.
2. Never invent numbers.
3. Do not perform new calculations unless the provided result already contains the required values.
4. Do not claim causation from correlation.
5. Mention important values when useful.
6. Use simple language.
7. If the result is empty, clearly say that no matching data was found.
8. Keep the explanation under 100 words.
"""


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------

def _format_result_for_prompt(result: Any) -> str:
    """Convert an analysis result to a string suitable for the AI prompt."""
    if isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return "Empty result — no matching data found."
        # Limit to first 20 rows for prompt brevity
        display = result.head(20)
        return display.to_string(index=False)

    if isinstance(result, dict):
        # Handle trend results with embedded DataFrames
        formatted = {}
        for k, v in result.items():
            if isinstance(v, pd.DataFrame):
                formatted[k] = v.head(10).to_dict(orient="records")
            else:
                formatted[k] = v
        return json.dumps(formatted, indent=2, default=str)

    if isinstance(result, (list, tuple)):
        return json.dumps(result, indent=2, default=str)

    return str(result)


def _format_plan_for_prompt(plan: Dict[str, Any]) -> str:
    """Convert an analysis plan to a string for the AI prompt."""
    # Remove large data objects from the plan representation
    clean_plan = {
        k: v for k, v in plan.items()
        if not isinstance(v, pd.DataFrame)
    }
    return json.dumps(clean_plan, indent=2, default=str)


# ---------------------------------------------------------------------------
# Local (rule-based) explainer — works without an AI model
# ---------------------------------------------------------------------------

def explain_result_local(
    user_question: str,
    analysis_result: Dict[str, Any],
) -> str:
    """Generate a simple natural-language explanation without calling an AI model.

    Uses template-based logic to produce a clear, factual explanation
    based on the actual result values.

    Args:
        user_question: The user's original question.
        analysis_result: The result dictionary from apply_analysis_plan.

    Returns:
        A natural-language explanation string.
    """
    operation = analysis_result.get("operation", "unknown")
    result = analysis_result.get("result")
    summary = analysis_result.get("summary", {})
    result_type = analysis_result.get("result_type", "")

    # ----- Scalar results (mean, median, mode, min, max, count, etc.) -----
    if result_type == "scalar":
        column = summary.get("column", "the column")
        op_name = summary.get("operation", operation)

        if result is None:
            return f"No data was found to calculate the {op_name} of {column}."

        # Format numeric values nicely
        if isinstance(result, float):
            formatted = f"{result:,.2f}"
        else:
            formatted = str(result)

        op_labels = {
            "mean": "average (mean)",
            "median": "median",
            "mode": "most frequent value (mode)",
            "min": "minimum value",
            "max": "maximum value",
            "count": "count of valid values",
            "variance": "variance",
            "std": "standard deviation",
        }
        label = op_labels.get(op_name, op_name)
        return f"The {label} of **{column}** is **{formatted}**."

    # ----- Quartiles (dict result) ----------------------------------------
    if operation == "quartiles" and isinstance(result, dict):
        column = summary.get("column", "the column")
        q1 = result.get("Q1", "N/A")
        q2 = result.get("Q2", "N/A")
        q3 = result.get("Q3", "N/A")
        return (
            f"Quartiles for **{column}**:\n"
            f"- **Q1 (25th percentile):** {q1:,.2f}\n"
            f"- **Q2 (Median):** {q2:,.2f}\n"
            f"- **Q3 (75th percentile):** {q3:,.2f}"
        )

    # ----- DataFrame results (groupby, top_n, filter, sort) ---------------
    if result_type == "dataframe" and isinstance(result, pd.DataFrame):
        if len(result) == 0:
            return "No matching data was found for your query."

        if operation == "groupby":
            group_col = summary.get("group_column", result.columns[0])
            value_col = summary.get("value_column", result.columns[-1])
            agg = summary.get("aggregation", "")
            n_rows = len(result)

            # Describe the top result
            top_row = result.iloc[0]
            top_name = top_row[group_col]
            top_value = top_row[value_col]

            if isinstance(top_value, float):
                formatted_val = f"{top_value:,.2f}"
            else:
                formatted_val = str(top_value)

            if n_rows == 1:
                return (
                    f"**{top_name}** has the highest {agg} of {value_col} "
                    f"with **{formatted_val}**."
                )

            lines = [
                f"Here are the results grouped by **{group_col}** "
                f"({agg} of {value_col}):\n"
            ]
            for idx, row in result.head(10).iterrows():
                val = row[value_col]
                if isinstance(val, float):
                    val = f"{val:,.2f}"
                lines.append(f"- **{row[group_col]}**: {val}")

            if n_rows > 10:
                lines.append(f"\n... and {n_rows - 10} more.")

            return "\n".join(lines)

        if operation == "top_n":
            n = summary.get("n", len(result))
            column = summary.get("column", result.columns[-1])

            lines = [f"Top {n} by **{column}**:\n"]
            for idx, row in result.iterrows():
                val = row[column]
                if isinstance(val, float):
                    val = f"{val:,.2f}"
                # Include other columns for context
                other_info = []
                for col in result.columns:
                    if col != column:
                        other_info.append(str(row[col]))
                label = ", ".join(other_info) if other_info else f"Row {idx + 1}"
                lines.append(f"{idx + 1}. **{label}**: {val}")

            return "\n".join(lines)

        if operation == "filter":
            filter_desc = summary.get("filter", "the condition")
            n_rows = len(result)
            return (
                f"**{n_rows}** row(s) match the condition: {filter_desc}."
            )

        if operation == "sort":
            sort_col = summary.get("sort_column", "")
            ascending = summary.get("ascending", True)
            direction = "ascending" if ascending else "descending"
            return (
                f"Data sorted by **{sort_col}** in {direction} order. "
                f"Showing **{len(result)}** rows."
            )

        # Generic DataFrame result
        return (
            f"The analysis returned **{len(result)}** rows "
            f"and **{len(result.columns)}** columns."
        )

    # ----- Correlation matrix ---------------------------------------------
    if operation == "correlation" and result_type == "matrix":
        if isinstance(result, pd.DataFrame):
            variables = summary.get("variables", list(result.columns))

            # Find strongest correlations (excluding diagonal)
            correlations = []
            for i, col1 in enumerate(result.columns):
                for j, col2 in enumerate(result.columns):
                    if i < j:
                        corr_val = result.iloc[i, j]
                        if not pd.isna(corr_val):
                            correlations.append((col1, col2, corr_val))

            correlations.sort(key=lambda x: abs(x[2]), reverse=True)

            lines = ["Correlation analysis results:\n"]
            for col1, col2, corr in correlations[:5]:
                strength = _correlation_strength(corr)
                direction = "positive" if corr > 0 else "negative"
                lines.append(
                    f"- **{col1}** and **{col2}**: {corr:.3f} "
                    f"({strength} {direction} correlation)"
                )

            if not correlations:
                return "No correlations could be calculated."

            return "\n".join(lines)

    # ----- Trend analysis -------------------------------------------------
    if operation == "trend" and isinstance(result, dict):
        direction = result.get("direction", "unknown")
        start_val = result.get("start_value")
        end_val = result.get("end_value")
        change = result.get("change")
        change_pct = result.get("change_percent")

        if start_val is None or end_val is None:
            return "No trend data was available for analysis."

        lines = [f"The trend is **{direction}**.\n"]
        lines.append(f"- Start value: **{start_val:,.2f}**")
        lines.append(f"- End value: **{end_val:,.2f}**")

        if change is not None:
            lines.append(f"- Absolute change: **{change:,.2f}**")
        if change_pct is not None:
            lines.append(f"- Percentage change: **{change_pct:.1f}%**")

        return "\n".join(lines)

    # ----- Fallback -------------------------------------------------------
    return f"Analysis complete. Operation: {operation}."


def _correlation_strength(corr: float) -> str:
    """Classify correlation strength."""
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        return "strong"
    elif abs_corr >= 0.5:
        return "moderate"
    elif abs_corr >= 0.3:
        return "weak"
    else:
        return "very weak"


# ---------------------------------------------------------------------------
# AI-powered explainer (builds prompt for external model)
# ---------------------------------------------------------------------------

def build_explanation_prompt(
    user_question: str,
    analysis_result: Dict[str, Any],
) -> str:
    """Build a prompt for an AI model to generate an explanation.

    This is used when an AI model is available for richer explanations.

    Args:
        user_question: The user's original question.
        analysis_result: The result dictionary from apply_analysis_plan.

    Returns:
        A formatted prompt string ready to send to an AI model.
    """
    result = analysis_result.get("result")
    formatted_result = _format_result_for_prompt(result)

    # Build a clean plan representation
    plan_info = {
        "operation": analysis_result.get("operation"),
        "summary": analysis_result.get("summary"),
    }
    formatted_plan = _format_plan_for_prompt(plan_info)

    return EXPLAIN_RESULT_PROMPT.format(
        user_question=user_question,
        analysis_plan=formatted_plan,
        result=formatted_result,
    )
