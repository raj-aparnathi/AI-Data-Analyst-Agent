"""
AI Planner Module

Contributions:
- M2: Cleaning-plan logic (build, parse, validate cleaning plans)
- M4: Analysis-plan logic (convert questions to structured analysis plans)

Provides helpers that:
1. Build the prompt for the AI model.
2. Parse and validate the raw AI response into a cleaning/analysis plan.
3. Identify common natural-language instructions directly
   via rule-based parsing.
4. Offer high-level plan generation functions.

This module does NOT execute any Pandas operations — it only produces
the structured JSON plans.
"""

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from agent.prompts import CLEANING_PLAN_PROMPT
from operations.cleaning import (
    SUPPORTED_FILL_METHODS,
    SUPPORTED_OPERATIONS,
    validate_cleaning_plan,
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_cleaning_prompt(
    user_request: str,
    df: pd.DataFrame,
) -> str:
    """Build the full prompt string for the AI model.

    Creates a concise dataset profile from *df* and injects it, along
    with the *user_request*, into the ``CLEANING_PLAN_PROMPT`` template.

    Args:
        user_request: Natural-language cleaning instruction from the user.
        df: The DataFrame the user wants cleaned.

    Returns:
        Ready-to-send prompt string.
    """
    profile = _build_dataset_profile(df)
    return CLEANING_PLAN_PROMPT.format(
        user_request=user_request,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_cleaning_response(raw_response: str) -> Dict[str, Any]:
    """Extract a JSON cleaning plan from the raw AI response.

    Handles common issues such as markdown code fences wrapping the
    JSON and stray text before/after the JSON block.

    Args:
        raw_response: Raw text returned by the AI model.

    Returns:
        Parsed cleaning plan dictionary.

    Raises:
        ValueError: If the response does not contain valid JSON or the
                    JSON does not have the expected structure.
    """
    cleaned = raw_response.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try to locate a JSON object if there's surrounding text
    if not cleaned.startswith("{"):
        brace_start = cleaned.find("{")
        if brace_start == -1:
            raise ValueError(
                "AI response does not contain a JSON object."
            )
        cleaned = cleaned[brace_start:]

    # Find the matching closing brace
    brace_end = cleaned.rfind("}")
    if brace_end == -1:
        raise ValueError(
            "AI response contains an incomplete JSON object."
        )
    cleaned = cleaned[: brace_end + 1]

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AI response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(plan, dict):
        raise ValueError("Parsed JSON is not a dictionary.")

    if "operations" not in plan:
        raise ValueError(
            "Parsed JSON does not contain an 'operations' key."
        )

    return plan


# ---------------------------------------------------------------------------
# Rule-based Natural Language Request Parser
# ---------------------------------------------------------------------------

def parse_request_rule_based(
    user_request: str,
    df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Parse common natural-language cleaning instructions directly.

    Identifies requests such as:
    - "Remove duplicate rows" / "Remove duplicates"
    - "Fill missing salary with median"
    - "Replace missing city values with mode"
    - "Remove rows with missing values"
    - "Drop the city column"
    - "Convert city names to lowercase"
    - "Remove extra spaces from names"
    - "Remove duplicates and fill missing salary with median"

    Args:
        user_request: Natural-language request text.
        df: Input DataFrame to match column names against.

    Returns:
        Structured plan dictionary with 'operations' list, or None if no
        recognized operations were identified.
    """
    operations: List[Dict[str, Any]] = []

    # Split into sub-clauses by conjunctions/punctuation
    clauses = re.split(
        r"\s*(?:;|\band\b|\bthen\b|,|\n)\s*",
        user_request,
        flags=re.IGNORECASE,
    )
    clauses = [c.strip() for c in clauses if c.strip()]

    cols_lower_map = {col.lower(): col for col in df.columns}

    for clause in clauses:
        c_lower = clause.lower()

        # 1. remove_duplicates
        if (
            re.search(
                r"\b(remove|drop|delete)?\s*(duplicate|duplicates|duplicate rows)\b|\bdeduplicate\b",
                c_lower,
            )
            and "column" not in c_lower
        ):
            operations.append({"type": "remove_duplicates"})
            continue

        # 2. fill_missing / replace missing / impute missing
        if any(w in c_lower for w in ["fill", "replace", "impute"]):
            method = None
            if "median" in c_lower:
                method = "median"
            elif "mean" in c_lower or "average" in c_lower:
                method = "mean"
            elif (
                "mode" in c_lower
                or "most frequent" in c_lower
                or "most common" in c_lower
            ):
                method = "mode"
            elif (
                "forward fill" in c_lower
                or "forward_fill" in c_lower
                or "ffill" in c_lower
            ):
                method = "forward_fill"
            elif (
                "backward fill" in c_lower
                or "backward_fill" in c_lower
                or "backfill" in c_lower
                or "bfill" in c_lower
            ):
                method = "backward_fill"

            # Match column name
            matched_col = None
            for col_l, original_col in cols_lower_map.items():
                if re.search(rf"\b{re.escape(col_l)}\b", c_lower):
                    matched_col = original_col
                    break

            if not matched_col:
                col_match = re.search(
                    r"(?:missing|column|in|of)\s+([a-zA-Z0-9_]+)",
                    c_lower,
                )
                if col_match:
                    candidate = col_match.group(1)
                    if candidate in cols_lower_map:
                        matched_col = cols_lower_map[candidate]
                    else:
                        matched_col = candidate

            if matched_col and method:
                operations.append({
                    "type": "fill_missing",
                    "column": matched_col,
                    "method": method,
                })
                continue

        # 3. drop_missing_rows
        if (
            re.search(
                r"\b(remove|drop|delete)\s+(rows?\s+with\s+(missing|null|nan)|missing\s+rows)\b",
                c_lower,
            )
            or (
                "missing" in c_lower
                and any(
                    w in c_lower
                    for w in ["remove rows", "drop rows", "delete rows"]
                )
            )
        ):
            target_cols = [
                original_col
                for col_l, original_col in cols_lower_map.items()
                if re.search(rf"\b{re.escape(col_l)}\b", c_lower)
            ]
            op: Dict[str, Any] = {"type": "drop_missing_rows"}
            if target_cols:
                op["columns"] = target_cols
            operations.append(op)
            continue

        # 4. drop_missing_columns
        if re.search(
            r"\b(drop|remove|delete)\s+(?:the\s+)?([a-zA-Z0-9_]+)\s+column\b|\b(drop|remove|delete)\s+column\s+([a-zA-Z0-9_]+)\b",
            c_lower,
        ):
            target_cols = [
                original_col
                for col_l, original_col in cols_lower_map.items()
                if re.search(rf"\b{re.escape(col_l)}\b", c_lower)
            ]
            if not target_cols:
                col_match = re.search(
                    r"\b(drop|remove|delete)\s+(?:the\s+)?([a-zA-Z0-9_]+)\s+column\b|\b(drop|remove|delete)\s+column\s+([a-zA-Z0-9_]+)\b",
                    c_lower,
                )
                if col_match:
                    cname = col_match.group(2) or col_match.group(4)
                    if cname and cname not in ["the", "this"]:
                        target_cols = [
                            cols_lower_map.get(cname.lower(), cname)
                        ]
            if target_cols:
                operations.append({
                    "type": "drop_missing_columns",
                    "columns": target_cols,
                })
                continue

        # 5. extra spaces
        if re.search(r"\bextra\s+spaces?\b|\bmultiple\s+spaces?\b", c_lower):
            operations.append({"type": "remove_extra_spaces"})
            continue

        # 6. strip spaces
        if re.search(
            r"\b(strip\s+spaces?|trim\s+spaces?|trim\s+whitespace|leading\s+and\s+trailing)\b",
            c_lower,
        ):
            operations.append({"type": "strip_spaces"})
            continue

        # 7. lowercase
        if "lowercase" in c_lower or "lower case" in c_lower:
            target_cols = [
                original_col
                for col_l, original_col in cols_lower_map.items()
                if re.search(rf"\b{re.escape(col_l)}\b", c_lower)
            ]
            op = {"type": "lowercase"}
            if target_cols:
                op["columns"] = target_cols
            operations.append(op)
            continue

        # 8. uppercase
        if "uppercase" in c_lower or "upper case" in c_lower:
            target_cols = [
                original_col
                for col_l, original_col in cols_lower_map.items()
                if re.search(rf"\b{re.escape(col_l)}\b", c_lower)
            ]
            op = {"type": "uppercase"}
            if target_cols:
                op["columns"] = target_cols
            operations.append(op)
            continue

    if operations:
        return {"operations": operations}
    return None


# ---------------------------------------------------------------------------
# High-level interface
# ---------------------------------------------------------------------------

def generate_cleaning_plan(
    user_request: str,
    df: pd.DataFrame,
    ai_response: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end cleaning-plan generation.

    If *ai_response* is provided, it is parsed and validated directly
    (useful when the caller already has the AI output).
    If *ai_response* is None, it first attempts rule-based natural-language
    parsing. If matching operations are found, the validated plan is returned.
    If no rule-based matches occur, the prompt dictionary is returned so an
    external AI model can be queried.

    Args:
        user_request: The user's natural-language request.
        df: The DataFrame being cleaned.
        ai_response: Optional raw AI model output to parse.

    Returns:
        Validated cleaning plan dictionary or prompt dictionary.

    Raises:
        ValueError: If the plan fails validation.
    """
    if ai_response is not None:
        plan = parse_cleaning_response(ai_response)
    else:
        rule_plan = parse_request_rule_based(user_request, df)
        if rule_plan and rule_plan.get("operations"):
            plan = rule_plan
        else:
            prompt = build_cleaning_prompt(user_request, df)
            return {"prompt": prompt, "status": "awaiting_ai_response"}

    errors = validate_cleaning_plan(plan, df)
    if errors:
        raise ValueError(
            "AI-generated cleaning plan failed validation:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_dataset_profile(df: pd.DataFrame) -> str:
    """Build a human-readable dataset profile for the prompt."""
    lines: List[str] = []
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")
    lines.append("Column details:")

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing = int(df[col].isnull().sum())
        unique = int(df[col].nunique())
        lines.append(
            f"  - {col}: type={dtype}, missing={missing}, "
            f"unique={unique}"
        )

    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())
    lines.append("")
    lines.append(f"Total missing values: {total_missing}")
    lines.append(f"Duplicate rows: {total_duplicates}")

    return "\n".join(lines)


# ===========================================================================
# M4 CONTRIBUTION — ANALYSIS PLAN LOGIC
# ===========================================================================

from agent.prompts import ANALYSIS_PLAN_PROMPT
from operations.analysis import SUPPORTED_ANALYSIS_OPERATIONS, SUPPORTED_AGGREGATIONS


# ---------------------------------------------------------------------------
# Analysis prompt builder
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    user_question: str,
    df: pd.DataFrame,
) -> str:
    """Build the full prompt string for the AI model to generate an analysis plan.

    Creates a concise dataset profile from *df* and injects it, along
    with the *user_question*, into the ``ANALYSIS_PLAN_PROMPT`` template.

    Args:
        user_question: Natural-language analysis question from the user.
        df: The DataFrame to analyse.

    Returns:
        Ready-to-send prompt string.
    """
    profile = _build_dataset_profile(df)
    return ANALYSIS_PLAN_PROMPT.format(
        user_question=user_question,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Analysis response parser
# ---------------------------------------------------------------------------

def parse_analysis_response(raw_response: str) -> Dict[str, Any]:
    """Extract a JSON analysis plan from the raw AI response.

    Handles common issues such as markdown code fences wrapping the
    JSON and stray text before/after the JSON block.

    Args:
        raw_response: Raw text returned by the AI model.

    Returns:
        Parsed analysis plan dictionary.

    Raises:
        ValueError: If the response does not contain valid JSON or
                    the JSON does not have the expected structure.
    """
    cleaned = raw_response.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_match = re.search(fence_pattern, cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Try to locate a JSON object if there's surrounding text
    if not cleaned.startswith("{"):
        brace_start = cleaned.find("{")
        if brace_start == -1:
            raise ValueError(
                "AI response does not contain a JSON object; not valid JSON."
            )
        cleaned = cleaned[brace_start:]

    # Find the matching closing brace
    brace_end = cleaned.rfind("}")
    if brace_end == -1:
        raise ValueError(
            "AI response contains an incomplete JSON object; not valid JSON."
        )
    cleaned = cleaned[: brace_end + 1]

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AI response is not valid JSON: {exc}"
        ) from exc

    if not isinstance(plan, dict):
        raise ValueError("Parsed JSON is not a dictionary.")

    if "operation" not in plan:
        raise ValueError(
            "Parsed JSON does not contain an 'operation' key."
        )

    return plan


# ---------------------------------------------------------------------------
# Rule-based analysis request parser
# ---------------------------------------------------------------------------

def parse_analysis_request_rule_based(
    user_question: str,
    df: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Parse common natural-language analysis questions directly.

    Identifies patterns such as:
    - "What is the average salary?"
    - "Which product has the highest sales?"
    - "Show top 5 products by revenue"
    - "Is salary correlated with experience?"
    - "Show products where sales > 100000"

    Args:
        user_question: Natural-language question text.
        df: Input DataFrame to match column names against.

    Returns:
        Structured analysis plan dictionary, or None if no
        recognized pattern was identified.
    """
    q_lower = user_question.lower().strip()
    cols_lower_map = {col.lower(): col for col in df.columns}

    def _find_column(text: str) -> Optional[str]:
        """Find a column name mentioned in text."""
        # Try exact match first (longest match wins)
        matches = []
        for col_l, original in cols_lower_map.items():
            if col_l in text:
                matches.append((col_l, original))
        if matches:
            # Return the longest match to handle multi-word columns
            matches.sort(key=lambda x: len(x[0]), reverse=True)
            return matches[0][1]
        return None

    def _find_two_columns(text: str) -> Optional[List[str]]:
        """Find two column names in text."""
        found = []
        for col_l, original in cols_lower_map.items():
            if col_l in text and original not in found:
                found.append(original)
        return found if len(found) >= 2 else None

    # 1. Average / Mean
    if re.search(r"\b(average|mean)\b", q_lower):
        # Check for "by" pattern → groupby
        by_match = re.search(r"\bby\s+(.+)", q_lower)
        if by_match:
            group_col = _find_column(by_match.group(1))
            # Find value column in the part before "by"
            before_by = q_lower.split(" by ")[0]
            value_col = _find_column(before_by)
            if group_col and value_col:
                return {
                    "operation": "groupby",
                    "group_column": group_col,
                    "value_column": value_col,
                    "aggregation": "mean",
                    "visualization": "bar",
                }
        else:
            col = _find_column(q_lower)
            if col:
                return {"operation": "mean", "column": col}

    # 2. Median
    if re.search(r"\bmedian\b", q_lower):
        col = _find_column(q_lower)
        if col:
            return {"operation": "median", "column": col}

    # 3. Mode
    if re.search(r"\bmode\b|\bmost\s+(frequent|common)\b", q_lower):
        col = _find_column(q_lower)
        if col:
            return {"operation": "mode", "column": col}

    # 4. Top-N
    top_match = re.search(r"\btop\s+(\d+)\b", q_lower)
    if top_match:
        n = int(top_match.group(1))
        col = _find_column(q_lower)
        if col:
            return {
                "operation": "top_n",
                "column": col,
                "n": n,
                "ascending": False,
                "visualization": "bar",
            }

    # 5. Correlation
    if re.search(r"\bcorrelat", q_lower):
        cols = _find_two_columns(q_lower)
        plan = {"operation": "correlation"}
        if cols:
            plan["columns"] = cols
        plan["visualization"] = "heatmap"
        return plan

    # 6. Highest / Lowest / Maximum / Minimum with groupby
    if re.search(r"\b(highest|most|largest|greatest|maximum)\b", q_lower):
        # Check for groupby pattern
        col = _find_column(q_lower)
        if col:
            # Look for another column that could be the group
            other_cols = [
                orig for cl, orig in cols_lower_map.items()
                if cl in q_lower and orig != col
            ]
            if other_cols:
                return {
                    "operation": "groupby",
                    "group_column": other_cols[0],
                    "value_column": col,
                    "aggregation": "sum",
                    "sort": "descending",
                    "limit": 1,
                    "visualization": "bar",
                }
            else:
                return {"operation": "max", "column": col}

    if re.search(r"\b(lowest|least|smallest|minimum)\b", q_lower):
        col = _find_column(q_lower)
        if col:
            other_cols = [
                orig for cl, orig in cols_lower_map.items()
                if cl in q_lower and orig != col
            ]
            if other_cols:
                return {
                    "operation": "groupby",
                    "group_column": other_cols[0],
                    "value_column": col,
                    "aggregation": "sum",
                    "sort": "ascending",
                    "limit": 1,
                    "visualization": "bar",
                }
            else:
                return {"operation": "min", "column": col}

    # 7. Count
    if re.search(r"\b(count|how\s+many)\b", q_lower):
        col = _find_column(q_lower)
        if col:
            return {"operation": "count", "column": col}

    # 8. Filter
    filter_match = re.search(
        r"where\s+(\w+)\s*(>|<|>=|<=|==|!=)\s*([\d.]+)",
        q_lower,
    )
    if filter_match:
        col_name = filter_match.group(1)
        matched_col = cols_lower_map.get(col_name)
        if matched_col:
            try:
                value = float(filter_match.group(3))
            except ValueError:
                value = filter_match.group(3)
            return {
                "operation": "filter",
                "column": matched_col,
                "operator": filter_match.group(2),
                "value": value,
            }

    # 9. Sort
    if re.search(r"\bsort\b", q_lower):
        col = _find_column(q_lower)
        if col:
            ascending = "descending" not in q_lower
            return {
                "operation": "sort",
                "column": col,
                "ascending": ascending,
            }

    # 10. Trend
    if re.search(r"\btrend\b|\bover\s+time\b|\bchange\b|\bgrowth\b", q_lower):
        # Try to find date and value columns
        date_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
            else:
                try:
                    pd.to_datetime(df[col].dropna().head(5))
                    date_cols.append(col)
                except Exception:
                    pass

        value_col = _find_column(q_lower)
        if date_cols and value_col:
            return {
                "operation": "trend",
                "date_column": date_cols[0],
                "value_column": value_col,
                "frequency": "monthly",
                "visualization": "line",
            }

    return None


# ---------------------------------------------------------------------------
# High-level analysis plan interface
# ---------------------------------------------------------------------------

def generate_analysis_plan(
    user_question: str,
    df: pd.DataFrame,
    ai_response: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end analysis-plan generation.

    If *ai_response* is provided, it is parsed directly.
    Otherwise, rule-based parsing is attempted first.  If no rule-based
    match is found, the prompt is returned so an external AI model
    can be queried.

    Args:
        user_question: The user's natural-language question.
        df: The DataFrame to analyse.
        ai_response: Optional raw AI model output to parse.

    Returns:
        Validated analysis plan dictionary, or a prompt dictionary.

    Raises:
        ValueError: If the plan fails validation.
    """
    if ai_response is not None:
        plan = parse_analysis_response(ai_response)
    else:
        rule_plan = parse_analysis_request_rule_based(user_question, df)
        if rule_plan:
            plan = rule_plan
        else:
            prompt = build_analysis_prompt(user_question, df)
            return {"prompt": prompt, "status": "awaiting_ai_response"}

    # Validate the plan
    operation = plan.get("operation")
    if operation not in SUPPORTED_ANALYSIS_OPERATIONS:
        raise ValueError(
            f"Unsupported operation: '{operation}'. "
            f"Supported: {sorted(SUPPORTED_ANALYSIS_OPERATIONS)}"
        )

    # Validate column references
    for key in ["column", "group_column", "value_column", "date_column"]:
        col = plan.get(key)
        if col and col not in df.columns:
            raise ValueError(
                f"Column '{col}' (from '{key}') does not exist. "
                f"Available: {list(df.columns)}"
            )

    if "columns" in plan and isinstance(plan["columns"], list):
        for col in plan["columns"]:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' does not exist. "
                    f"Available: {list(df.columns)}"
                )

    return plan
