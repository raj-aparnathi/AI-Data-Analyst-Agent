"""
AI Planner Module (M2 Contribution — Cleaning-Plan Logic)

Provides helpers that:
1. Build the prompt for the AI model.
2. Parse and validate the raw AI response into a cleaning plan.
3. Identify common natural-language cleaning instructions directly
   via rule-based parsing.
4. Offer a high-level *generate_cleaning_plan* function that ties
   everything together.

This module does NOT execute any Pandas operations — it only produces
the structured JSON plan that *operations/cleaning.py* will execute.
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
# MEMBER 3 CONTRIBUTION — TRANSFORMATION PLAN GENERATION
# ===========================================================================

import json as _json_m3
import re as _re_m3
from typing import Optional as _Optional_m3


def build_transformation_prompt_m3(
    user_request: str,
    df,
) -> str:
    """Build the prompt for the AI transformation planner (M3).

    Args:
        user_request: Natural-language transformation instruction.
        df: The DataFrame to be transformed.

    Returns:
        Ready-to-send prompt string.
    """
    from agent.prompts import build_transformation_prompt
    return build_transformation_prompt(user_request, df)


def parse_transformation_response(raw_response: str) -> dict:
    """Extract a JSON transformation plan from the raw AI response.

    Handles markdown code fences and stray surrounding text.

    Args:
        raw_response: Raw text returned by the AI model.

    Returns:
        Parsed transformation plan dictionary.

    Raises:
        ValueError: If the response is not valid JSON or lacks the
                    expected structure.
    """
    cleaned = raw_response.strip()

    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    fence_match = _re_m3.search(fence_pattern, cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("{"):
        brace_start = cleaned.find("{")
        if brace_start == -1:
            raise ValueError("AI response does not contain a JSON object.")
        cleaned = cleaned[brace_start:]

    brace_end = cleaned.rfind("}")
    if brace_end == -1:
        raise ValueError("AI response contains an incomplete JSON object.")
    cleaned = cleaned[: brace_end + 1]

    try:
        plan = _json_m3.loads(cleaned)
    except _json_m3.JSONDecodeError as exc:
        raise ValueError(f"AI response is not valid JSON: {exc}") from exc

    if not isinstance(plan, dict):
        raise ValueError("Parsed JSON is not a dictionary.")
    if "operations" not in plan:
        raise ValueError("Parsed JSON does not contain an 'operations' key.")

    return plan


def parse_transformation_request_rule_based(
    user_request: str,
    df,
) -> _Optional_m3[dict]:
    """Parse common natural-language transformation instructions directly.

    Covers requests such as:
    - "Convert gender to one-hot encoding."
    - "Label encode the city column."
    - "Normalize salary using Min-Max."
    - "Standardize age using Z-score."
    - "Convert age to integer."
    - "Rename salary to annual_salary."
    - "Drop the customer_id column."
    - "Create total_price from quantity and price."
    - "Extract year from purchase_date."

    Args:
        user_request: Natural-language request text.
        df: Input DataFrame to match column names against.

    Returns:
        Structured plan dict with 'operations' list, or None if no
        recognized operations were identified.
    """
    import pandas as pd

    operations = []
    clauses = _re_m3.split(
        r"\s*(?:;|\band\b|\bthen\b|,|\n)\s*",
        user_request,
        flags=_re_m3.IGNORECASE,
    )
    clauses = [c.strip() for c in clauses if c.strip()]
    cols_lower_map = {col.lower(): col for col in df.columns}

    for clause in clauses:
        c = clause.lower()

        # one_hot_encode
        if _re_m3.search(
            r"\b(one.?hot|dummy|dummies|get.?dummies)\b", c
        ):
            matched = _find_column(c, cols_lower_map)
            if matched:
                operations.append({
                    "operation": "one_hot_encode",
                    "column": matched,
                })
            continue

        # label_encode
        if _re_m3.search(r"\blabel.?encod", c):
            matched = _find_column(c, cols_lower_map)
            if matched:
                operations.append({
                    "operation": "label_encode",
                    "column": matched,
                })
            continue

        # min_max_normalize
        if _re_m3.search(r"\b(min.?max|minmax)\b", c):
            matched = _find_column(c, cols_lower_map)
            if matched:
                operations.append({
                    "operation": "min_max_normalize",
                    "column": matched,
                })
            continue

        # z_score_normalize
        if _re_m3.search(r"\b(z.?score|standardize|zscore)\b", c):
            matched = _find_column(c, cols_lower_map)
            if matched:
                operations.append({
                    "operation": "z_score_normalize",
                    "column": matched,
                })
            continue

        # drop_column
        if _re_m3.search(
            r"\b(drop|remove|delete)\s+(?:the\s+)?([a-z0-9_]+)\s+column\b"
            r"|\b(drop|remove|delete)\s+column\s+([a-z0-9_]+)\b",
            c,
        ):
            matched = _find_column(c, cols_lower_map)
            if matched:
                operations.append({
                    "operation": "drop_column",
                    "column": matched,
                })
            continue

        # rename_column: "rename X to Y" or "rename X as Y"
        rename_match = _re_m3.search(
            r"\brename\s+([a-z0-9_]+)\s+(?:to|as)\s+([a-z0-9_]+)\b", c
        )
        if rename_match:
            old_l, new_l = rename_match.group(1), rename_match.group(2)
            old_real = cols_lower_map.get(old_l, old_l)
            operations.append({
                "operation": "rename_column",
                "old_name": old_real,
                "new_name": new_l,
            })
            continue

        # convert_datatype: "convert X to int/float/datetime/..."
        dtype_match = _re_m3.search(
            r"\bconvert\s+([a-z0-9_]+)\s+to\s+(int|float|string|bool|datetime)\b",
            c,
        )
        if dtype_match:
            col_l, dtype_str = dtype_match.group(1), dtype_match.group(2)
            real_col = cols_lower_map.get(col_l, col_l)
            operations.append({
                "operation": "convert_datatype",
                "column": real_col,
                "dtype": dtype_str,
            })
            continue

        # extract_date_part: "extract year/month/day/dayofweek from X"
        date_match = _re_m3.search(
            r"\bextract\s+(year|month|day|dayofweek)\s+from\s+([a-z0-9_]+)\b",
            c,
        )
        if date_match:
            part, col_l = date_match.group(1), date_match.group(2)
            real_col = cols_lower_map.get(col_l, col_l)
            operations.append({
                "operation": "extract_date_part",
                "column": real_col,
                "part": part,
            })
            continue

    if operations:
        return {"operations": operations}
    return None


def generate_transformation_plan(
    user_request: str,
    df,
    ai_response: _Optional_m3[str] = None,
) -> dict:
    """End-to-end transformation-plan generation (M3).

    If *ai_response* is provided it is parsed and validated directly.
    Otherwise, rule-based parsing is attempted first; if that yields
    operations the validated plan is returned.  If rule-based parsing
    finds nothing, a prompt dict is returned so the caller can query
    the AI model.

    Args:
        user_request: Natural-language transformation request.
        df: The DataFrame to transform.
        ai_response: Optional raw AI model output to parse.

    Returns:
        Validated transformation plan dict  OR
        ``{"prompt": ..., "status": "awaiting_ai_response"}`` if the AI
        needs to be called.

    Raises:
        ValueError: If the plan fails validation.
    """
    from operations.transformation import validate_transformation_plan

    if ai_response is not None:
        plan = parse_transformation_response(ai_response)
    else:
        rule_plan = parse_transformation_request_rule_based(user_request, df)
        if rule_plan and rule_plan.get("operations"):
            plan = rule_plan
        else:
            prompt = build_transformation_prompt_m3(user_request, df)
            return {"prompt": prompt, "status": "awaiting_ai_response"}

    errors = validate_transformation_plan(plan, df)
    if errors:
        raise ValueError(
            "Transformation plan failed validation:\n"
            + "\n".join(f"  * {e}" for e in errors)
        )

    return plan


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _find_column(clause_lower: str, cols_lower_map: dict) -> _Optional_m3[str]:
    """Return the real column name whose lower-case form appears in clause."""
    for col_l, original in cols_lower_map.items():
        if _re_m3.search(rf"\b{_re_m3.escape(col_l)}\b", clause_lower):
            return original
    return None
