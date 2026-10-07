"""
AI Prompts Module (M2 Contribution — Data Cleaning)

Contains reusable prompt templates for the AI planner to generate
structured JSON cleaning plans from natural-language user requests.

Other team members may add their own prompts to this file.
"""

# ---------------------------------------------------------------------------
# Cleaning-Plan Generation Prompt
# ---------------------------------------------------------------------------

CLEANING_PLAN_PROMPT = """You are a data-cleaning planner.

Given:
1. A user request describing the cleaning operations they want.
2. A dataset profile with column names, types, row count, and missing-value
   information.

Your task is to create a **JSON cleaning plan** — a single JSON object with
one key called "operations" whose value is an ordered list of operations.

──────────────────────────────────────────────────────────────
SUPPORTED OPERATION TYPES
──────────────────────────────────────────────────────────────

1. remove_duplicates
   No extra parameters.

2. drop_missing_rows
   Optional parameter:
     - columns (list[str]): only consider these columns when deciding
       which rows to drop.  Omit to consider all columns.

3. drop_missing_columns
   Required parameter:
     - columns (list[str]): column names to remove.

4. fill_missing
   Required parameters:
     - column (str): exact column name from the dataset.
     - method (str): one of "mean", "median", "mode",
       "forward_fill", "backward_fill".
   Rules for method selection:
     - "mean" and "median" → ONLY for numerical columns.
     - "mode" → works for both numerical and categorical columns.
     - "forward_fill" / "backward_fill" → works for any column type.

5. strip_spaces
   No extra parameters.  Strips leading/trailing whitespace from all
   string columns.

6. remove_extra_spaces
   No extra parameters.  Collapses multiple internal spaces into one
   in all string columns.

7. lowercase
   Optional parameter:
     - columns (list[str]): specific columns to convert.
       Omit to apply to all string columns.

8. uppercase
   Optional parameter:
     - columns (list[str]): specific columns to convert.
       Omit to apply to all string columns.

──────────────────────────────────────────────────────────────
STRICT RULES
──────────────────────────────────────────────────────────────

1. Use ONLY the operation types listed above.
2. Use EXACT column names from the dataset profile — never invent columns.
3. Do NOT perform operations the user did not request.
4. Do NOT include explanations, markdown, or text outside the JSON.
5. Return ONLY valid JSON.
6. For "fill_missing", always include both "column" and "method".
7. Do NOT use "mean" or "median" on non-numerical columns.

──────────────────────────────────────────────────────────────
EXAMPLES
──────────────────────────────────────────────────────────────

User: "Remove duplicates and fill missing salary with median."
Dataset profile shows salary is float64.

{{
  "operations": [
    {{"type": "remove_duplicates"}},
    {{"type": "fill_missing", "column": "salary", "method": "median"}}
  ]
}}

User: "Replace missing city values with mode and convert city to lowercase."
Dataset profile shows city is object.

{{
  "operations": [
    {{"type": "fill_missing", "column": "city", "method": "mode"}},
    {{"type": "lowercase", "columns": ["city"]}}
  ]
}}

──────────────────────────────────────────────────────────────
YOUR INPUTS
──────────────────────────────────────────────────────────────

User request:
{user_request}

Dataset profile:
{profile}

Return the JSON cleaning plan now.
"""


# ---------------------------------------------------------------------------
# Cleaning-Explanation Prompt  (optional, used by the explainer/UI)
# ---------------------------------------------------------------------------

CLEANING_EXPLANATION_PROMPT = """You are a helpful data-analyst assistant.

The user requested the following data-cleaning operations, and they have
been applied successfully.

Cleaning report:
{report}

Provide a short, friendly summary of what was done and how the dataset
changed.  Use bullet points.  Keep it under 150 words.
"""
