"""
agent/prompts.py — AI Prompt Templates
========================================
Reusable prompt templates that format dataset context and requests
for the AI agent.

Contributions:
- Member 1: Dataset understanding & profiling prompts
- Member 2: AI-driven data cleaning prompts
- Member 4: Analysis planning & result explanation prompts
"""

# ===========================================================================
# MEMBER 2 CONTRIBUTION — DATA CLEANING PROMPTS
# ===========================================================================

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


CLEANING_EXPLANATION_PROMPT = """You are a helpful data-analyst assistant.

The user requested the following data-cleaning operations, and they have
been applied successfully.

Cleaning report:
{report}

Provide a short, friendly summary of what was done and how the dataset
changed.  Use bullet points.  Keep it under 150 words.
"""


# ===========================================================================
# MEMBER 1 CONTRIBUTION — DATASET UNDERSTANDING PROMPTS
# ===========================================================================

def build_dataset_summary_prompt(profile: dict) -> str:
    """
    Build a prompt that asks the AI to summarise the dataset.

    The prompt embeds the structured profile so the model has full
    context without needing to see the raw data.

    Parameters
    ----------
    profile : dict
        The dictionary returned by ``profile_dataset(df)``.

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    missing_detail = ""
    if profile.get("missing_values"):
        lines = [
            f"  - {col}: {count} missing ({profile.get('missing_percentages', {}).get(col, 0)}%)"
            for col, count in profile["missing_values"].items()
        ]
        missing_detail = "\n".join(lines)
    else:
        missing_detail = "  None"

    prompt = f"""You are a data analyst assistant.

The user has uploaded a dataset with the following characteristics:

**Shape**: {profile.get('rows', 0):,} rows × {profile.get('columns', 0)} columns

**Column Names**: {', '.join(profile.get('column_names', []))}

**Column Classification**:
- Numerical columns : {', '.join(profile.get('numerical_columns', [])) or 'None'}
- Categorical columns: {', '.join(profile.get('categorical_columns', [])) or 'None'}
- Datetime columns   : {', '.join(profile.get('datetime_columns', [])) or 'None'}
- Identifier columns : {', '.join(profile.get('id_columns', [])) or 'None'}

**Missing Values** (total: {profile.get('total_missing', 0):,}):
{missing_detail}

**Duplicate Rows**: {profile.get('duplicate_rows', 0):,}

**Data Types**:
{_format_dict(profile.get('data_types', {}))}

**Memory Usage**: {profile.get('memory_usage', 'N/A')}

Based on the information above, provide a concise summary of the dataset.
Highlight any potential data quality issues and suggest what kind of
analysis might be appropriate for this data."""

    return prompt


def build_column_identification_prompt(profile: dict) -> str:
    """
    Build a prompt that asks the AI to analyse and describe each column.

    Useful for generating documentation or helping the user understand
    what each column represents.

    Parameters
    ----------
    profile : dict
        The dictionary returned by ``profile_dataset(df)``.

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    column_details = []
    for col in profile.get("column_names", []):
        dtype = profile.get("data_types", {}).get(col, "unknown")
        unique = profile.get("unique_values", {}).get(col, 0)
        missing = profile.get("missing_values", {}).get(col, 0)
        category = _classify_column(col, profile)
        column_details.append(
            f"- **{col}**: dtype={dtype}, unique={unique}, "
            f"missing={missing}, classification={category}"
        )

    columns_text = "\n".join(column_details)

    prompt = f"""You are a data analyst assistant.

The user has uploaded a dataset with {profile.get('rows', 0):,} rows and {profile.get('columns', 0)} columns.
Below is a summary of each column:

{columns_text}

For each column, please:
1. Describe what the column likely represents.
2. Confirm or suggest a better classification (numerical / categorical / datetime / identifier).
3. Note any potential issues (e.g. high missing rate, low cardinality numeric)."""

    return prompt


def build_data_quality_prompt(profile: dict) -> str:
    """
    Build a prompt focused on data quality issues and recommended actions.

    Parameters
    ----------
    profile : dict
        The dictionary returned by ``profile_dataset(df)``.

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    if profile.get("missing_values"):
        missing_lines = [
            f"  - {col}: {count} ({profile.get('missing_percentages', {}).get(col, 0)}%)"
            for col, count in profile["missing_values"].items()
        ]
        missing_section = "\n".join(missing_lines)
    else:
        missing_section = "  No missing values."

    prompt = f"""You are a data quality expert.

Analyse the following data quality report and provide actionable recommendations.

**Dataset Size**: {profile.get('rows', 0):,} rows × {profile.get('columns', 0)} columns

**Missing Values** (total: {profile.get('total_missing', 0):,}):
{missing_section}

**Duplicate Rows**: {profile.get('duplicate_rows', 0):,}

**Data Types**:
{_format_dict(profile.get('data_types', {}))}

**Unique Value Counts**:
{_format_dict(profile.get('unique_values', {}))}

Please:
1. Assess the overall data quality (Good / Fair / Poor).
2. For each column with missing values, suggest an imputation strategy.
3. Recommend whether duplicate rows should be removed and why.
4. Flag any columns whose data type may need conversion.
5. Suggest any additional data cleaning steps."""

    return prompt


def build_dataset_context(profile: dict) -> str:
    """
    Return a compact context block that other agents/prompts can embed
    at the top of their own prompts to give the model awareness of the
    dataset structure.

    This is NOT a full prompt — it's a reusable *context snippet*.

    Parameters
    ----------
    profile : dict
        The dictionary returned by ``profile_dataset(df)``.

    Returns
    -------
    str
        A compact, markdown-formatted context block.
    """
    context = f"""### Dataset Context
- **Rows**: {profile.get('rows', 0):,}
- **Columns**: {profile.get('columns', 0)}
- **Column Names**: {', '.join(profile.get('column_names', []))}
- **Numerical**: {', '.join(profile.get('numerical_columns', [])) or 'None'}
- **Categorical**: {', '.join(profile.get('categorical_columns', [])) or 'None'}
- **Datetime**: {', '.join(profile.get('datetime_columns', [])) or 'None'}
- **Identifiers**: {', '.join(profile.get('id_columns', [])) or 'None'}
- **Total Missing**: {profile.get('total_missing', 0):,}
- **Duplicate Rows**: {profile.get('duplicate_rows', 0):,}
- **Memory**: {profile.get('memory_usage', 'N/A')}"""

    return context


# ---------------------------------------------------------------------------
# Private helpers for Member 1 prompts
# ---------------------------------------------------------------------------

def _format_dict(d: dict) -> str:
    """Format a dict as an indented list of `key: value` lines."""
    lines = [f"  - {k}: {v}" for k, v in d.items()]
    return "\n".join(lines)


def _classify_column(col: str, profile: dict) -> str:
    """Return the classification label for a column from the profile."""
    if col in profile.get("numerical_columns", []):
        return "numerical"
    if col in profile.get("categorical_columns", []):
        return "categorical"
    if col in profile.get("datetime_columns", []):
        return "datetime"
    if col in profile.get("id_columns", []):
        return "identifier"
    return "unknown"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing agent/prompts.py independently")
    print("=" * 60)

    sample_profile = {
        "rows": 1000,
        "columns": 5,
        "column_names": ["Customer_ID", "Age", "Salary", "City", "Signup_Date"],
        "numerical_columns": ["Age", "Salary"],
        "categorical_columns": ["City"],
        "datetime_columns": ["Signup_Date"],
        "id_columns": ["Customer_ID"],
        "missing_values": {"Salary": 12, "City": 3},
        "missing_percentages": {"Salary": 1.2, "City": 0.3},
        "total_missing": 15,
        "duplicate_rows": 2,
        "data_types": {
            "Customer_ID": "int64",
            "Age": "int64",
            "Salary": "float64",
            "City": "object",
            "Signup_Date": "datetime64[ns]",
        },
        "unique_values": {
            "Customer_ID": 1000,
            "Age": 45,
            "Salary": 820,
            "City": 8,
            "Signup_Date": 300,
        },
        "memory_usage": "156.40 KB",
    }

    # 1. Summary Prompt
    print("\n--- 1. DATASET SUMMARY PROMPT ---")
    print(build_dataset_summary_prompt(sample_profile)[:300] + "\n...")

    # 2. Column Identification Prompt
    print("\n--- 2. COLUMN IDENTIFICATION PROMPT ---")
    print(build_column_identification_prompt(sample_profile)[:300] + "\n...")

    # 3. Data Quality Prompt
    print("\n--- 3. DATA QUALITY PROMPT ---")
    print(build_data_quality_prompt(sample_profile)[:300] + "\n...")

    # 4. Context Builder
    print("\n--- 4. DATASET CONTEXT SNIPPET ---")
    print(build_dataset_context(sample_profile))

    print("\n[SUCCESS] agent/prompts.py passed standalone checks.")


# ===========================================================================
# MEMBER 4 CONTRIBUTION — ANALYSIS & VISUALIZATION PROMPTS
# ===========================================================================

ANALYSIS_PLAN_PROMPT = """You are an AI data-analysis planner.

Your task is to convert the user's natural-language question
into a safe structured JSON analysis plan.

Supported operations:
- mean          (requires: column)
- median        (requires: column)
- mode          (requires: column)
- min           (requires: column)
- max           (requires: column)
- count         (requires: column)
- variance      (requires: column)
- std           (requires: column)
- quartiles     (requires: column)
- groupby       (requires: group_column, value_column, aggregation)
                 aggregation must be one of: sum, mean, median, min, max, count
                 optional: sort ("ascending"/"descending"), limit (integer)
- sort          (requires: column)
                 optional: ascending (true/false), limit (integer)
- filter        (requires: column, operator, value)
                 operator must be one of: >, <, >=, <=, ==, !=
- top_n         (requires: column, n)
                 optional: ascending (true/false)
- correlation   (optional: columns as list)
- trend         (requires: date_column, value_column)
                 optional: frequency ("daily"/"monthly"/"yearly")

Supported visualizations:
- bar
- line
- histogram
- scatter
- pie
- box
- heatmap

Rules:
1. Return valid JSON only.
2. Use exact column names from the dataset.
3. Never invent columns.
4. Never generate Python code.
5. Never use arbitrary expressions.
6. Do not calculate the answer yourself.
7. Use only supported operations.
8. Select a visualization only when appropriate.
9. Do not modify the dataset.
10. Do not claim causation from correlation.

User question:
{user_question}

Dataset profile:
{profile}

Return a single JSON object with the analysis plan now.
"""


ANALYSIS_EXPLANATION_PROMPT = """You are a data analyst explaining an analysis result to a user.

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
3. Do not perform new calculations unless the provided result already
   contains the required values.
4. Do not claim causation from correlation.
5. Mention important values when useful.
6. Use simple language.
7. If the result is empty, clearly say that no matching data was found.
"""
