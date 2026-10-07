"""
agent/prompts.py — AI Prompt Templates (M1 Contribution)
========================================================
Reusable prompt templates that format structured dataset profile
information into natural-language context for the AI agent.

**Scope**: Only dataset-understanding prompts are defined here.
Other team members will add their own prompt sections (cleaning,
analysis, visualisation, etc.) separately.

Usage
-----
    from agent.prompts import build_dataset_summary_prompt
    prompt = build_dataset_summary_prompt(profile)

Author : Member 1 (Dataset Understanding & File Handling)
"""


# ---------------------------------------------------------------------------
# 1. Dataset Summary Prompt
# ---------------------------------------------------------------------------

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
    if profile["missing_values"]:
        lines = [
            f"  - {col}: {count} missing ({profile['missing_percentages'][col]}%)"
            for col, count in profile["missing_values"].items()
        ]
        missing_detail = "\n".join(lines)
    else:
        missing_detail = "  None"

    prompt = f"""You are a data analyst assistant.

The user has uploaded a dataset with the following characteristics:

**Shape**: {profile['rows']:,} rows × {profile['columns']} columns

**Column Names**: {', '.join(profile['column_names'])}

**Column Classification**:
- Numerical columns : {', '.join(profile['numerical_columns']) or 'None'}
- Categorical columns: {', '.join(profile['categorical_columns']) or 'None'}
- Datetime columns   : {', '.join(profile['datetime_columns']) or 'None'}
- Identifier columns : {', '.join(profile.get('id_columns', [])) or 'None'}

**Missing Values** (total: {profile['total_missing']:,}):
{missing_detail}

**Duplicate Rows**: {profile['duplicate_rows']:,}

**Data Types**:
{_format_dict(profile['data_types'])}

**Memory Usage**: {profile['memory_usage']}

Based on the information above, provide a concise summary of the dataset.
Highlight any potential data quality issues and suggest what kind of
analysis might be appropriate for this data."""

    return prompt


# ---------------------------------------------------------------------------
# 2. Column Identification Prompt
# ---------------------------------------------------------------------------

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
    for col in profile["column_names"]:
        dtype = profile["data_types"][col]
        unique = profile["unique_values"][col]
        missing = profile["missing_values"].get(col, 0)
        category = _classify_column(col, profile)
        column_details.append(
            f"- **{col}**: dtype={dtype}, unique={unique}, "
            f"missing={missing}, classification={category}"
        )

    columns_text = "\n".join(column_details)

    prompt = f"""You are a data analyst assistant.

The user has uploaded a dataset with {profile['rows']:,} rows and {profile['columns']} columns.
Below is a summary of each column:

{columns_text}

For each column, please:
1. Describe what the column likely represents.
2. Confirm or suggest a better classification (numerical / categorical / datetime / identifier).
3. Note any potential issues (e.g. high missing rate, low cardinality numeric)."""

    return prompt


# ---------------------------------------------------------------------------
# 3. Data Quality Prompt
# ---------------------------------------------------------------------------

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

    # Missing values section
    if profile["missing_values"]:
        missing_lines = [
            f"  - {col}: {count} ({profile['missing_percentages'][col]}%)"
            for col, count in profile["missing_values"].items()
        ]
        missing_section = "\n".join(missing_lines)
    else:
        missing_section = "  No missing values."

    prompt = f"""You are a data quality expert.

Analyse the following data quality report and provide actionable recommendations.

**Dataset Size**: {profile['rows']:,} rows × {profile['columns']} columns

**Missing Values** (total: {profile['total_missing']:,}):
{missing_section}

**Duplicate Rows**: {profile['duplicate_rows']:,}

**Data Types**:
{_format_dict(profile['data_types'])}

**Unique Value Counts**:
{_format_dict(profile['unique_values'])}

Please:
1. Assess the overall data quality (Good / Fair / Poor).
2. For each column with missing values, suggest an imputation strategy.
3. Recommend whether duplicate rows should be removed and why.
4. Flag any columns whose data type may need conversion.
5. Suggest any additional data cleaning steps."""

    return prompt


# ---------------------------------------------------------------------------
# 4. Context Builder (for other agents)
# ---------------------------------------------------------------------------

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
- **Rows**: {profile['rows']:,}
- **Columns**: {profile['columns']}
- **Column Names**: {', '.join(profile['column_names'])}
- **Numerical**: {', '.join(profile['numerical_columns']) or 'None'}
- **Categorical**: {', '.join(profile['categorical_columns']) or 'None'}
- **Datetime**: {', '.join(profile['datetime_columns']) or 'None'}
- **Identifiers**: {', '.join(profile.get('id_columns', [])) or 'None'}
- **Total Missing**: {profile['total_missing']:,}
- **Duplicate Rows**: {profile['duplicate_rows']:,}
- **Memory**: {profile['memory_usage']}"""

    return context


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _format_dict(d: dict) -> str:
    """Format a dict as an indented list of `key: value` lines."""
    lines = [f"  - {k}: {v}" for k, v in d.items()]
    return "\n".join(lines)


def _classify_column(col: str, profile: dict) -> str:
    """Return the classification label for a column from the profile."""
    if col in profile["numerical_columns"]:
        return "numerical"
    if col in profile["categorical_columns"]:
        return "categorical"
    if col in profile["datetime_columns"]:
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
            "Signup_Date": "datetime64[ns]"
        },
        "unique_values": {
            "Customer_ID": 1000,
            "Age": 45,
            "Salary": 820,
            "City": 8,
            "Signup_Date": 300
        },
        "memory_usage": "156.40 KB"
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
