"""
Tests for operations/cleaning.py (M2)

Covers: remove_duplicates, strip_spaces, remove_extra_spaces,
to_lowercase, to_uppercase, validate_cleaning_plan, and
apply_cleaning_plan — including multi-step plans and error handling.
"""

import numpy as np
import pandas as pd
import pytest

from operations.cleaning import (
    SUPPORTED_OPERATIONS,
    apply_cleaning_plan,
    remove_duplicates,
    remove_extra_spaces,
    strip_spaces,
    to_lowercase,
    to_uppercase,
    validate_cleaning_plan,
)
from agent.planner import (
    build_cleaning_prompt,
    generate_cleaning_plan,
    parse_cleaning_response,
    parse_request_rule_based,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame with duplicates, whitespace issues, and missing values."""
    return pd.DataFrame({
        "name": ["  Raj  ", "Priya", "Raj   Aparnathi", "Amit", "  Raj  "],
        "city": ["AHMEDABAD", "surat", "Ahmedabad", "RAJKOT", "AHMEDABAD"],
        "salary": [45000.0, 50000.0, 52000.0, np.nan, 45000.0],
        "age": [25, 30, 30, 28, 25],
    })


@pytest.fixture
def dup_df():
    """DataFrame with exact duplicate rows."""
    return pd.DataFrame({
        "a": [1, 2, 2, 3, 3, 3],
        "b": ["x", "y", "y", "z", "z", "z"],
    })


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_duplicates(self, dup_df):
        result, removed = remove_duplicates(dup_df)
        assert removed == 3  # 1 dup of (2,y), 2 dups of (3,z)
        assert len(result) == 3

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, removed = remove_duplicates(df)
        assert removed == 0
        assert len(result) == 3

    def test_does_not_mutate_original(self, dup_df):
        original_len = len(dup_df)
        _ = remove_duplicates(dup_df)
        assert len(dup_df) == original_len


# ---------------------------------------------------------------------------
# strip_spaces
# ---------------------------------------------------------------------------

class TestStripSpaces:
    def test_strips_leading_trailing(self):
        df = pd.DataFrame({"name": ["  Raj  ", "  Priya  "]})
        result = strip_spaces(df)
        assert result["name"].tolist() == ["Raj", "Priya"]

    def test_preserves_internal_spaces(self):
        df = pd.DataFrame({"name": ["  Raj Aparnathi  "]})
        result = strip_spaces(df)
        assert result["name"].iloc[0] == "Raj Aparnathi"

    def test_ignores_numeric_columns(self):
        df = pd.DataFrame({"num": [1, 2, 3], "text": [" a ", " b ", " c "]})
        result = strip_spaces(df)
        assert result["num"].tolist() == [1, 2, 3]
        assert result["text"].tolist() == ["a", "b", "c"]

    def test_does_not_mutate_original(self):
        df = pd.DataFrame({"name": ["  Raj  "]})
        _ = strip_spaces(df)
        assert df["name"].iloc[0] == "  Raj  "


# ---------------------------------------------------------------------------
# remove_extra_spaces
# ---------------------------------------------------------------------------

class TestRemoveExtraSpaces:
    def test_collapses_internal_spaces(self):
        df = pd.DataFrame({"name": ["Raj   Aparnathi"]})
        result = remove_extra_spaces(df)
        assert result["name"].iloc[0] == "Raj Aparnathi"

    def test_also_strips_edges(self):
        df = pd.DataFrame({"name": ["  Raj   Aparnathi  "]})
        result = remove_extra_spaces(df)
        assert result["name"].iloc[0] == "Raj Aparnathi"

    def test_ignores_numeric_columns(self):
        df = pd.DataFrame({"num": [1, 2], "text": ["a  b", "c  d"]})
        result = remove_extra_spaces(df)
        assert result["num"].tolist() == [1, 2]
        assert result["text"].tolist() == ["a b", "c d"]


# ---------------------------------------------------------------------------
# to_lowercase
# ---------------------------------------------------------------------------

class TestToLowercase:
    def test_lowercase_all_string_cols(self):
        df = pd.DataFrame({"city": ["AHMEDABAD", "SURAT"]})
        result = to_lowercase(df)
        assert result["city"].tolist() == ["ahmedabad", "surat"]

    def test_lowercase_specific_columns(self):
        df = pd.DataFrame({
            "city": ["AHMEDABAD"],
            "name": ["RAJ"],
        })
        result = to_lowercase(df, columns=["city"])
        assert result["city"].iloc[0] == "ahmedabad"
        assert result["name"].iloc[0] == "RAJ"  # untouched

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"city": ["A"]})
        with pytest.raises(ValueError, match="not found"):
            to_lowercase(df, columns=["nonexistent"])


# ---------------------------------------------------------------------------
# to_uppercase
# ---------------------------------------------------------------------------

class TestToUppercase:
    def test_uppercase_all_string_cols(self):
        df = pd.DataFrame({"city": ["ahmedabad", "surat"]})
        result = to_uppercase(df)
        assert result["city"].tolist() == ["AHMEDABAD", "SURAT"]

    def test_uppercase_specific_columns(self):
        df = pd.DataFrame({
            "city": ["ahmedabad"],
            "name": ["raj"],
        })
        result = to_uppercase(df, columns=["city"])
        assert result["city"].iloc[0] == "AHMEDABAD"
        assert result["name"].iloc[0] == "raj"  # untouched

    def test_invalid_column_raises(self):
        df = pd.DataFrame({"city": ["a"]})
        with pytest.raises(ValueError, match="not found"):
            to_uppercase(df, columns=["nonexistent"])


# ---------------------------------------------------------------------------
# validate_cleaning_plan
# ---------------------------------------------------------------------------

class TestValidateCleaningPlan:
    def test_valid_plan(self, sample_df):
        plan = {
            "operations": [
                {"type": "remove_duplicates"},
                {"type": "fill_missing", "column": "salary", "method": "median"},
            ]
        }
        errors = validate_cleaning_plan(plan, sample_df)
        assert errors == []

    def test_missing_operations_key(self, sample_df):
        errors = validate_cleaning_plan({}, sample_df)
        assert any("'operations'" in e for e in errors)

    def test_unsupported_operation(self, sample_df):
        plan = {"operations": [{"type": "magic_clean"}]}
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("unsupported" in e.lower() for e in errors)

    def test_fill_missing_no_column(self, sample_df):
        plan = {"operations": [{"type": "fill_missing", "method": "mean"}]}
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("column" in e.lower() for e in errors)

    def test_fill_missing_no_method(self, sample_df):
        plan = {"operations": [{"type": "fill_missing", "column": "salary"}]}
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("method" in e.lower() for e in errors)

    def test_fill_missing_invalid_method(self, sample_df):
        plan = {
            "operations": [
                {"type": "fill_missing", "column": "salary", "method": "unknown_method"}
            ]
        }
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("unsupported" in e.lower() or "unknown_method" in e for e in errors)

    def test_fill_missing_nonexistent_column(self, sample_df):
        plan = {
            "operations": [
                {"type": "fill_missing", "column": "bonus", "method": "mean"}
            ]
        }
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("does not exist" in e for e in errors)

    def test_drop_missing_columns_no_columns(self, sample_df):
        plan = {"operations": [{"type": "drop_missing_columns"}]}
        errors = validate_cleaning_plan(plan, sample_df)
        assert any("columns" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# apply_cleaning_plan
# ---------------------------------------------------------------------------

class TestApplyCleaningPlan:
    def test_single_remove_duplicates(self, dup_df):
        plan = {"operations": [{"type": "remove_duplicates"}]}
        cleaned, report = apply_cleaning_plan(dup_df, plan)
        assert report["duplicates_removed"] == 3
        assert report["rows_after"] == 3
        assert "remove_duplicates" in report["operations_applied"]

    def test_single_fill_missing_median(self):
        df = pd.DataFrame({"salary": [45000.0, 50000.0, 52000.0, np.nan]})
        plan = {
            "operations": [
                {"type": "fill_missing", "column": "salary", "method": "median"}
            ]
        }
        cleaned, report = apply_cleaning_plan(df, plan)
        assert cleaned["salary"].isnull().sum() == 0
        assert report["missing_values_filled"] == 1

    def test_multi_step_plan(self, sample_df):
        plan = {
            "operations": [
                {"type": "remove_duplicates"},
                {"type": "fill_missing", "column": "salary", "method": "median"},
                {"type": "strip_spaces"},
                {"type": "lowercase", "columns": ["city"]},
            ]
        }
        cleaned, report = apply_cleaning_plan(sample_df, plan)
        assert len(report["operations_applied"]) == 4
        assert cleaned["salary"].isnull().sum() == 0

    def test_report_tracks_before_after(self, sample_df):
        plan = {"operations": [{"type": "remove_duplicates"}]}
        _, report = apply_cleaning_plan(sample_df, plan)
        assert "rows_before" in report
        assert "rows_after" in report
        assert "missing_values_before" in report
        assert "missing_values_after" in report

    def test_invalid_plan_raises(self, sample_df):
        plan = {"operations": [{"type": "nonexistent_op"}]}
        with pytest.raises(ValueError, match="validation failed"):
            apply_cleaning_plan(sample_df, plan)

    def test_does_not_mutate_original(self, sample_df):
        original_copy = sample_df.copy()
        plan = {
            "operations": [
                {"type": "remove_duplicates"},
                {"type": "fill_missing", "column": "salary", "method": "mean"},
            ]
        }
        _ = apply_cleaning_plan(sample_df, plan)
        pd.testing.assert_frame_equal(sample_df, original_copy)

    def test_empty_plan(self, sample_df):
        plan = {"operations": []}
        cleaned, report = apply_cleaning_plan(sample_df, plan)
        assert report["rows_before"] == report["rows_after"]
        assert report["operations_applied"] == []


# ---------------------------------------------------------------------------
# Planner Tests (M2 Planner & Intent Identification)
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_identify_remove_duplicates(self, sample_df):
        plan = generate_cleaning_plan("Remove duplicate rows", sample_df)
        assert plan["operations"] == [{"type": "remove_duplicates"}]

    def test_identify_fill_missing_median(self, sample_df):
        plan = generate_cleaning_plan("Fill missing salary with median", sample_df)
        assert plan["operations"] == [
            {"type": "fill_missing", "column": "salary", "method": "median"}
        ]

    def test_identify_replace_missing_mode(self, sample_df):
        plan = generate_cleaning_plan("Replace missing city values with mode", sample_df)
        assert plan["operations"] == [
            {"type": "fill_missing", "column": "city", "method": "mode"}
        ]

    def test_identify_remove_rows_with_missing_values(self, sample_df):
        plan = generate_cleaning_plan("Remove rows with missing values", sample_df)
        assert plan["operations"] == [{"type": "drop_missing_rows"}]

    def test_identify_drop_city_column(self, sample_df):
        plan = generate_cleaning_plan("Drop the city column", sample_df)
        assert plan["operations"] == [
            {"type": "drop_missing_columns", "columns": ["city"]}
        ]

    def test_identify_convert_to_lowercase(self, sample_df):
        plan = generate_cleaning_plan("Convert city names to lowercase", sample_df)
        assert plan["operations"] == [
            {"type": "lowercase", "columns": ["city"]}
        ]

    def test_identify_remove_extra_spaces(self, sample_df):
        plan = generate_cleaning_plan("Remove extra spaces from names", sample_df)
        assert plan["operations"] == [{"type": "remove_extra_spaces"}]

    def test_identify_combined_request(self, sample_df):
        plan = generate_cleaning_plan(
            "Remove duplicates and fill missing salary with median",
            sample_df,
        )
        assert plan["operations"] == [
            {"type": "remove_duplicates"},
            {"type": "fill_missing", "column": "salary", "method": "median"},
        ]

    def test_parse_ai_response_with_code_fence(self, sample_df):
        raw = """```json
        {
            "operations": [
                {"type": "remove_duplicates"},
                {"type": "strip_spaces"}
            ]
        }
        ```"""
        plan = parse_cleaning_response(raw)
        assert len(plan["operations"]) == 2
        assert plan["operations"][0]["type"] == "remove_duplicates"

    def test_parse_ai_response_with_surrounding_text(self, sample_df):
        raw = """Here is your cleaning plan:
        {"operations": [{"type": "remove_duplicates"}]}
        Hope this helps!"""
        plan = parse_cleaning_response(raw)
        assert plan["operations"] == [{"type": "remove_duplicates"}]

    def test_parse_ai_response_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_cleaning_response("{invalid json operations}")

    def test_generate_plan_with_ai_response(self, sample_df):
        ai_resp = '{"operations": [{"type": "remove_duplicates"}]}'
        plan = generate_cleaning_plan("clean data", sample_df, ai_response=ai_resp)
        assert plan["operations"] == [{"type": "remove_duplicates"}]

    def test_generate_plan_unrecognized_returns_prompt(self, sample_df):
        result = generate_cleaning_plan("Do some complex machine learning clustering", sample_df)
        assert "prompt" in result
        assert result["status"] == "awaiting_ai_response"

    def test_build_cleaning_prompt_contains_profile(self, sample_df):
        prompt = build_cleaning_prompt("clean my data", sample_df)
        assert "clean my data" in prompt
        assert "Rows: 5" in prompt
        assert "salary" in prompt


# ---------------------------------------------------------------------------
# End-to-End Integration Tests on Real CSV Dataset
# ---------------------------------------------------------------------------

class TestEndToEndCSVIntegration:
    @pytest.fixture
    def csv_df(self):
        import os
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "test_sample.csv"
        )
        return pd.read_csv(csv_path)

    def test_full_pipeline_on_csv(self, csv_df):
        original_copy = csv_df.copy()

        # Step 1: Detect missing values before cleaning
        from operations.missing import detect_missing
        missing_info = detect_missing(csv_df)
        assert missing_info["total_missing"] > 0
        assert "salary" in missing_info["columns"]

        # Step 2: Generate plan from natural language
        request = (
            "Remove duplicate rows and strip spaces and "
            "fill missing salary with median and "
            "convert city to lowercase"
        )
        plan = generate_cleaning_plan(request, csv_df)
        assert len(plan["operations"]) == 4

        # Step 3: Apply cleaning plan
        cleaned_df, report = apply_cleaning_plan(csv_df, plan)

        # Step 4: Verify results
        assert report["rows_before"] == len(csv_df)
        assert report["rows_after"] < len(csv_df)
        assert report["duplicates_removed"] == 2
        assert cleaned_df["salary"].isnull().sum() == 0
        # city values that are present should be lowercase
        non_null_cities = cleaned_df["city"].dropna().tolist()
        assert all(c == c.lower() for c in non_null_cities)

        # Step 5: Verify original DataFrame was not mutated
        pd.testing.assert_frame_equal(csv_df, original_copy)


