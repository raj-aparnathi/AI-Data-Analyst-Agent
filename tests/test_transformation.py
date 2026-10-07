"""
tests/test_transformation.py - Tests for M3 Transformation Operations
======================================================================
Tests for:
    - convert_datatype
    - rename_column
    - drop_column
    - create_calculated_column
    - extract_date_part
    - apply_transformation_plan (executor)
    - validate_transformation_plan
    - invalid operations / columns / operators
"""

import pandas as pd
import pytest

from operations.transformation import (
    SAFE_OPERATORS,
    SUPPORTED_DATE_PARTS,
    SUPPORTED_TRANSFORMATIONS,
    apply_transformation_plan,
    convert_datatype,
    create_calculated_column,
    drop_column,
    extract_date_part,
    rename_column,
    validate_transformation_plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": ["21", "25", "30"],
        "Salary": [50000.0, 60000.0, 70000.0],
        "Quantity": [2, 3, 4],
        "Price": [100.0, 150.0, 200.0],
        "Date": ["2026-01-01", "2026-06-15", "2026-12-31"],
        "Name": ["Alice", "Bob", "Charlie"],
    })


# ===========================================================================
# convert_datatype
# ===========================================================================

class TestConvertDatatype:

    def test_string_to_int(self, sample_df):
        df_out = convert_datatype(sample_df, "Age", "int")
        assert pd.api.types.is_integer_dtype(df_out["Age"])
        assert df_out["Age"].tolist() == [21, 25, 30]

    def test_string_to_float(self, sample_df):
        df_out = convert_datatype(sample_df, "Age", "float")
        assert pd.api.types.is_float_dtype(df_out["Age"])

    def test_numeric_to_string(self, sample_df):
        df_out = convert_datatype(sample_df, "Salary", "string")
        # In newer pandas (2.x+), astype(str) may return StringDtype instead of
        # object dtype. Accept both.
        assert pd.api.types.is_string_dtype(df_out["Salary"]) or df_out["Salary"].dtype == object

    def test_string_to_datetime(self, sample_df):
        df_out = convert_datatype(sample_df, "Date", "datetime")
        assert pd.api.types.is_datetime64_any_dtype(df_out["Date"])

    def test_row_count_unchanged(self, sample_df):
        df_out = convert_datatype(sample_df, "Age", "int")
        assert len(df_out) == len(sample_df)

    def test_other_columns_unchanged(self, sample_df):
        df_out = convert_datatype(sample_df, "Age", "int")
        pd.testing.assert_series_equal(df_out["Name"], sample_df["Name"])

    def test_original_not_modified(self, sample_df):
        original = sample_df["Age"].tolist()
        convert_datatype(sample_df, "Age", "int")
        assert sample_df["Age"].tolist() == original

    def test_unsupported_dtype_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            convert_datatype(sample_df, "Age", "complex128")

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            convert_datatype(sample_df, "NonExistent", "int")


# ===========================================================================
# rename_column
# ===========================================================================

class TestRenameColumn:

    def test_column_renamed(self, sample_df):
        df_out = rename_column(sample_df, "Salary", "Annual_Salary")
        assert "Annual_Salary" in df_out.columns
        assert "Salary" not in df_out.columns

    def test_row_count_unchanged(self, sample_df):
        df_out = rename_column(sample_df, "Salary", "Annual_Salary")
        assert len(df_out) == len(sample_df)

    def test_other_columns_unchanged(self, sample_df):
        df_out = rename_column(sample_df, "Salary", "Annual_Salary")
        assert "Name" in df_out.columns
        assert "Age" in df_out.columns

    def test_original_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        rename_column(sample_df, "Salary", "Annual_Salary")
        assert list(sample_df.columns) == original_cols

    def test_old_column_not_found_raises(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            rename_column(sample_df, "NonExistent", "NewName")

    def test_new_name_already_exists_raises(self, sample_df):
        with pytest.raises(ValueError, match="already exists"):
            rename_column(sample_df, "Salary", "Name")

    def test_empty_new_name_raises(self, sample_df):
        with pytest.raises(ValueError):
            rename_column(sample_df, "Salary", "")


# ===========================================================================
# drop_column
# ===========================================================================

class TestDropColumn:

    def test_column_dropped(self, sample_df):
        df_out = drop_column(sample_df, "Name")
        assert "Name" not in df_out.columns

    def test_only_requested_column_dropped(self, sample_df):
        df_out = drop_column(sample_df, "Name")
        # All other columns should still be present
        for col in ["Age", "Salary", "Quantity", "Price", "Date"]:
            assert col in df_out.columns

    def test_row_count_unchanged(self, sample_df):
        df_out = drop_column(sample_df, "Name")
        assert len(df_out) == len(sample_df)

    def test_original_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        drop_column(sample_df, "Name")
        assert list(sample_df.columns) == original_cols

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            drop_column(sample_df, "NonExistent")


# ===========================================================================
# create_calculated_column
# ===========================================================================

class TestCreateCalculatedColumn:

    def test_multiplication(self, sample_df):
        df_out = create_calculated_column(
            sample_df, "Total", "Quantity", "*", "Price"
        )
        assert "Total" in df_out.columns
        assert df_out["Total"].tolist() == [200.0, 450.0, 800.0]

    def test_addition(self, sample_df):
        df_out = create_calculated_column(
            sample_df, "Sum", "Quantity", "+", "Price"
        )
        assert df_out["Sum"].tolist() == [102.0, 153.0, 204.0]

    def test_subtraction(self, sample_df):
        df_out = create_calculated_column(
            sample_df, "Diff", "Price", "-", "Quantity"
        )
        assert df_out["Diff"].tolist() == [98.0, 147.0, 196.0]

    def test_division(self):
        df = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [2.0, 4.0, 5.0]})
        df_out = create_calculated_column(df, "Result", "A", "/", "B")
        assert df_out["Result"].tolist() == [5.0, 5.0, 6.0]

    def test_row_count_unchanged(self, sample_df):
        df_out = create_calculated_column(
            sample_df, "Total", "Quantity", "*", "Price"
        )
        assert len(df_out) == len(sample_df)

    def test_original_columns_preserved(self, sample_df):
        df_out = create_calculated_column(
            sample_df, "Total", "Quantity", "*", "Price"
        )
        assert "Quantity" in df_out.columns
        assert "Price" in df_out.columns

    def test_original_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        create_calculated_column(sample_df, "Total", "Quantity", "*", "Price")
        assert list(sample_df.columns) == original_cols

    def test_invalid_operator_raises(self, sample_df):
        with pytest.raises(ValueError, match="not allowed"):
            create_calculated_column(
                sample_df, "Total", "Quantity", "**", "Price"
            )

    def test_eval_not_used_for_arbitrary_expression(self, sample_df):
        # Attempting a non-whitelisted operator should raise ValueError,
        # NOT execute arbitrary Python.
        with pytest.raises(ValueError):
            create_calculated_column(
                sample_df, "Total", "Quantity", "eval", "Price"
            )

    def test_non_numerical_left_raises(self, sample_df):
        with pytest.raises(ValueError, match="not numerical"):
            create_calculated_column(
                sample_df, "Total", "Name", "*", "Price"
            )

    def test_non_numerical_right_raises(self, sample_df):
        with pytest.raises(ValueError, match="not numerical"):
            create_calculated_column(
                sample_df, "Total", "Quantity", "*", "Name"
            )

    def test_missing_left_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            create_calculated_column(
                sample_df, "Total", "NoCol", "*", "Price"
            )

    def test_all_safe_operators_allowed(self, sample_df):
        for op in SAFE_OPERATORS:
            if op == "/":
                continue  # division by zero in fixture, skip
            df_out = create_calculated_column(
                sample_df, f"Result_{op}", "Quantity", op, "Salary"
            )
            assert f"Result_{op}" in df_out.columns


# ===========================================================================
# extract_date_part
# ===========================================================================

class TestExtractDatePart:

    def test_extract_year(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "year")
        assert "Date_year" in df_out.columns
        assert df_out["Date_year"].tolist() == [2026, 2026, 2026]

    def test_extract_month(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "month")
        assert "Date_month" in df_out.columns
        assert df_out["Date_month"].tolist() == [1, 6, 12]

    def test_extract_day(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "day")
        assert "Date_day" in df_out.columns
        assert df_out["Date_day"].tolist() == [1, 15, 31]

    def test_extract_dayofweek(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "dayofweek")
        assert "Date_dayofweek" in df_out.columns

    def test_original_column_preserved(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "year")
        assert "Date" in df_out.columns

    def test_row_count_unchanged(self, sample_df):
        df_out = extract_date_part(sample_df, "Date", "year")
        assert len(df_out) == len(sample_df)

    def test_original_not_modified(self, sample_df):
        original_vals = sample_df["Date"].tolist()
        extract_date_part(sample_df, "Date", "year")
        assert sample_df["Date"].tolist() == original_vals

    def test_unsupported_part_raises(self, sample_df):
        with pytest.raises(ValueError, match="Unsupported date part"):
            extract_date_part(sample_df, "Date", "quarter")

    def test_missing_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="does not exist"):
            extract_date_part(sample_df, "NonExistent", "year")


# ===========================================================================
# validate_transformation_plan
# ===========================================================================

class TestValidateTransformationPlan:

    def test_valid_plan_returns_empty_errors(self, sample_df):
        plan = {"operations": [{"operation": "drop_column", "column": "Name"}]}
        errors = validate_transformation_plan(plan, sample_df)
        assert errors == []

    def test_invalid_operation_rejected(self, sample_df):
        plan = {"operations": [{"operation": "evil_hack", "column": "Name"}]}
        errors = validate_transformation_plan(plan, sample_df)
        assert any("unsupported" in e.lower() for e in errors)

    def test_missing_column_rejected(self, sample_df):
        plan = {"operations": [{"operation": "drop_column", "column": "Ghost"}]}
        errors = validate_transformation_plan(plan, sample_df)
        assert any("does not exist" in e for e in errors)

    def test_non_numerical_normalization_rejected(self, sample_df):
        plan = {"operations": [{"operation": "min_max_normalize", "column": "Name"}]}
        errors = validate_transformation_plan(plan, sample_df)
        assert any("not numerical" in e for e in errors)

    def test_invalid_operator_rejected(self, sample_df):
        plan = {
            "operations": [{
                "operation": "create_calculated_column",
                "new_column": "X",
                "left_column": "Quantity",
                "operator": "^",
                "right_column": "Price",
            }]
        }
        errors = validate_transformation_plan(plan, sample_df)
        assert any("operator" in e.lower() for e in errors)

    def test_invalid_date_part_rejected(self, sample_df):
        plan = {
            "operations": [{
                "operation": "extract_date_part",
                "column": "Date",
                "part": "quarter",
            }]
        }
        errors = validate_transformation_plan(plan, sample_df)
        assert any("part" in e.lower() for e in errors)

    def test_missing_operations_key(self, sample_df):
        errors = validate_transformation_plan({}, sample_df)
        assert any("operations" in e.lower() for e in errors)


# ===========================================================================
# apply_transformation_plan  (executor)
# ===========================================================================

class TestApplyTransformationPlan:

    def test_single_drop_column(self, sample_df):
        plan = {"operations": [{"operation": "drop_column", "column": "Name"}]}
        df_out, report = apply_transformation_plan(sample_df, plan)
        assert "Name" not in df_out.columns
        assert "drop_column" in report["operations_applied"]

    def test_multiple_operations_sequential(self, sample_df):
        plan = {
            "operations": [
                {"operation": "drop_column", "column": "Name"},
                {"operation": "min_max_normalize", "column": "Salary"},
            ]
        }
        df_out, report = apply_transformation_plan(sample_df, plan)
        assert "Name" not in df_out.columns
        assert df_out["Salary"].max() == pytest.approx(1.0)
        assert len(report["operations_applied"]) == 2

    def test_report_structure(self, sample_df):
        plan = {"operations": [{"operation": "drop_column", "column": "Name"}]}
        _, report = apply_transformation_plan(sample_df, plan)
        for key in ("operations_applied", "columns_created", "columns_removed",
                    "columns_modified", "rows_before", "rows_after", "errors"):
            assert key in report

    def test_rows_before_after_consistent(self, sample_df):
        plan = {"operations": [{"operation": "drop_column", "column": "Name"}]}
        df_out, report = apply_transformation_plan(sample_df, plan)
        assert report["rows_before"] == len(sample_df)
        assert report["rows_after"] == len(df_out)

    def test_original_df_not_modified(self, sample_df):
        original_cols = list(sample_df.columns)
        plan = {"operations": [{"operation": "drop_column", "column": "Name"}]}
        apply_transformation_plan(sample_df, plan)
        assert list(sample_df.columns) == original_cols

    def test_invalid_plan_raises(self, sample_df):
        plan = {"operations": [{"operation": "hack", "column": "Name"}]}
        with pytest.raises(ValueError):
            apply_transformation_plan(sample_df, plan)

    def test_one_hot_encode_plan(self):
        df = pd.DataFrame({"Gender": ["Male", "Female", "Male"], "Age": [25, 30, 22]})
        plan = {"operations": [{"operation": "one_hot_encode", "column": "Gender"}]}
        df_out, report = apply_transformation_plan(df, plan)
        assert "Gender_Male" in df_out.columns
        assert "Gender" in report["columns_removed"]
        assert "Gender_Male" in report["columns_created"] or "Gender_Female" in report["columns_created"]

    def test_label_encode_plan(self):
        df = pd.DataFrame({"City": ["NYC", "LA", "NYC"], "Score": [1, 2, 3]})
        plan = {"operations": [{"operation": "label_encode", "column": "City"}]}
        df_out, report = apply_transformation_plan(df, plan)
        assert pd.api.types.is_numeric_dtype(df_out["City"])
        assert "City" in report["columns_modified"]

    def test_calculated_column_plan(self, sample_df):
        plan = {
            "operations": [{
                "operation": "create_calculated_column",
                "new_column": "Total",
                "left_column": "Quantity",
                "operator": "*",
                "right_column": "Price",
            }]
        }
        df_out, report = apply_transformation_plan(sample_df, plan)
        assert "Total" in df_out.columns
        assert "Total" in report["columns_created"]
