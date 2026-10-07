"""
tests/test_normalization.py - Tests for M3 Normalization Operations
====================================================================
Tests for:
    - min_max_normalize
    - z_score_normalize
    - non-numerical column rejection
    - constant column (std=0) handling
"""

import numpy as np
import pandas as pd
import pytest

from operations.normalization import min_max_normalize, z_score_normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def salary_df():
    return pd.DataFrame({"Salary": [10.0, 20.0, 30.0], "Name": ["A", "B", "C"]})


@pytest.fixture
def constant_df():
    return pd.DataFrame({"Value": [5.0, 5.0, 5.0]})


@pytest.fixture
def categorical_df():
    return pd.DataFrame({"Gender": ["Male", "Female", "Male"]})


# ===========================================================================
# min_max_normalize tests
# ===========================================================================

class TestMinMaxNormalize:

    def test_basic_scaling(self, salary_df):
        df_out = min_max_normalize(salary_df, "Salary")
        assert df_out["Salary"].min() == pytest.approx(0.0)
        assert df_out["Salary"].max() == pytest.approx(1.0)

    def test_midpoint_value(self, salary_df):
        df_out = min_max_normalize(salary_df, "Salary")
        # Salary 20 is midpoint of [10, 30] -> 0.5
        assert df_out["Salary"].iloc[1] == pytest.approx(0.5)

    def test_row_count_unchanged(self, salary_df):
        df_out = min_max_normalize(salary_df, "Salary")
        assert len(df_out) == len(salary_df)

    def test_other_columns_unchanged(self, salary_df):
        df_out = min_max_normalize(salary_df, "Salary")
        pd.testing.assert_series_equal(df_out["Name"], salary_df["Name"])

    def test_original_df_not_modified(self, salary_df):
        original_vals = salary_df["Salary"].tolist()
        min_max_normalize(salary_df, "Salary")
        assert salary_df["Salary"].tolist() == original_vals

    def test_constant_column_returns_zeros(self, constant_df):
        df_out = min_max_normalize(constant_df, "Value")
        assert (df_out["Value"] == 0.0).all()

    def test_single_row(self):
        df = pd.DataFrame({"X": [42.0]})
        df_out = min_max_normalize(df, "X")
        assert df_out["X"].iloc[0] == pytest.approx(0.0)


class TestMinMaxNormalizeErrors:

    def test_non_numerical_column(self, categorical_df):
        with pytest.raises(ValueError, match="not numerical"):
            min_max_normalize(categorical_df, "Gender")

    def test_column_not_found(self, salary_df):
        with pytest.raises(ValueError, match="does not exist"):
            min_max_normalize(salary_df, "NonExistent")


# ===========================================================================
# z_score_normalize tests
# ===========================================================================

class TestZScoreNormalize:

    def test_mean_is_zero(self, salary_df):
        df_out = z_score_normalize(salary_df, "Salary")
        assert df_out["Salary"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_std_is_one(self, salary_df):
        df_out = z_score_normalize(salary_df, "Salary")
        assert df_out["Salary"].std() == pytest.approx(1.0, abs=1e-10)

    def test_known_values(self, salary_df):
        # Salary [10, 20, 30]: mean=20, std=10 (population) -> [-1, 0, 1]
        # pandas std uses ddof=1 by default: std=10 -> same result here
        df_out = z_score_normalize(salary_df, "Salary")
        assert df_out["Salary"].iloc[1] == pytest.approx(0.0, abs=1e-10)

    def test_row_count_unchanged(self, salary_df):
        df_out = z_score_normalize(salary_df, "Salary")
        assert len(df_out) == len(salary_df)

    def test_other_columns_unchanged(self, salary_df):
        df_out = z_score_normalize(salary_df, "Salary")
        pd.testing.assert_series_equal(df_out["Name"], salary_df["Name"])

    def test_original_df_not_modified(self, salary_df):
        original_vals = salary_df["Salary"].tolist()
        z_score_normalize(salary_df, "Salary")
        assert salary_df["Salary"].tolist() == original_vals

    def test_constant_column_returns_zeros_no_crash(self, constant_df):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df_out = z_score_normalize(constant_df, "Value")
            # Should issue a UserWarning about constant column
            assert any(issubclass(x.category, UserWarning) for x in w)
        assert (df_out["Value"] == 0.0).all()

    def test_constant_column_no_zero_division_exception(self, constant_df):
        # Must NOT raise ZeroDivisionError
        try:
            z_score_normalize(constant_df, "Value")
        except ZeroDivisionError:
            pytest.fail("z_score_normalize raised ZeroDivisionError for constant column")


class TestZScoreNormalizeErrors:

    def test_non_numerical_column(self, categorical_df):
        with pytest.raises(ValueError, match="not numerical"):
            z_score_normalize(categorical_df, "Gender")

    def test_column_not_found(self, salary_df):
        with pytest.raises(ValueError, match="does not exist"):
            z_score_normalize(salary_df, "NonExistent")
