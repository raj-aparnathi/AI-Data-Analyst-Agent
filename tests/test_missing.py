"""
Tests for operations/missing.py (M2)

Covers: detect_missing, drop_missing_rows, drop_missing_columns,
fill_missing_mean, fill_missing_median, fill_missing_mode,
fill_missing_forward, fill_missing_backward, and error handling.
"""

import numpy as np
import pandas as pd
import pytest

from operations.missing import (
    detect_missing,
    drop_missing_columns,
    drop_missing_rows,
    fill_missing_backward,
    fill_missing_forward,
    fill_missing_mean,
    fill_missing_median,
    fill_missing_mode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame with a mix of numerical and categorical data + NaNs."""
    return pd.DataFrame({
        "name": ["Raj", "Priya", None, "Amit", "Raj"],
        "city": ["Ahmedabad", "Surat", "Ahmedabad", None, "Surat"],
        "salary": [45000.0, 50000.0, 52000.0, np.nan, 48000.0],
        "age": [25, np.nan, 30, 28, 25],
    })


@pytest.fixture
def all_missing_col_df():
    """DataFrame where one column is entirely NaN."""
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def empty_df():
    """Empty DataFrame."""
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# detect_missing
# ---------------------------------------------------------------------------

class TestDetectMissing:
    def test_detects_counts(self, sample_df):
        result = detect_missing(sample_df)
        assert result["total_missing"] == 4  # name=1, city=1, salary=1, age=1
        # Check specific columns
        assert "salary" in result["columns"]
        assert result["columns"]["salary"]["count"] == 1

    def test_percentages(self, sample_df):
        result = detect_missing(sample_df)
        # salary has 1 missing out of 5 → 20%
        assert result["columns"]["salary"]["percentage"] == 20.0

    def test_empty_dataframe(self, empty_df):
        result = detect_missing(empty_df)
        assert result["total_missing"] == 0
        assert result["columns"] == {}

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = detect_missing(df)
        assert result["total_missing"] == 0
        assert result["columns"] == {}


# ---------------------------------------------------------------------------
# drop_missing_rows
# ---------------------------------------------------------------------------

class TestDropMissingRows:
    def test_drop_all_columns(self, sample_df):
        result = drop_missing_rows(sample_df)
        # Rows with any NaN removed
        assert result.isnull().sum().sum() == 0
        assert len(result) < len(sample_df)

    def test_drop_specific_columns(self, sample_df):
        result = drop_missing_rows(sample_df, columns=["salary"])
        assert result["salary"].isnull().sum() == 0
        # Other columns may still have NaN
        assert len(result) == 4  # only 1 row had missing salary

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            drop_missing_rows(sample_df, columns=["nonexistent"])

    def test_does_not_mutate_original(self, sample_df):
        original_len = len(sample_df)
        _ = drop_missing_rows(sample_df)
        assert len(sample_df) == original_len


# ---------------------------------------------------------------------------
# drop_missing_columns
# ---------------------------------------------------------------------------

class TestDropMissingColumns:
    def test_drop_columns(self, sample_df):
        result = drop_missing_columns(sample_df, columns=["salary"])
        assert "salary" not in result.columns
        assert len(result.columns) == len(sample_df.columns) - 1

    def test_drop_multiple_columns(self, sample_df):
        result = drop_missing_columns(sample_df, columns=["salary", "city"])
        assert "salary" not in result.columns
        assert "city" not in result.columns

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            drop_missing_columns(sample_df, columns=["nonexistent"])

    def test_does_not_mutate_original(self, sample_df):
        original_cols = list(sample_df.columns)
        _ = drop_missing_columns(sample_df, columns=["salary"])
        assert list(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# fill_missing_mean
# ---------------------------------------------------------------------------

class TestFillMissingMean:
    def test_fills_with_mean(self, sample_df):
        result = fill_missing_mean(sample_df, "salary")
        assert result["salary"].isnull().sum() == 0
        expected_mean = sample_df["salary"].mean()
        # The previously-NaN value should now be the mean
        filled_value = result.loc[sample_df["salary"].isnull(), "salary"].values[0]
        assert filled_value == pytest.approx(expected_mean)

    def test_non_numeric_raises(self, sample_df):
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_mean(sample_df, "city")

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            fill_missing_mean(sample_df, "nonexistent")

    def test_does_not_mutate_original(self, sample_df):
        original_nulls = sample_df["salary"].isnull().sum()
        _ = fill_missing_mean(sample_df, "salary")
        assert sample_df["salary"].isnull().sum() == original_nulls


# ---------------------------------------------------------------------------
# fill_missing_median
# ---------------------------------------------------------------------------

class TestFillMissingMedian:
    def test_fills_with_median(self, sample_df):
        result = fill_missing_median(sample_df, "salary")
        assert result["salary"].isnull().sum() == 0
        expected_median = sample_df["salary"].median()
        filled_value = result.loc[sample_df["salary"].isnull(), "salary"].values[0]
        assert filled_value == pytest.approx(expected_median)

    def test_non_numeric_raises(self, sample_df):
        with pytest.raises(ValueError, match="not numerical"):
            fill_missing_median(sample_df, "city")

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            fill_missing_median(sample_df, "nonexistent")


# ---------------------------------------------------------------------------
# fill_missing_mode
# ---------------------------------------------------------------------------

class TestFillMissingMode:
    def test_fills_categorical_with_mode(self, sample_df):
        result = fill_missing_mode(sample_df, "city")
        assert result["city"].isnull().sum() == 0
        # Mode of city should be either "Ahmedabad" or "Surat"
        mode_value = sample_df["city"].mode().iloc[0]
        filled = result.loc[sample_df["city"].isnull(), "city"].values[0]
        assert filled == mode_value

    def test_fills_numerical_with_mode(self, sample_df):
        result = fill_missing_mode(sample_df, "age")
        assert result["age"].isnull().sum() == 0

    def test_all_nan_raises(self, all_missing_col_df):
        with pytest.raises(ValueError, match="Cannot compute mode"):
            fill_missing_mode(all_missing_col_df, "b")

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            fill_missing_mode(sample_df, "nonexistent")


# ---------------------------------------------------------------------------
# fill_missing_forward
# ---------------------------------------------------------------------------

class TestFillMissingForward:
    def test_forward_fill_column(self):
        df = pd.DataFrame({"val": [1.0, np.nan, 3.0, np.nan]})
        result = fill_missing_forward(df, "val")
        assert result["val"].tolist() == [1.0, 1.0, 3.0, 3.0]

    def test_forward_fill_all(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": ["x", None, "z"],
        })
        result = fill_missing_forward(df)
        assert result["a"].tolist() == [1.0, 1.0, 3.0]
        assert result["b"].tolist() == ["x", "x", "z"]

    def test_leading_nan_stays(self):
        """Forward fill cannot fill a NaN at the very start."""
        df = pd.DataFrame({"val": [np.nan, 2.0, np.nan]})
        result = fill_missing_forward(df, "val")
        assert pd.isna(result["val"].iloc[0])
        assert result["val"].iloc[2] == 2.0

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            fill_missing_forward(sample_df, "nonexistent")


# ---------------------------------------------------------------------------
# fill_missing_backward
# ---------------------------------------------------------------------------

class TestFillMissingBackward:
    def test_backward_fill_column(self):
        df = pd.DataFrame({"val": [np.nan, 2.0, np.nan, 4.0]})
        result = fill_missing_backward(df, "val")
        assert result["val"].tolist() == [2.0, 2.0, 4.0, 4.0]

    def test_backward_fill_all(self):
        df = pd.DataFrame({
            "a": [np.nan, 2.0, 3.0],
            "b": [None, "y", "z"],
        })
        result = fill_missing_backward(df)
        assert result["a"].tolist() == [2.0, 2.0, 3.0]
        assert result["b"].tolist() == ["y", "y", "z"]

    def test_trailing_nan_stays(self):
        """Backward fill cannot fill a NaN at the very end."""
        df = pd.DataFrame({"val": [1.0, np.nan, np.nan]})
        result = fill_missing_backward(df, "val")
        assert result["val"].iloc[0] == 1.0
        assert pd.isna(result["val"].iloc[2])

    def test_invalid_column_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            fill_missing_backward(sample_df, "nonexistent")
