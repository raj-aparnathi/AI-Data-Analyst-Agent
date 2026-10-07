"""
tests/test_encoding.py - Tests for M3 Encoding Operations
==========================================================
Tests for:
    - label_encode
    - one_hot_encode
    - invalid column handling
    - missing value handling
"""

import numpy as np
import pandas as pd
import pytest

from operations.encoding import label_encode, one_hot_encode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gender_df():
    """Simple 5-row DataFrame with a Gender column."""
    return pd.DataFrame({
        "Gender": ["Male", "Female", "Male", "Female", "Male"],
        "Age": [25, 30, 22, 28, 35],
    })


@pytest.fixture
def gender_df_with_nan():
    """DataFrame with a missing value in Gender."""
    return pd.DataFrame({
        "Gender": ["Male", None, "Male", "Female", None],
        "Age": [25, 30, 22, 28, 35],
    })


@pytest.fixture
def city_df():
    """DataFrame with a City column (3 unique values)."""
    return pd.DataFrame({
        "City": ["NYC", "LA", "NYC", "Chicago", "LA"],
        "Score": [10, 20, 30, 40, 50],
    })


# ===========================================================================
# label_encode tests
# ===========================================================================

class TestLabelEncode:

    def test_output_exists(self, gender_df):
        df_out, metadata = label_encode(gender_df, "Gender")
        assert df_out is not None
        assert metadata is not None

    def test_values_are_encoded(self, gender_df):
        df_out, metadata = label_encode(gender_df, "Gender")
        assert pd.api.types.is_numeric_dtype(df_out["Gender"])
        assert set(df_out["Gender"].dropna().unique()).issubset({0, 1})

    def test_row_count_unchanged(self, gender_df):
        df_out, _ = label_encode(gender_df, "Gender")
        assert len(df_out) == len(gender_df)

    def test_other_columns_unchanged(self, gender_df):
        df_out, _ = label_encode(gender_df, "Gender")
        pd.testing.assert_series_equal(df_out["Age"], gender_df["Age"])

    def test_metadata_contains_mapping(self, gender_df):
        _, metadata = label_encode(gender_df, "Gender")
        assert "mapping" in metadata
        assert isinstance(metadata["mapping"], dict)
        assert "Female" in metadata["mapping"]
        assert "Male" in metadata["mapping"]

    def test_mapping_is_correct(self, gender_df):
        _, metadata = label_encode(gender_df, "Gender")
        # LabelEncoder sorts alphabetically: Female=0, Male=1
        assert metadata["mapping"]["Female"] == 0
        assert metadata["mapping"]["Male"] == 1

    def test_original_df_not_modified(self, gender_df):
        original_values = gender_df["Gender"].tolist()
        label_encode(gender_df, "Gender")
        assert gender_df["Gender"].tolist() == original_values

    def test_missing_values_preserved(self, gender_df_with_nan):
        df_out, _ = label_encode(gender_df_with_nan, "Gender")
        nan_positions = gender_df_with_nan["Gender"].isna()
        assert df_out["Gender"][nan_positions].isna().all()

    def test_missing_values_not_corrupted(self, gender_df_with_nan):
        df_out, _ = label_encode(gender_df_with_nan, "Gender")
        # Non-null rows must be numeric
        non_null = df_out["Gender"].dropna()
        assert len(non_null) > 0

    def test_row_count_unchanged_with_nan(self, gender_df_with_nan):
        df_out, _ = label_encode(gender_df_with_nan, "Gender")
        assert len(df_out) == len(gender_df_with_nan)


class TestLabelEncodeInvalidColumn:

    def test_raises_for_missing_column(self, gender_df):
        with pytest.raises(ValueError, match="does not exist"):
            label_encode(gender_df, "NonExistentColumn")

    def test_error_message_mentions_column(self, gender_df):
        with pytest.raises(ValueError, match="NonExistentColumn"):
            label_encode(gender_df, "NonExistentColumn")


# ===========================================================================
# one_hot_encode tests
# ===========================================================================

class TestOneHotEncode:

    def test_expected_columns_created(self, gender_df):
        df_out = one_hot_encode(gender_df, "Gender")
        assert "Gender_Female" in df_out.columns
        assert "Gender_Male" in df_out.columns

    def test_original_column_removed(self, gender_df):
        df_out = one_hot_encode(gender_df, "Gender")
        assert "Gender" not in df_out.columns

    def test_row_count_unchanged(self, gender_df):
        df_out = one_hot_encode(gender_df, "Gender")
        assert len(df_out) == len(gender_df)

    def test_values_are_binary(self, gender_df):
        df_out = one_hot_encode(gender_df, "Gender")
        for col in ["Gender_Female", "Gender_Male"]:
            assert set(df_out[col].unique()).issubset({0, 1})

    def test_three_categories(self, city_df):
        df_out = one_hot_encode(city_df, "City")
        assert "City_NYC" in df_out.columns
        assert "City_LA" in df_out.columns
        assert "City_Chicago" in df_out.columns
        assert "City" not in df_out.columns

    def test_row_count_unchanged_three_cats(self, city_df):
        df_out = one_hot_encode(city_df, "City")
        assert len(df_out) == len(city_df)

    def test_other_columns_preserved(self, gender_df):
        df_out = one_hot_encode(gender_df, "Gender")
        assert "Age" in df_out.columns
        pd.testing.assert_series_equal(df_out["Age"], gender_df["Age"])

    def test_original_df_not_modified(self, gender_df):
        original_cols = list(gender_df.columns)
        one_hot_encode(gender_df, "Gender")
        assert list(gender_df.columns) == original_cols

    def test_nan_produces_all_zero_row(self, gender_df_with_nan):
        df_out = one_hot_encode(gender_df_with_nan, "Gender")
        nan_pos = gender_df_with_nan["Gender"].isna()
        for idx in df_out[nan_pos].index:
            row = df_out.loc[idx, [c for c in df_out.columns if c.startswith("Gender_")]]
            assert (row == 0).all(), "NaN row should produce all-zero indicators"


class TestOneHotEncodeInvalidColumn:

    def test_raises_for_missing_column(self, gender_df):
        with pytest.raises(ValueError, match="does not exist"):
            one_hot_encode(gender_df, "NonExistentColumn")

    def test_error_message_mentions_column(self, gender_df):
        with pytest.raises(ValueError, match="NonExistentColumn"):
            one_hot_encode(gender_df, "NonExistentColumn")
