"""
tests/test_file_handler.py — Unit Tests for file_handler & dataset_profiler (M1)
=================================================================================
Uses pytest and creates temporary files to test the full load → profile pipeline.

Run with:
    pytest tests/test_file_handler.py -v

Author : Member 1 (Dataset Understanding & File Handling)
"""

import io
import os
import json
import tempfile

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# We avoid importing streamlit at module level so the tests can run without
# a Streamlit runtime.  Instead we mock st.error where needed.
# ---------------------------------------------------------------------------
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper: create an in-memory file object that mimics st.UploadedFile
# ---------------------------------------------------------------------------

class FakeUploadedFile(io.BytesIO):
    """
    Minimal stand-in for streamlit.runtime.uploaded_file_manager.UploadedFile.
    Only the `.name` and `.size` attributes are needed by file_handler.
    """

    def __init__(self, name: str, data: bytes):
        super().__init__(data)
        self.name = name
        self.size = len(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_bytes() -> bytes:
    """Return bytes for a small CSV."""
    return b"Name,Age,City\nAlice,30,London\nBob,25,Paris\nCharlie,35,Berlin\n"


@pytest.fixture
def sample_json_bytes() -> bytes:
    """Return bytes for a small JSON dataset."""
    data = [
        {"Name": "Alice", "Age": 30, "City": "London"},
        {"Name": "Bob", "Age": 25, "City": "Paris"},
    ]
    return json.dumps(data).encode("utf-8")


@pytest.fixture
def sample_txt_tab_bytes() -> bytes:
    """Return bytes for a tab-separated TXT file."""
    return b"Name\tAge\tCity\nAlice\t30\tLondon\nBob\t25\tParis\n"


@pytest.fixture
def sample_txt_comma_bytes() -> bytes:
    """Return bytes for a comma-separated TXT file."""
    return b"Name,Age,City\nAlice,30,London\nBob,25,Paris\n"


@pytest.fixture
def sample_excel_bytes() -> bytes:
    """Create a tiny Excel file in-memory and return its bytes."""
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [30, 25]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════
# Tests for utils/file_handler.py
# ═══════════════════════════════════════════════════════════════════════════

class TestGetFileExtension:
    """Tests for get_file_extension()."""

    def test_csv(self):
        from utils.file_handler import get_file_extension
        f = FakeUploadedFile("data.csv", b"")
        assert get_file_extension(f) == ".csv"

    def test_uppercase(self):
        from utils.file_handler import get_file_extension
        f = FakeUploadedFile("DATA.CSV", b"")
        assert get_file_extension(f) == ".csv"

    def test_xlsx(self):
        from utils.file_handler import get_file_extension
        f = FakeUploadedFile("report.xlsx", b"")
        assert get_file_extension(f) == ".xlsx"

    def test_json(self):
        from utils.file_handler import get_file_extension
        f = FakeUploadedFile("records.json", b"")
        assert get_file_extension(f) == ".json"

    def test_txt(self):
        from utils.file_handler import get_file_extension
        f = FakeUploadedFile("notes.TXT", b"")
        assert get_file_extension(f) == ".txt"


class TestValidateFileType:
    """Tests for validate_file_type()."""

    def test_supported(self):
        from utils.file_handler import validate_file_type
        for ext in ["csv", "xlsx", "xls", "json", "txt"]:
            f = FakeUploadedFile(f"file.{ext}", b"")
            assert validate_file_type(f) is True

    def test_unsupported(self):
        from utils.file_handler import validate_file_type
        for ext in ["pdf", "docx", "png", "py"]:
            f = FakeUploadedFile(f"file.{ext}", b"")
            assert validate_file_type(f) is False


class TestLoadDataset:
    """Tests for load_dataset()."""

    @patch("utils.file_handler.st")
    def test_load_csv(self, mock_st, sample_csv_bytes):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("data.csv", sample_csv_bytes)
        df = load_dataset(f)
        assert df is not None
        assert list(df.columns) == ["Name", "Age", "City"]
        assert len(df) == 3

    @patch("utils.file_handler.st")
    def test_load_json(self, mock_st, sample_json_bytes):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("data.json", sample_json_bytes)
        df = load_dataset(f)
        assert df is not None
        assert len(df) == 2

    @patch("utils.file_handler.st")
    def test_load_txt_tab(self, mock_st, sample_txt_tab_bytes):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("data.txt", sample_txt_tab_bytes)
        df = load_dataset(f)
        assert df is not None
        assert df.shape[1] == 3  # 3 columns detected

    @patch("utils.file_handler.st")
    def test_load_txt_comma(self, mock_st, sample_txt_comma_bytes):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("data.txt", sample_txt_comma_bytes)
        df = load_dataset(f)
        assert df is not None
        assert df.shape[1] == 3

    @patch("utils.file_handler.st")
    def test_load_excel(self, mock_st, sample_excel_bytes):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("data.xlsx", sample_excel_bytes)
        df = load_dataset(f)
        assert df is not None
        assert list(df.columns) == ["Name", "Age"]

    @patch("utils.file_handler.st")
    def test_unsupported_extension(self, mock_st):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("image.png", b"\x89PNG")
        result = load_dataset(f)
        assert result is None
        mock_st.error.assert_called_once()

    @patch("utils.file_handler.st")
    def test_empty_csv(self, mock_st):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("empty.csv", b"")
        result = load_dataset(f)
        assert result is None
        mock_st.error.assert_called()

    @patch("utils.file_handler.st")
    def test_corrupted_file(self, mock_st):
        from utils.file_handler import load_dataset
        f = FakeUploadedFile("bad.xlsx", b"this is not valid excel content")
        result = load_dataset(f)
        assert result is None
        mock_st.error.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# Tests for utils/dataset_profiler.py
# ═══════════════════════════════════════════════════════════════════════════

class TestProfileDataset:
    """Tests for profile_dataset()."""

    def _sample_df(self) -> pd.DataFrame:
        """Create a small DataFrame with mixed types and some missing data."""
        return pd.DataFrame({
            "Customer_ID": [1, 2, 3, 4, 5],
            "Name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "Age": [30, 25, 35, None, 28],
            "Salary": [50000.0, 60000.0, None, 70000.0, 55000.0],
            "City": ["London", "Paris", "Berlin", "London", "Paris"],
        })

    def test_basic_shape(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["rows"] == 5
        assert profile["columns"] == 5

    def test_column_names(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["column_names"] == list(df.columns)

    def test_numerical_excludes_id(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        # Customer_ID should be detected as an identifier, not numerical
        assert "Customer_ID" not in profile["numerical_columns"]
        assert "Customer_ID" in profile["id_columns"]

    def test_numerical_includes_real_numbers(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert "Age" in profile["numerical_columns"]
        assert "Salary" in profile["numerical_columns"]

    def test_categorical_columns(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert "Name" in profile["categorical_columns"]
        assert "City" in profile["categorical_columns"]

    def test_missing_values(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["total_missing"] == 2
        assert "Age" in profile["missing_values"]
        assert "Salary" in profile["missing_values"]
        assert profile["missing_values"]["Age"] == 1
        assert profile["missing_values"]["Salary"] == 1

    def test_missing_percentages(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["missing_percentages"]["Age"] == 20.0   # 1/5 * 100
        assert profile["missing_percentages"]["Salary"] == 20.0

    def test_duplicate_rows_zero(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["duplicate_rows"] == 0

    def test_duplicate_rows_nonzero(self):
        from utils.dataset_profiler import profile_dataset
        df = pd.DataFrame({"A": [1, 1, 2], "B": ["x", "x", "y"]})
        profile = profile_dataset(df)
        assert profile["duplicate_rows"] == 1

    def test_data_types(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert "float64" in profile["data_types"]["Salary"]
        # Python 3.13+ with pandas may use 'str' instead of 'object'
        assert profile["data_types"]["Name"] in ("object", "str", "string")

    def test_unique_values(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert profile["unique_values"]["City"] == 3  # London, Paris, Berlin

    def test_memory_usage_is_string(self):
        from utils.dataset_profiler import profile_dataset
        df = self._sample_df()
        profile = profile_dataset(df)
        assert isinstance(profile["memory_usage"], str)


class TestGenerateDatasetDescription:
    """Tests for generate_dataset_description()."""

    def test_returns_string(self):
        from utils.dataset_profiler import generate_dataset_description
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        desc = generate_dataset_description(df)
        assert isinstance(desc, str)
        assert "3" in desc     # rows
        assert "2" in desc     # columns

    def test_no_missing(self):
        from utils.dataset_profiler import generate_dataset_description
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        desc = generate_dataset_description(df)
        assert "no missing" in desc.lower()

    def test_no_duplicates(self):
        from utils.dataset_profiler import generate_dataset_description
        df = pd.DataFrame({"A": [1, 2], "B": ["x", "y"]})
        desc = generate_dataset_description(df)
        assert "no duplicate" in desc.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Tests for utils/helpers.py
# ═══════════════════════════════════════════════════════════════════════════

class TestFormatFileSize:
    """Tests for format_file_size()."""

    def test_bytes(self):
        from utils.helpers import format_file_size
        assert format_file_size(512) == "512 B"

    def test_kilobytes(self):
        from utils.helpers import format_file_size
        assert format_file_size(1024) == "1.00 KB"

    def test_megabytes(self):
        from utils.helpers import format_file_size
        result = format_file_size(1_500_000)
        assert "MB" in result

    def test_zero(self):
        from utils.helpers import format_file_size
        assert format_file_size(0) == "0 B"

    def test_negative(self):
        from utils.helpers import format_file_size
        assert format_file_size(-100) == "0 B"


class TestSafeColumnName:
    """Tests for safe_column_name()."""

    def test_basic(self):
        from utils.helpers import safe_column_name
        assert safe_column_name("Customer ID") == "customer_id"

    def test_hyphens(self):
        from utils.helpers import safe_column_name
        assert safe_column_name("Total-Sales") == "total_sales"

    def test_extra_spaces(self):
        from utils.helpers import safe_column_name
        assert safe_column_name("  First   Name  ") == "first_name"

    def test_already_clean(self):
        from utils.helpers import safe_column_name
        assert safe_column_name("age") == "age"


class TestCalculateMissingPercentage:
    """Tests for calculate_missing_percentage()."""

    def test_normal(self):
        from utils.helpers import calculate_missing_percentage
        assert calculate_missing_percentage(25, 5000) == 0.5

    def test_zero_rows(self):
        from utils.helpers import calculate_missing_percentage
        assert calculate_missing_percentage(10, 0) == 0.0

    def test_no_missing(self):
        from utils.helpers import calculate_missing_percentage
        assert calculate_missing_percentage(0, 100) == 0.0


class TestLooksLikeIdColumn:
    """Tests for looks_like_id_column()."""

    def test_customer_id(self):
        from utils.helpers import looks_like_id_column
        s = pd.Series([1, 2, 3, 4, 5])
        assert looks_like_id_column("Customer_ID", s) is True

    def test_age_is_not_id(self):
        from utils.helpers import looks_like_id_column
        s = pd.Series([30, 25, 35, 28, 30])
        assert looks_like_id_column("Age", s) is False

    def test_all_unique_ints(self):
        from utils.helpers import looks_like_id_column
        s = pd.Series([100, 200, 300, 400, 500])
        # Name doesn't match an ID keyword, but all values are unique ints
        assert looks_like_id_column("mystery_col", s) is True

    def test_float_not_id(self):
        from utils.helpers import looks_like_id_column
        s = pd.Series([1.1, 2.2, 3.3])
        assert looks_like_id_column("mystery_col", s) is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
