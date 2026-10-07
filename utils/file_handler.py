"""
utils/file_handler.py — Safe File Loading (M1)
===============================================
Accepts a user-uploaded file (via Streamlit's UploadedFile),
validates the extension, and reads the data into a Pandas DataFrame.

Supported formats: CSV, Excel (.xlsx / .xls), JSON, TXT.

Author : Member 1 (Dataset Understanding & File Handling)
"""

import os
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All file extensions this module can handle.
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".txt"}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_file_extension(file) -> str:
    """
    Extract the lowercased file extension from an uploaded file object.

    Parameters
    ----------
    file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The file object returned by st.file_uploader().

    Returns
    -------
    str
        The extension including the leading dot, e.g. ".csv".
    """
    # Handle both filepath strings and UploadedFile objects (which have .name)
    filename = file if isinstance(file, str) else getattr(file, "name", "")
    _, ext = os.path.splitext(filename)
    return ext.lower()


def validate_file_type(file) -> bool:
    """
    Check whether the uploaded file has a supported extension.

    Parameters
    ----------
    file : UploadedFile
        The Streamlit uploaded-file object.

    Returns
    -------
    bool
        True if the extension is in SUPPORTED_EXTENSIONS.
    """
    return get_file_extension(file) in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_dataset(file) -> pd.DataFrame:
    """
    Read an uploaded file into a Pandas DataFrame.

    Workflow
    --------
    1. Detect the file extension.
    2. Validate that the format is supported.
    3. Attempt to read the file using the appropriate Pandas reader.
    4. Verify the result is non-empty.
    5. Return the DataFrame, **or** return ``None`` and display a
       user-friendly error via ``st.error()`` so the Streamlit app
       never crashes.

    Parameters
    ----------
    file : UploadedFile
        The Streamlit uploaded-file object.

    Returns
    -------
    pd.DataFrame or None
        The loaded DataFrame, or None if loading failed.

    Supported Formats
    -----------------
    .csv   — Comma-separated values (read_csv)
    .xlsx  — Excel workbook        (read_excel)
    .xls   — Legacy Excel          (read_excel)
    .json  — JSON                  (read_json)
    .txt   — Delimited text        (read_csv with auto-detected separator)
    """

    # ── Step 1: Validate extension ────────────────────────────────────────
    extension = get_file_extension(file)

    if extension not in SUPPORTED_EXTENSIONS:
        st.error(
            f"❌ Unsupported file type: **{extension}**\n\n"
            f"Please upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        return None

    # ── Step 2: Try to read the file ──────────────────────────────────────
    try:
        df = _read_by_extension(file, extension)

    except pd.errors.EmptyDataError:
        st.error("❌ The uploaded file is **empty**. Please upload a file that contains data.")
        return None

    except ValueError as ve:
        st.error(f"❌ Value error while reading the file: {ve}")
        return None

    except Exception as exc:
        # Catch-all for corrupted files, encoding problems, etc.
        st.error(
            f"❌ Could not read the file.\n\n"
            f"**Error:** {exc}\n\n"
            f"Make sure the file is not corrupted and matches the "
            f"**{extension}** format."
        )
        return None

    # ── Step 3: Verify we actually got data ───────────────────────────────
    if df is None or df.empty:
        st.error("❌ The file was read successfully but contains **no data** (0 rows).")
        return None

    return df


# ---------------------------------------------------------------------------
# Private readers
# ---------------------------------------------------------------------------

def _read_by_extension(file, extension: str) -> pd.DataFrame:
    """
    Dispatch to the correct Pandas reader based on the file extension.

    For TXT files the separator is auto-detected: we first try tab (\\t),
    then fall back to comma, then to the Python engine's built-in sniffer.
    """

    if extension == ".csv":
        return pd.read_csv(file)

    if extension in (".xlsx", ".xls"):
        return pd.read_excel(file)

    if extension == ".json":
        return pd.read_json(file)

    if extension == ".txt":
        return _read_txt(file)

    # Should never reach here because of the earlier validation.
    raise ValueError(f"Unsupported file extension: {extension}")


def _read_txt(file) -> pd.DataFrame:
    """
    Read a TXT file that may be tab-separated or comma-separated.

    Strategy
    --------
    1. Try tab-separated first  (most common for .txt exports).
    2. If that yields only one column, try comma-separated.
    3. If that also yields only one column, fall back to the Python
       engine with ``sep=None`` which uses csv.Sniffer to auto-detect
       the delimiter.
    """

    # Try tab-separated
    df = pd.read_csv(file, sep="\t")
    if df.shape[1] > 1:
        return df

    # Reset the file pointer if it's a file-like object so we can re-read
    if hasattr(file, "seek"):
        file.seek(0)

    # Try comma-separated
    df = pd.read_csv(file, sep=",")
    if df.shape[1] > 1:
        return df

    # Reset again
    if hasattr(file, "seek"):
        file.seek(0)

    # Let Python's csv.Sniffer auto-detect the delimiter
    df = pd.read_csv(file, sep=None, engine="python")
    return df


if __name__ == "__main__":
    import io

    print("=" * 60)
    print("Testing utils/file_handler.py independently")
    print("=" * 60)

    class DummyFile(io.BytesIO):
        def __init__(self, name: str, data: bytes):
            super().__init__(data)
            self.name = name

    # 1. Test get_file_extension and validate_file_type
    print("\n1. Testing extension validation:")
    test_files = ["data.csv", "report.xlsx", "dump.json", "log.txt", "doc.pdf"]
    for fname in test_files:
        f = DummyFile(fname, b"")
        ext = get_file_extension(f)
        valid = validate_file_type(f)
        print(f"   {fname:<12} -> Ext: {ext:<6} Valid: {valid}")

    # 2. Test reading CSV
    print("\n2. Testing load_dataset with in-memory CSV:")
    csv_bytes = b"Name,Age,Salary\nAlice,30,50000\nBob,25,60000\nCharlie,35,70000\n"
    f_csv = DummyFile("test_data.csv", csv_bytes)
    df_csv = load_dataset(f_csv)
    print(f"   Loaded shape: {df_csv.shape}")
    print(f"   Columns: {list(df_csv.columns)}")

    # 3. Test reading JSON
    print("\n3. Testing load_dataset with in-memory JSON:")
    json_bytes = b'[{"Product": "A", "Sales": 100}, {"Product": "B", "Sales": 150}]'
    f_json = DummyFile("test_data.json", json_bytes)
    df_json = load_dataset(f_json)
    print(f"   Loaded shape: {df_json.shape}")
    print(f"   Columns: {list(df_json.columns)}")

    # 4. Test reading TXT (tab-delimited)
    print("\n4. Testing load_dataset with tab-delimited TXT:")
    txt_bytes = b"Col1\tCol2\nVal1\tVal2\n"
    f_txt = DummyFile("test_data.txt", txt_bytes)
    df_txt = load_dataset(f_txt)
    print(f"   Loaded shape: {df_txt.shape}")
    print(f"   Columns: {list(df_txt.columns)}")

    print("\n[SUCCESS] utils/file_handler.py passed standalone checks.")
