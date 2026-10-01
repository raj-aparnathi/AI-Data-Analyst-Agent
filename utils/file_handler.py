"""Simple helpers for reading uploaded files."""

import pandas as pd


def read_file(uploaded_file):
    """Read a CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    raise ValueError("Please upload a CSV or Excel file.")
