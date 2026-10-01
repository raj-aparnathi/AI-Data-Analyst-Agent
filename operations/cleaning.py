"""Basic cleaning operations."""


def remove_duplicates(df):
    """Remove duplicate rows."""
    return df.drop_duplicates()


def lowercase(df, column):
    """Convert a text column to lowercase."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")
    df[column] = df[column].astype(str).str.lower()
    return df


def strip_spaces(df, column):
    """Remove extra spaces from a text column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")
    df[column] = df[column].astype(str).str.strip()
    return df
