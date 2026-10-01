"""Encoding operations for categorical columns."""

import pandas as pd


def encode(df, column, method):
    """Encode a column using label encoding or one-hot encoding."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' was not found in the dataset.")

    if method == "label":
        # Male -> 1, Female -> 0 (alphabetical order)
        categories = sorted(df[column].dropna().unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        df[column] = df[column].map(mapping)
    elif method == "one_hot":
        # Gender -> Gender_Male, Gender_Female
        dummies = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[column])
    else:
        raise ValueError(f"Unknown encoding method '{method}'. Use label or one_hot.")

    return df
