"""
Pandas-based preprocessing utilities for CICIDS2017 data preparation.
"""

import re

import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase snake_case."""
    renamed = {}
    for col_name in df.columns:
        new_name = (
            str(col_name).strip()
            .lower()
            .replace(" ", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )
        new_name = re.sub(r"_+", "_", new_name)
        renamed[col_name] = new_name
    return df.rename(columns=renamed)


def handle_infinity_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Infinity values with NaN."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df


def align_schema(df: pd.DataFrame, ref_columns: list[str]) -> pd.DataFrame:
    """Align DataFrame schema by adding missing columns as NaN."""
    aligned = df.copy()
    for column in ref_columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[ref_columns]


def load_csv_file(file_path: str) -> pd.DataFrame:
    """Read a single CICIDS2017 CSV file and apply basic cleaning."""
    df = pd.read_csv(file_path, low_memory=False)
    df = clean_column_names(df)
    return handle_infinity_values(df)
