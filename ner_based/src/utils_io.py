# src/utils_io.py
"""I/O helpers for reading and writing pipeline artefacts."""

from pathlib import Path
import pandas as pd


def ensure_dir(d: Path) -> None:
    """Create *d* and all parents if they do not exist."""
    d.mkdir(parents=True, exist_ok=True)


def read_parquet(p: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(p)


def write_parquet(df: pd.DataFrame, p: Path) -> None:
    """
    Write *df* to a Parquet file at *p*.

    Converts all-NA object columns to string type to satisfy PyArrow's
    strict schema inference.
    """
    ensure_dir(p.parent)
    for col in df.columns:
        if df[col].dtype == "object" and df[col].isna().all():
            df[col] = df[col].astype("string")
    df.to_parquet(p, index=False, engine="pyarrow")


def write_csv(df: pd.DataFrame, p: Path) -> None:
    """Write *df* to a UTF-8 CSV file at *p*."""
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8")
