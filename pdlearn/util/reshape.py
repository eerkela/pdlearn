from __future__ import annotations
from typing import Any, Union

import numpy as np
import pandas as pd


column_specifier = Union[
    pd.Series,
    pd.DataFrame,
    list,
    tuple,
    pd.Index,
    np.ndarray,
    Any
]


def extract_columns(df: pd.DataFrame, cols: column_specifier) -> pd.DataFrame:
    """Extract columns from a ``pandas.DataFrame`` for fitting."""
    # DataFrame
    if isinstance(cols, pd.DataFrame):
        if not df[cols.columns].equals(cols):
            raise ValueError(f"DataFrame columns do not match parent:\n{cols}")

    # Series
    elif isinstance(cols, pd.Series):
        if cols.name is None:
           raise ValueError(
               f"Series specifier must have a '.name' attribute:\n{cols}"
            )
        if not df[cols.name].equals(cols):
            raise ValueError("Series does not match parent")
        cols = pd.DataFrame(cols)

    # sequence
    elif isinstance(cols, (list, tuple, pd.Index, np.ndarray)):
        cols = df[cols]

    # scalar
    else:
        cols = df[[cols]]

    return cols


def split_train_test(
    df: pd.DataFrame,
    train_size: float = 0.67,
    seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Split a DataFrame into separate train and test sets."""
    if not 0.0 < train_size <= 1.0:
        raise ValueError(
            f"'train_size' must be between 0 and 1, not {train_size}"
        )

    if train_size == 1.0:
        train = df
        test = None
    else:
        np.random.seed(seed)
        mask = np.random.rand(df.shape[0]) < train_size
        train = df[mask]
        test = df[~mask]

    return train, test
