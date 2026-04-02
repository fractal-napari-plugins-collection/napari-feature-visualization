"""Shared utility functions for napari-feature-visualization."""

from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=16)
def get_df(path) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame, cached by path.

    The result is memoised so that repeated calls with the same path (e.g.
    during live-update mode) avoid redundant disk I/O.

    Note: the cache is keyed on the path object only — if the underlying file
    is modified between calls, the cached (stale) DataFrame is returned.
    To force a reload, clear the cache explicitly: ``get_df.cache_clear()``.
    """
    return pd.read_csv(path)
