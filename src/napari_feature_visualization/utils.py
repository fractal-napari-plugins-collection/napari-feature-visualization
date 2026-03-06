# Util functions for plugins
from functools import lru_cache

import pandas as pd


@lru_cache(maxsize=16)
def get_df(path):
    return pd.read_csv(path)
