import os
import numpy as np
import pandas as pd

def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return None
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")


