# cell_utils.py
from typing import Any
def get_cell_like(T: Any, row: int, col: int):
    """
    Flexible access to a "cell" at (row, col) similar to MATLAB's T{row,col}.
    Accepts nested lists, numpy arrays, or pandas DataFrames.
    row, col are 0-based indexes here.
    """
    try:
        import pandas as pd
        if isinstance(T, pd.DataFrame):
            return T.iloc[row, col]
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(T, np.ndarray):
            return T[row, col]
    except Exception:
        pass
    try:
        return T[row][col]  # nested lists
    except Exception as ex:
        raise TypeError(f"Unable to index into T at ({row},{col}): {ex}")
