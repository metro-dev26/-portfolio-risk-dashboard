"""Fama-French 3-factor loading and analysis. Pure — no UI.
Factors are stored as decimals (Mkt-RF, SMB, HML, RF are simple daily returns)."""
import os
import pandas as pd

_FACTORS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "factors.csv")
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML"]


def load_factors():
    """Return the factor DataFrame (DatetimeIndex; columns Mkt-RF, SMB, HML, RF; decimals)."""
    return pd.read_csv(_FACTORS, index_col=0, parse_dates=True)
