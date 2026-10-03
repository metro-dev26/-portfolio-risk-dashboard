"""Fama-French 3-factor loading and analysis. Pure — no UI.
Factors are stored as decimals (Mkt-RF, SMB, HML, RF are simple daily returns)."""
import numpy as np
import pandas as pd

from risk_engine.config import TRADING_DAYS

FACTOR_NAMES = ["Mkt-RF", "SMB", "HML"]


def load_factors():
    """Return the factor DataFrame (DatetimeIndex; columns Mkt-RF, SMB, HML, RF; decimals)."""
    from risk_engine.data import load_snapshot
    return load_snapshot().factors


def factor_regression(port_returns, factors):
    """OLS of excess portfolio return (r_p - RF) on the 3 factors, via lstsq."""
    common = port_returns.index.intersection(factors.index)
    y = port_returns.loc[common].to_numpy() - factors.loc[common, "RF"].to_numpy()
    X = factors.loc[common, FACTOR_NAMES].to_numpy()
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    alpha, beta_vec = float(coef[0]), coef[1:]
    resid = y - design @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "alpha_daily": alpha,
        "alpha_annual": alpha * TRADING_DAYS,
        "betas": {n: float(b) for n, b in zip(FACTOR_NAMES, beta_vec)},
        "beta_vec": beta_vec,
        "r2": r2,
        "resid_var_daily": ss_res / len(y),
        "factor_cov": np.cov(X, rowvar=False),
        "n_obs": int(len(common)),
    }


def variance_attribution(reg):
    """Split portfolio variance into per-factor + idiosyncratic fractions (sum = 1).

    Systematic variance = bᵀ Σ_f b; each factor's component is b_i · (Σ_f b)_i,
    which sums to the systematic total. Residual variance is idiosyncratic.
    """
    b = reg["beta_vec"]
    cov = reg["factor_cov"]
    resid_var = reg["resid_var_daily"]
    cov_b = cov @ b
    per_factor = b * cov_b
    systematic = float(b @ cov_b)
    total = systematic + resid_var
    if total <= 0:
        out = {n: 0.0 for n in FACTOR_NAMES}
        out["Idiosyncratic"] = 1.0
        return out
    out = {n: float(pf / total) for n, pf in zip(FACTOR_NAMES, per_factor)}
    out["Idiosyncratic"] = float(resid_var / total)
    return out
