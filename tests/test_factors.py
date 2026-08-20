import numpy as np
import pandas as pd
from risk_engine import factors as fac


def test_regression_recovers_known_betas():
    """Build a portfolio return series that IS a known linear combo of the real
    factors plus tiny noise; the regression must recover the betas."""
    f = fac.load_factors()
    true_b = np.array([1.10, -0.30, 0.45])   # Mkt, SMB, HML
    true_alpha = 0.0002
    rng = np.random.default_rng(0)
    excess = (f[fac.FACTOR_NAMES].to_numpy() @ true_b
              + true_alpha
              + rng.normal(0, 1e-5, len(f)))
    port = pd.Series(excess + f["RF"].to_numpy(), index=f.index)  # add RF back (regression removes it)
    reg = fac.factor_regression(port, f)
    assert abs(reg["betas"]["Mkt-RF"] - 1.10) < 1e-2
    assert abs(reg["betas"]["SMB"] - (-0.30)) < 1e-2
    assert abs(reg["betas"]["HML"] - 0.45) < 1e-2
    assert abs(reg["alpha_daily"] - true_alpha) < 1e-4
    assert reg["r2"] > 0.99
    assert reg["n_obs"] > 1500


def test_regression_aligns_on_common_dates():
    f = fac.load_factors()
    # portfolio covering only half the dates still regresses on the overlap
    half = f.index[: len(f) // 2]
    port = pd.Series(np.zeros(len(half)), index=half)
    reg = fac.factor_regression(port, f)
    assert reg["n_obs"] == len(half)


def test_variance_attribution_sums_to_one():
    f = fac.load_factors()
    rng = np.random.default_rng(1)
    port = pd.Series(f[fac.FACTOR_NAMES].to_numpy() @ np.array([1.0, 0.2, -0.1])
                     + f["RF"].to_numpy() + rng.normal(0, 1e-4, len(f)), index=f.index)
    reg = fac.factor_regression(port, f)
    attr = fac.variance_attribution(reg)
    assert set(attr) == {"Mkt-RF", "SMB", "HML", "Idiosyncratic"}
    assert abs(sum(attr.values()) - 1.0) < 1e-9


def test_near_pure_factor_portfolio_is_mostly_systematic():
    """Tiny residual noise → idiosyncratic fraction is small, market dominates."""
    f = fac.load_factors()
    rng = np.random.default_rng(2)
    port = pd.Series(f[fac.FACTOR_NAMES].to_numpy() @ np.array([1.0, 0.0, 0.0])
                     + f["RF"].to_numpy() + rng.normal(0, 1e-6, len(f)), index=f.index)
    reg = fac.factor_regression(port, f)
    attr = fac.variance_attribution(reg)
    assert attr["Idiosyncratic"] < 0.05
    assert attr["Mkt-RF"] > 0.8
