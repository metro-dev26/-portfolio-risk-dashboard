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


def test_variance_attribution_allows_negative_factor_fraction():
    """A factor's fraction is NOT clamped to [0, 1] — it can go negative when its
    beta has the opposite sign to the covariance-weighted contribution of the
    other factors. Here b_Mkt is small and negative while SMB carries a large
    positive beta and Mkt-SMB covariance is positive, so the cross-covariance
    term dominates and flips Mkt-RF's own contribution negative. Verified
    empirically against the real factors.csv (see task report)."""
    f = fac.load_factors()
    b = np.array([-0.2, 2.3, -0.8])  # Mkt, SMB, HML
    rng = np.random.default_rng(3)
    port = pd.Series(f[fac.FACTOR_NAMES].to_numpy() @ b
                     + f["RF"].to_numpy() + rng.normal(0, 1e-5, len(f)), index=f.index)
    reg = fac.factor_regression(port, f)
    attr = fac.variance_attribution(reg)
    assert attr["Mkt-RF"] < -0.01
    assert abs(sum(attr.values()) - 1.0) < 1e-9


def test_variance_attribution_zero_total_fallback():
    """total <= 0 (no systematic, no idiosyncratic variance) must hit the guard
    branch: every factor fraction is exactly 0.0 and Idiosyncratic is 1.0."""
    reg = {
        "beta_vec": np.zeros(3),
        "factor_cov": np.eye(3),
        "resid_var_daily": 0.0,
    }
    attr = fac.variance_attribution(reg)
    assert attr == {"Mkt-RF": 0.0, "SMB": 0.0, "HML": 0.0, "Idiosyncratic": 1.0}
