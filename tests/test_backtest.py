import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as _t
from risk_engine import backtest


def _normal_series(n=6000, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, sd, n))


def test_correctly_specified_var_passes_kupiec():
    """A stationary normal series backtested with historical VaR should breach
    at ~5% and Kupiec should NOT reject."""
    r = _normal_series()
    res = backtest.rolling_var_breaches(r, conf=0.95, window=500, method="historical")
    n, x = len(res["breach"]), int(res["breach"].sum())
    rate = x / n
    assert 0.03 < rate < 0.07
    _, p = backtest.kupiec_pof(n, x, 0.95)
    assert p > 0.05   # fail to reject correct model


def test_too_tight_var_is_rejected_by_kupiec():
    """If we deliberately claim a far-too-tight VaR, breaches explode and Kupiec rejects."""
    n, x = 1000, 200   # 20% breaches vs 5% expected
    lr, p = backtest.kupiec_pof(n, x, 0.95)
    assert p < 0.01


def test_christoffersen_detects_clustering():
    """Breaches that arrive in one contiguous block violate independence."""
    b = np.zeros(1000, dtype=int)
    b[100:150] = 1   # 50 breaches, all clustered
    _, p_clustered = backtest.christoffersen_cc(b, 0.95)
    # a spread-out arrangement of the same count should look far more independent
    rng = np.random.default_rng(3)
    b2 = np.zeros(1000, dtype=int)
    b2[rng.choice(1000, size=50, replace=False)] = 1
    _, p_spread = backtest.christoffersen_cc(b2, 0.95)
    assert p_clustered < p_spread


def test_backtest_var_summary_shape():
    r = _normal_series()
    rows = backtest.backtest_var(r, conf=0.95, window=500,
                                 methods=("historical", "gaussian"))
    assert {row["method"] for row in rows} == {"historical", "gaussian"}
    for row in rows:
        assert 0.0 <= row["kupiec_p"] <= 1.0
        assert isinstance(row["passed"], bool)


def test_rolling_var_breaches_student_t():
    """Student-t method should handle heavy-tailed distributions and return valid result."""
    # Generate heavy-tailed sample from t-distribution with 4 degrees of freedom
    rng = np.random.default_rng(42)
    heavy_tailed = pd.Series(_t.rvs(4, loc=0, scale=0.01, size=1500, random_state=42))

    res = backtest.rolling_var_breaches(heavy_tailed, conf=0.95, window=250, method="student_t")

    # Verify result dict structure
    assert "dates" in res and "var" in res and "realized" in res and "breach" in res

    # Verify breach is 0/1 only
    assert set(res["breach"]) <= {0, 1}

    # Verify length consistency
    expected_len = len(heavy_tailed) - 250
    assert len(res["dates"]) == expected_len
    assert len(res["var"]) == expected_len
    assert len(res["realized"]) == expected_len
    assert len(res["breach"]) == expected_len


def test_rolling_var_breaches_invalid_method_raises():
    """Invalid method should raise ValueError."""
    r = pd.Series(np.random.randn(300))
    with pytest.raises(ValueError, match="unknown method"):
        backtest.rolling_var_breaches(r, conf=0.95, window=250, method="bogus")
