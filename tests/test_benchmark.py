import numpy as np
import pandas as pd
import pytest

from risk_engine import benchmark, metrics
from risk_engine.config import TRADING_DAYS


def _legacy(prc, spyc):
    """app.py:576-584 before the extraction, verbatim math."""
    b_mu, b_sd = spyc.mean(), spyc.std()
    b_ret, b_vol = b_mu * TRADING_DAYS, b_sd * np.sqrt(TRADING_DAYS)
    b_sharpe = (b_mu / b_sd) * np.sqrt(TRADING_DAYS) if b_sd > 0 else 0.0
    b_wealth = np.exp(spyc.cumsum())
    b_dd = (b_wealth / b_wealth.cummax() - 1.0).min()
    beta = float(np.cov(prc, spyc)[0, 1] / np.var(spyc)) if np.var(spyc) > 0 else 0.0
    return b_ret, b_vol, b_sharpe, b_dd, beta


def _pr(lr):
    w = np.array([0.5, 0.3, 0.2])
    return metrics.portfolio_returns(lr, ["AAPL", "JPM", "TLT"], w)


def test_benchmark_column_matches_the_old_inline_math(market):
    _, lr = market
    pr, spy = _pr(lr), lr["SPY"]
    c = benchmark.compare(pr, spy, 100_000)
    b_ret, b_vol, b_sharpe, b_dd, _ = _legacy(pr, spy)
    got = {r["label"]: r["benchmark"] for r in c.rows}
    assert got["Annualized return"] == pytest.approx(b_ret)
    assert got["Annualized volatility (risk)"] == pytest.approx(b_vol)
    assert got["Sharpe ratio"] == pytest.approx(b_sharpe)
    assert got["Maximum drawdown"] == pytest.approx(b_dd)


def test_beta_uses_one_degrees_of_freedom_convention(market):
    # The old code divided an ddof=1 covariance by an ddof=0 variance, inflating beta by n/(n-1).
    _, lr = market
    pr, spy = _pr(lr), lr["SPY"]
    n = len(spy)
    legacy_beta = _legacy(pr, spy)[4]
    assert benchmark.beta(pr, spy) == pytest.approx(legacy_beta * (n - 1) / n)


def test_spy_against_itself_has_beta_one_and_identical_growth(market):
    _, lr = market
    spy = lr["SPY"]
    c = benchmark.compare(spy, spy, 10_000)
    assert c.beta == pytest.approx(1.0)
    assert c.growth_portfolio.iloc[-1] == pytest.approx(c.growth_benchmark.iloc[-1])
    assert c.growth_portfolio.iloc[0] == pytest.approx(10_000 * np.exp(spy.iloc[0]))


def test_rows_are_in_display_order_with_direction():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    p = pd.Series([0.01, -0.02, 0.005, 0.0], idx)
    c = benchmark.compare(p, p * 0.5, 1000)
    assert [r["label"] for r in c.rows] == ["Annualized return", "Annualized volatility (risk)",
                                            "Sharpe ratio", "Maximum drawdown"]
    assert [r["higher_is_better"] for r in c.rows] == [True, False, True, True]
    assert [r["kind"] for r in c.rows] == ["pct", "pct", "ratio", "pct"]
