import numpy as np
from scipy.stats import norm
from risk_engine import metrics
from tests.conftest import HOLDINGS, CONF


def test_historical_var_of_normal_matches_theory(normal_returns):
    var, cvar = metrics.historical_var_cvar(normal_returns, 0.95)
    # 95% VaR of N(0, 0.01) ≈ -1.645 * 0.01
    assert abs(var - (-1.645 * 0.01)) < 5e-4
    assert cvar < var  # expected shortfall is worse than the threshold


def test_gaussian_var_matches_closed_form(normal_returns):
    var, _ = metrics.gaussian_var_cvar(normal_returns, 0.95)
    mu, sd = normal_returns.mean(), normal_returns.std()
    assert abs(var - (mu + sd * norm.ppf(0.05))) < 1e-9


def test_metrics_reproduce_golden(market, ref_weights, golden):
    _, lr = market
    pr = metrics.portfolio_returns(lr, HOLDINGS, ref_weights)
    h_var, h_cvar = metrics.historical_var_cvar(pr, CONF)
    g_var, g_cvar = metrics.gaussian_var_cvar(pr, CONF)
    assert abs(h_var - golden["hist_var"]) < 1e-9
    assert abs(h_cvar - golden["hist_cvar"]) < 1e-9
    assert abs(g_var - golden["gauss_var"]) < 1e-9
    assert abs(g_cvar - golden["gauss_cvar"]) < 1e-9
    assert abs(metrics.annualized_vol(pr) - golden["ann_vol"]) < 1e-9
    assert abs(metrics.sharpe_ratio(pr) - golden["sharpe"]) < 1e-9
    assert abs(metrics.max_drawdown(pr) - golden["max_dd"]) < 1e-9
