import numpy as np
from risk_engine import optimize
from tests.conftest import HOLDINGS


def test_ledoit_wolf_is_symmetric_and_psd(market):
    _, lr = market
    cov = optimize.ledoit_wolf_cov(lr, HOLDINGS)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10   # PSD


def test_min_variance_not_worse_than_equal_weight(market):
    _, lr = market
    cov = optimize.sample_cov(lr, HOLDINGS)
    w_mv = optimize.min_variance_weights(cov)
    eq = np.ones(len(HOLDINGS)) / len(HOLDINGS)
    v_mv = float(np.sqrt(w_mv @ cov @ w_mv))
    v_eq = float(np.sqrt(eq @ cov @ eq))
    assert v_mv <= v_eq + 1e-9
    assert abs(w_mv.sum() - 1.0) < 1e-6
    assert (w_mv >= -1e-9).all()


def test_risk_contribution_sums_to_one(market, ref_weights):
    _, lr = market
    cov = optimize.sample_cov(lr, HOLDINGS)
    rc = optimize.risk_contribution(cov, ref_weights)
    assert abs(rc.sum() - 1.0) < 1e-9


def test_optimizer_reproduces_golden(market, ref_weights, golden):
    _, lr = market
    mu_v = optimize.annualized_mean(lr, HOLDINGS)
    cov = optimize.sample_cov(lr, HOLDINGS)
    w_ms = optimize.max_sharpe_weights(mu_v, cov)
    w_mv = optimize.min_variance_weights(cov)
    assert abs(optimize.perf(w_ms, mu_v, cov)[2] - golden["max_sharpe"]) < 1e-4
    assert abs(optimize.perf(w_mv, mu_v, cov)[2] - golden["min_var_sharpe"]) < 1e-4
    rc = optimize.risk_contribution(cov, ref_weights)
    assert np.allclose(rc, golden["risk_pct"], atol=1e-9)
