import numpy as np
import pytest
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


# The seven portfolio-return keys in golden.json (hist/gauss VaR and CVaR, ann_vol, sharpe,
# max_dd) use the exact daily-rebalanced log return; tools/snapshot_reference.py makes them.
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


def test_student_t_has_fatter_tail_than_gaussian_on_heavy_data():
    from scipy.stats import t as tdist
    import pandas as pd
    rng = np.random.default_rng(1)
    heavy = pd.Series(tdist.rvs(df=3, size=200_000, random_state=rng) * 0.01)
    t_var, _ = metrics.student_t_var_cvar(heavy, 0.99)
    g_var, _ = metrics.gaussian_var_cvar(heavy, 0.99)
    # at 99%, a t-fit sees the fat tail Gaussian misses → more extreme (more negative)
    assert t_var < g_var


def test_student_t_cvar_worse_than_var():
    from scipy.stats import t as tdist
    import pandas as pd
    rng = np.random.default_rng(2)
    heavy = pd.Series(tdist.rvs(df=4, size=100_000, random_state=rng) * 0.01)
    var, cvar = metrics.student_t_var_cvar(heavy, 0.95)
    assert cvar < var


def test_portfolio_return_is_the_log_of_the_weighted_simple_returns():
    """A daily-rebalanced portfolio earns sum(w * simple return) a day, so its log return is
    log(sum(w * exp(r))), not sum(w * r). Two assets, three days, worked by hand."""
    import pandas as pd
    lr = pd.DataFrame({"A": [np.log(2.0), 0.0, np.log(0.8), np.nan],
                       "B": [np.log(0.5), np.log(3.0), np.log(0.8), 1.0]})
    pr = metrics.portfolio_returns(lr, ["A", "B"], np.array([0.25, 0.75]))
    # day 1: .25*2 + .75*.5 = .875;  day 2: .25*1 + .75*3 = 2.5;  day 3: .8 whatever the weights
    assert len(pr) == 3                                  # the row with a missing return is dropped
    np.testing.assert_allclose(pr.to_numpy(), np.log([0.875, 2.5, 0.8]), rtol=0, atol=1e-12)
    # one dollar, rebalanced daily, ends at the product of the daily gross returns
    assert float(np.exp(pr.cumsum()).iloc[-1]) == pytest.approx(0.875 * 2.5 * 0.8)
