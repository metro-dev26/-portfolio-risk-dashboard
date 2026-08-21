import json

from risk_engine.analyze import analyze_portfolio

HOLDINGS = {"AAPL": 20000, "MSFT": 20000, "JPM": 20000, "XOM": 20000, "TLT": 20000}


def test_analyze_returns_expected_structure():
    r = analyze_portfolio(HOLDINGS)
    assert set(r) == {"metrics", "factors", "optimizer", "backtest"}
    assert {"hist_var", "gaussian_var", "sharpe", "annual_vol", "max_drawdown"} <= set(r["metrics"])
    assert set(r["factors"]) == {"betas", "alpha_annual", "r2", "variance_split"}
    assert {"current_sharpe", "max_sharpe_weights", "min_variance_weights", "top_sector"} <= set(r["optimizer"])
    assert set(r["backtest"]) == {"historical", "gaussian"}


def test_analyze_values_are_finite_and_sane():
    r = analyze_portfolio(HOLDINGS)
    assert r["metrics"]["hist_var"] < 0                      # a loss quantile is negative
    assert 0 <= r["metrics"]["annual_vol"] < 5
    ms = r["optimizer"]["max_sharpe_weights"]
    assert abs(sum(ms.values()) - 1.0) < 1e-6                # optimizer weights sum to 1
    assert abs(sum(r["factors"]["variance_split"].values()) - 1.0) < 1e-9


def test_analyze_is_json_serializable():
    json.dumps(analyze_portfolio(HOLDINGS))                  # raises if numpy leaks


def test_include_backtest_false_omits_backtest():
    r = analyze_portfolio(HOLDINGS, include_backtest=False)
    assert "backtest" not in r
