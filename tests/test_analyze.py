import dataclasses
import json

import pytest

from risk_engine import data
from risk_engine.analyze import PortfolioError, analyze_portfolio

HOLDINGS = {"AAPL": 20000, "MSFT": 20000, "JPM": 20000, "XOM": 20000, "TLT": 20000}


def test_analyze_returns_expected_structure():
    r = analyze_portfolio(HOLDINGS)
    assert set(r) == {"metrics", "factors", "optimizer", "backtest", "data"}
    assert {"hist_var", "gaussian_var", "sharpe", "annual_vol", "max_drawdown"} <= set(r["metrics"])
    assert set(r["factors"]) == {"betas", "alpha_annual", "r2", "variance_split", "observations"}
    assert {"current_sharpe", "max_sharpe_weights", "min_variance_weights", "top_sector"} <= set(r["optimizer"])
    assert set(r["backtest"]) == {"historical", "gaussian"}
    assert r["data"]["as_of"] == "2026-06-18" and r["data"]["window_status"] == "ok"
    assert r["data"]["window_message"] == ""
    assert all(row["observations"] >= 100 and isinstance(row["passed"], bool)
               for row in r["backtest"].values())
    assert r["data"]["crisis_coverage"]["COVID-19 Crash"] == []


def test_unresolvable_ticker_raises_with_reason_per_ticker():
    with pytest.raises(PortfolioError) as e:
        analyze_portfolio({"AAPL": 1000, "XYZQQ": 500})
    assert e.value.problems[0]["ticker"] == "XYZQQ"
    assert "disabled" in e.value.problems[0]["reason"]     # tests run with live lookups off


def test_empty_factor_data_is_refused_with_a_reason():
    snap = data.load_snapshot(use_memo=False)
    bare = dataclasses.replace(snap, factors=snap.factors.iloc[0:0])
    with pytest.raises(PortfolioError) as e:
        analyze_portfolio(HOLDINGS, snapshot=bare)
    assert e.value.problems[0]["ticker"] == "(factors)"
    assert "is empty" in e.value.problems[0]["reason"]


def _with_sectors(**sectors):
    snap = data.load_snapshot(use_memo=False)
    universe = {t: {**meta, "sector": sectors.get(t, meta["sector"])} for t, meta in snap.universe.items()}
    return dataclasses.replace(snap, universe=universe)


def test_broad_market_funds_are_not_the_top_sector():
    r = analyze_portfolio({"SPY": 60000, "TLT": 40000})
    assert r["optimizer"]["top_sector"] == "None identifiable"
    assert r["optimizer"]["top_sector_pct"] == 0.0


def test_bonds_are_reported_by_asset_class_not_as_the_top_sector():
    r = analyze_portfolio({"AAPL": 30000, "JPM": 20000, "TLT": 40000, "IEF": 10000})
    assert r["optimizer"]["top_sector"] == "Technology"
    assert abs(r["optimizer"]["top_sector_pct"] - 0.30) < 1e-9
    assert r["optimizer"]["non_equity_pct"] == pytest.approx({"Bond": 0.50})


def test_holdings_with_no_real_sector_are_not_the_top_sector():
    snap = _with_sectors(AAPL=data.UNKNOWN_SECTOR, JPM=data.FUND_SECTOR)
    r = analyze_portfolio({"AAPL": 40000, "JPM": 30000, "XOM": 30000}, snapshot=snap)
    assert r["optimizer"]["top_sector"] == "Energy"
    assert abs(r["optimizer"]["top_sector_pct"] - 0.30) < 1e-9


def test_top_sector_says_so_when_no_holding_has_a_sector_to_name():
    snap = _with_sectors(AAPL=data.UNKNOWN_SECTOR, MSFT="International")
    r = analyze_portfolio({"AAPL": 50000, "MSFT": 30000, "SPY": 20000}, snapshot=snap)
    assert r["optimizer"]["top_sector"] == "None identifiable"
    assert r["optimizer"]["top_sector_pct"] == 0.0


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


def test_asymmetric_weights_pin_ticker_ordering():
    """Equal weights can't catch a tickers<->weights<->w_ms zip shuffle. An
    asymmetric portfolio does: 80% AAPL (Technology) vs 20% TLT (Govt Bonds).
    If the pairing were shuffled, top_sector / top_sector_pct would be wrong."""
    r = analyze_portfolio({"AAPL": 80000, "TLT": 20000})
    assert r["optimizer"]["top_sector"] == "Technology"
    assert abs(r["optimizer"]["top_sector_pct"] - 0.80) < 1e-9
    assert set(r["optimizer"]["max_sharpe_weights"]) == {"AAPL", "TLT"}


def test_confidence_is_threaded_through():
    """A higher confidence level must push VaR to a more extreme (more negative)
    loss quantile — proves `confidence` actually reaches the metric functions."""
    r95 = analyze_portfolio(HOLDINGS, include_backtest=False)
    r99 = analyze_portfolio(HOLDINGS, confidence=0.99, include_backtest=False)
    assert r99["metrics"]["hist_var"] < r95["metrics"]["hist_var"]
