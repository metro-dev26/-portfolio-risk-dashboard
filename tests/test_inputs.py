import numpy as np
import pytest

from risk_engine import data
from risk_engine.inputs import HOLDINGS_PROBLEM, PortfolioError, prepare


def _snap():
    return data.load_snapshot()


def test_prepare_builds_weights_returns_and_benchmark():
    inp = prepare({"AAPL": 30000, "TLT": 10000}, _snap())
    assert inp.tickers == ["AAPL", "TLT"] and inp.value == 40000.0
    assert inp.weights.tolist() == [0.75, 0.25]
    assert inp.pr.index.equals(inp.spy_r.index)
    # exact rebalanced log return since project A
    first = np.log(0.75 * np.exp(inp.lr["AAPL"].iloc[0]) + 0.25 * np.exp(inp.lr["TLT"].iloc[0]))
    assert inp.pr.iloc[0] == pytest.approx(first)
    assert inp.window.status == "ok"


def test_holding_spy_itself_does_not_duplicate_the_benchmark_column():
    inp = prepare({"SPY": 50000, "AAPL": 50000}, _snap())
    assert list(inp.lr.columns).count("SPY") == 1
    assert inp.spy_r.equals(inp.lr["SPY"])


def test_rejected_tickers_raise_unless_allowed():
    with pytest.raises(PortfolioError) as e:
        prepare({"AAPL": 1000, "MSFT": 1000, "XYZQQ": 5}, _snap())
    assert e.value.problems[0]["ticker"] == "XYZQQ"
    inp = prepare({"AAPL": 1000, "MSFT": 1000, "XYZQQ": 5}, _snap(), allow_rejected=True)
    assert inp.tickers == ["AAPL", "MSFT"] and [r.ticker for r in inp.res.rejected] == ["XYZQQ"]


def test_fewer_than_two_usable_holdings_is_refused():
    with pytest.raises(PortfolioError) as e:
        prepare({"AAPL": 1000, "XYZQQ": 5}, _snap(), allow_rejected=True)
    assert e.value.problems[0]["ticker"] == HOLDINGS_PROBLEM
    assert [(r["ticker"], r["amount"]) for r in e.value.rejected] == [("XYZQQ", 5)]


def test_analyze_still_exports_portfolio_error():
    from risk_engine.analyze import PortfolioError as Reexported
    assert Reexported is PortfolioError
