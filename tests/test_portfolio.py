import numpy as np
import pandas as pd
import pytest

from risk_engine import data, portfolio
from tests.conftest import HOLDINGS


def snap():
    return data.load_snapshot(use_memo=False)


def ok_live(series):
    def fetch(sym):
        meta = {"name": sym, "sector": "Technology", "type": "stock", "asset_class": "Equity",
                "in_sp500": False, "curated": False}
        return data.LiveResult(sym, "ok", "", series.rename(sym), meta)
    return fetch


def test_known_tickers_come_from_the_snapshot():
    r = portfolio.resolve_holdings({"aapl": 1000, "TLT": 500}, snap(),
                                   live_fetch=lambda s: (_ for _ in ()).throw(AssertionError))
    assert r.tickers == ["AAPL", "TLT"] and r.rejected == []
    assert r.source == {"AAPL": "snapshot", "TLT": "snapshot"}
    assert r.meta["TLT"]["asset_class"] == "Bond"


def test_unknown_ticker_goes_live_and_failures_carry_reasons():
    s = snap()
    calls = []

    def fetch(sym):
        calls.append(sym)
        return data.LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
    r = portfolio.resolve_holdings({"AAPL": 1000, "XYZQQ": 300}, s, live_fetch=fetch)
    assert calls == ["XYZQQ"]
    assert r.rejected == [portfolio.Rejection("XYZQQ", 300, "no such ticker on Yahoo Finance")]


def test_live_cap_and_time_budget():
    s = snap()
    base = s.prices["AAPL"].dropna()
    many = {f"ZZ{i}": 100.0 for i in range(7)}
    r = portfolio.resolve_holdings(many, s, live_fetch=ok_live(base), max_live=5)
    assert len(r.tickers) == 5
    assert [x.ticker for x in r.rejected] == ["ZZ5", "ZZ6"]
    assert all("max 5" in x.reason for x in r.rejected)

    ticks = iter([0.0, 0.0, 11.0, 11.0, 11.0])
    r2 = portfolio.resolve_holdings({"ZZ1": 1, "ZZ2": 1}, s, live_fetch=ok_live(base),
                                    budget_s=10.0, clock=lambda: next(ticks))
    assert r2.tickers == ["ZZ1"]
    assert r2.rejected[0].ticker == "ZZ2" and "time budget" in r2.rejected[0].reason


def test_quarantined_ticker_is_rejected_not_looked_up():
    s = snap()
    held = data.Snapshot(s.prices, s.universe, s.factors,
                         dict(s.report, quarantined={"XOM": "suspected bad print on 2026-06-17"}),
                         "pinned")
    r = portfolio.resolve_holdings({"AAPL": 1, "XOM": 1}, held,
                                   live_fetch=lambda t: (_ for _ in ()).throw(AssertionError))
    assert r.rejected[0].ticker == "XOM" and "bad print" in r.rejected[0].reason


def test_window_matches_legacy_returns_for_full_history_portfolio(market):
    _, legacy_lr = market
    r = portfolio.resolve_holdings({t: 1.0 for t in HOLDINGS}, snap())
    w = portfolio.portfolio_window(r.prices, r.tickers)
    assert w.status == "ok"
    diff = (w.returns[HOLDINGS] - legacy_lr[HOLDINGS]).abs().max().max()
    assert diff < 1e-12


def _young_listing(s, start="2024-04-02"):
    young = s.prices["AAPL"].copy() * 0.3
    young[young.index < start] = np.nan
    return young


def test_young_listing_shortens_the_window_but_never_erases_crisis_history():
    s = snap()
    r = portfolio.resolve_holdings({"AAPL": 1, "TLT": 1, "NEWCO": 1}, s,
                                   live_fetch=ok_live(_young_listing(s)))
    w = portfolio.portfolio_window(r.prices, r.tickers)
    assert w.limiting == "NEWCO" and w.start >= pd.Timestamp("2024-04-02")
    assert w.status == "ok"                          # 2024-04 -> 2026-06 is > 504 days
    cov = {c["label"]: c["missing"] for c in portfolio.crisis_coverage(r.prices, r.tickers)}
    assert cov["COVID-19 Crash"] == ["NEWCO"]
    assert cov["2025 Tariff Shock"] == []


def test_short_and_too_short_windows():
    s = snap()
    short = portfolio.resolve_holdings({"AAPL": 1, "NEWCO": 1}, s,
                                       live_fetch=ok_live(_young_listing(s, "2025-01-02")))
    w = portfolio.portfolio_window(short.prices, short.tickers)
    assert w.status == "short" and "NEWCO" in w.message
    tiny = portfolio.resolve_holdings({"AAPL": 1, "NEWCO": 1}, s,
                                      live_fetch=ok_live(_young_listing(s, "2026-01-02")))
    w2 = portfolio.portfolio_window(tiny.prices, tiny.tickers)
    assert w2.status == "too_short" and "NEWCO" in w2.message
    assert "trading days of shared history" in w2.message


def test_benchmark_is_added_once_even_when_held():
    s = snap()
    r = portfolio.resolve_holdings({"AAPL": 1, "SPY": 1}, s)
    w = portfolio.portfolio_window(r.prices, r.tickers, benchmark=s.prices["SPY"])
    assert list(w.returns.columns).count("SPY") == 1
    r2 = portfolio.resolve_holdings({"AAPL": 1, "TLT": 1}, s)
    w2 = portfolio.portfolio_window(r2.prices, r2.tickers, benchmark=s.prices["SPY"])
    assert "SPY" in w2.returns.columns and w2.returns["SPY"].notna().all()


def _no_network(sym):
    raise AssertionError(f"unexpected lookup of {sym}")


def test_duplicate_tickers_merge_amounts_and_columns():
    r = portfolio.resolve_holdings({"aapl": 100, "AAPL": 50, " Aapl ": 25, "TLT": 10}, snap(),
                                   live_fetch=_no_network)
    assert r.holdings == {"AAPL": 175, "TLT": 10}
    assert list(r.prices.columns) == ["AAPL", "TLT"]


def test_duplicate_live_ticker_is_fetched_once_and_merged():
    s = snap()
    calls = []
    fetch = ok_live(s.prices["AAPL"].dropna())

    def counted(sym):
        calls.append(sym)
        return fetch(sym)
    r = portfolio.resolve_holdings({"brk.b": 100, "BRK-B": 50}, s, live_fetch=counted)
    assert calls == ["BRK-B"]
    assert r.holdings == {"BRK-B": 150} and list(r.prices.columns) == ["BRK-B"]


def test_duplicate_of_a_rejected_ticker_is_one_rejection_with_summed_amount():
    calls = []

    def fetch(sym):
        calls.append(sym)
        return data.LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
    r = portfolio.resolve_holdings({"xyzqq": 100, "XYZQQ": 40}, snap(), live_fetch=fetch)
    assert calls == ["XYZQQ"]
    assert r.rejected == [portfolio.Rejection("XYZQQ", 140, "no such ticker on Yahoo Finance")]


def test_input_order_is_kept():
    r = portfolio.resolve_holdings({"TLT": 1, "AAPL": 1, "JPM": 1}, snap(), live_fetch=_no_network)
    assert r.tickers == ["TLT", "AAPL", "JPM"]
    assert list(r.prices.columns) == ["TLT", "AAPL", "JPM"]


def test_no_holdings_resolves_to_an_empty_resolution():
    r = portfolio.resolve_holdings({}, snap(), live_fetch=_no_network)
    assert r.tickers == [] and r.rejected == [] and r.prices.empty


def test_live_series_with_no_prices_is_rejected_with_a_reason():
    s = snap()
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    all_nan = pd.Series(np.nan, index=s.prices.index)
    for series in (empty, all_nan, None):
        def fetch(sym, series=series):
            meta = {"name": sym, "sector": "Technology", "type": "stock",
                    "asset_class": "Equity", "in_sp500": False, "curated": False}
            return data.LiveResult(sym, "ok", "", series, meta)
        r = portfolio.resolve_holdings({"AAPL": 1, "GHOST": 5}, s, live_fetch=fetch)
        assert r.tickers == ["AAPL"]
        assert len(r.rejected) == 1 and r.rejected[0].ticker == "GHOST"
        assert "no price history" in r.rejected[0].reason


def test_snapshot_column_with_no_prices_is_rejected_with_a_reason():
    s = snap()
    prices = s.prices.assign(HOLLOW=np.nan)
    universe = dict(s.universe, HOLLOW=s.universe["AAPL"])
    hollow = data.Snapshot(prices, universe, s.factors, s.report, "pinned")
    r = portfolio.resolve_holdings({"AAPL": 1, "HOLLOW": 2}, hollow, live_fetch=_no_network)
    assert r.tickers == ["AAPL"] and "HOLLOW" not in r.prices.columns
    assert r.rejected[0].ticker == "HOLLOW" and "no price history" in r.rejected[0].reason


def test_window_with_an_all_nan_column_is_refused_not_crashed():
    s = snap()
    prices = s.prices[["AAPL", "TLT"]].assign(GHOST=np.nan)
    w = portfolio.portfolio_window(prices, ["AAPL", "TLT", "GHOST"])
    assert w.status == "too_short" and w.limiting == "GHOST"
    assert "GHOST" in w.message and w.n_days == 0 and w.returns.empty
    assert "GHOST" not in w.first_dates


def test_window_with_no_tickers_is_refused():
    w = portfolio.portfolio_window(pd.DataFrame(), [])
    assert w.status == "too_short" and w.n_days == 0 and w.returns.empty
    assert w.message


def test_single_ticker_window():
    s = snap()
    w = portfolio.portfolio_window(s.prices, ["AAPL"], benchmark=s.prices["SPY"])
    assert w.status == "ok" and w.limiting == "AAPL"
    assert list(w.returns.columns) == ["AAPL", "SPY"]


def test_holdings_with_no_shared_trading_day_are_refused_not_crashed():
    idx = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame({"OLD": pd.Series(100.0, index=idx[:10]),
                           "NEW": pd.Series(50.0, index=idx[20:])})
    w = portfolio.portfolio_window(prices, ["OLD", "NEW"])
    assert w.status == "too_short" and w.limiting == "NEW" and w.n_days == 0
    assert "OLD" in w.message and "NEW" in w.message


def test_a_single_shared_day_gives_no_returns_and_is_refused():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"A": pd.Series(100.0, index=idx[:5]),
                           "B": pd.Series(50.0, index=idx[4:])})
    w = portfolio.portfolio_window(prices, ["A", "B"])
    assert w.status == "too_short" and w.n_days == 0 and w.returns.empty
    assert w.start == idx[4] and w.end == idx[4]


def test_repeated_tickers_are_analysed_once():
    s = snap()
    w = portfolio.portfolio_window(s.prices, ["AAPL", "TLT", "AAPL"])
    assert list(w.returns.columns) == ["AAPL", "TLT"]


def test_a_holding_with_no_prices_misses_every_crisis():
    s = snap()
    prices = s.prices[["AAPL"]].assign(GHOST=np.nan)
    cov = portfolio.crisis_coverage(prices, ["AAPL", "GHOST", "GHOST"])
    assert len(cov) == 4
    assert all(c["missing"] == ["GHOST"] for c in cov)
    assert {"label", "start", "end", "context", "missing"} <= set(cov[0])


def test_interleaved_dates_with_no_overlap_are_refused_not_crashed():
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame({"ODD": pd.Series(10.0, index=idx[0::2]),
                           "EVEN": pd.Series(20.0, index=idx[1::2])})
    w = portfolio.portfolio_window(prices, ["ODD", "EVEN"])
    assert w.status == "too_short" and w.n_days == 0 and w.returns.empty
    assert "These holdings' price dates never line up" in w.message


def _synthetic(n_rows, names=("A", "B"), seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n_rows)
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.01, (n_rows, len(names)))
    return pd.DataFrame(100.0 * np.exp(steps.cumsum(axis=0)), index=idx, columns=list(names))


@pytest.mark.parametrize("rows,status,n_days", [
    (505, "ok", 504), (504, "short", 503), (253, "short", 252), (252, "too_short", 251)])
def test_window_status_changes_exactly_at_the_thresholds(rows, status, n_days):
    w = portfolio.portfolio_window(_synthetic(rows), ["A", "B"])
    assert (w.n_days, w.status) == (n_days, status)
    assert len(w.returns) == n_days


def test_a_gap_is_never_filled_it_becomes_one_multi_day_return_for_every_holding():
    p = _synthetic(300)
    gap, after, before = p.index[100], p.index[101], p.index[99]
    gappy = p.copy()
    gappy.loc[gap, "B"] = np.nan
    w = portfolio.portfolio_window(gappy, ["A", "B"])
    assert gap not in w.returns.index and len(w.returns) == 298
    for t in ("A", "B"):
        assert w.returns.loc[after, t] == pytest.approx(np.log(p.loc[after, t] / p.loc[before, t]))


def test_benchmark_missing_a_day_drops_that_day_instead_of_inventing_a_zero_return():
    p = _synthetic(300, names=("A", "B", "SPY"))
    gap, after, before = p.index[100], p.index[101], p.index[99]
    bench = p["SPY"].drop(index=gap)
    w = portfolio.portfolio_window(p[["A", "B"]], ["A", "B"], benchmark=bench)
    assert gap not in w.returns.index and (w.returns["SPY"] != 0).all()
    assert w.returns.loc[after, "SPY"] == pytest.approx(np.log(p.loc[after, "SPY"]
                                                               / p.loc[before, "SPY"]))
    assert w.returns.loc[after, "A"] == pytest.approx(np.log(p.loc[after, "A"]
                                                             / p.loc[before, "A"]))


def test_benchmark_listed_later_than_the_holdings_sets_the_window_start():
    p = _synthetic(300, names=("A", "B", "SPY"))
    late = p["SPY"].iloc[50:]
    w = portfolio.portfolio_window(p[["A", "B"]], ["A", "B"], benchmark=late)
    assert w.start == late.index[0] and w.returns["SPY"].notna().all()
    assert w.returns.notna().all().all() and len(w.returns) == 249


def test_unnamed_benchmark_is_called_spy():
    p = _synthetic(300, names=("A", "B", "X"))
    w = portfolio.portfolio_window(p[["A", "B"]], ["A", "B"],
                                   benchmark=p["X"].rename(None))
    assert list(w.returns.columns) == ["A", "B", "SPY"]


def _ending(as_of, bdays_before, rows=300):
    end = pd.Timestamp(np.busday_offset(as_of, -bdays_before))
    return pd.Series(100.0, index=pd.bdate_range(end=end, periods=rows))


def test_live_series_that_stopped_trading_is_rejected_as_possibly_delisted():
    s = snap()
    stale = _ending(s.as_of, 20)
    r = portfolio.resolve_holdings({"AAPL": 1, "OLDCO": 5}, s, live_fetch=ok_live(stale))
    assert r.tickers == ["AAPL"] and "OLDCO" not in r.prices.columns
    assert r.rejected == [portfolio.Rejection(
        "OLDCO", 5, f"no recent prices (last close {stale.index[-1].date()}) — possibly delisted")]


def test_live_series_is_current_up_to_five_business_days_behind_the_snapshot():
    s = snap()
    fresh = portfolio.resolve_holdings({"EDGE": 1}, s, live_fetch=ok_live(_ending(s.as_of, 5)))
    assert fresh.tickers == ["EDGE"]
    behind = portfolio.resolve_holdings({"EDGE": 1}, s, live_fetch=ok_live(_ending(s.as_of, 6)))
    assert behind.tickers == [] and "no recent prices" in behind.rejected[0].reason


def test_malformed_tickers_are_rejected_without_using_a_live_lookup():
    s = snap()
    calls = []
    fetch = ok_live(_ending(s.as_of, 0))

    def counted(sym):
        calls.append(sym)
        return fetch(sym)
    unknown = [f"ZZ{i}" for i in range(5)]
    holdings = {"AAPL": 1, "": 1, "A/B": 1, **{t: 1 for t in unknown}}
    r = portfolio.resolve_holdings(holdings, s, live_fetch=counted, max_live=5)
    assert calls == unknown and r.tickers == ["AAPL"] + unknown
    assert [(x.ticker, x.reason) for x in r.rejected] == [
        ("", "not a valid ticker symbol"), ("A/B", "not a valid ticker symbol")]
