import io
import json
import os
import re
import sys
import types
import urllib.error
import warnings
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import build_snapshot as bs  # noqa: E402
from risk_engine import data  # noqa: E402

WIKI = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr></table>"""


def series(values, start="2026-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_parse_sp500_maps_gics_to_yahoo_and_dots_to_dashes():
    assert bs.parse_sp500(WIKI) == [("AAPL", "Apple Inc.", "Technology"),
                                    ("BRK-B", "Berkshire Hathaway", "Financial Services")]


def test_universe_keeps_previous_list_when_table_looks_wrong():
    rows = [(f"T{i}", "x", "Technology") for i in range(400)]
    prev = {"OLD": {"name": "old"}}
    uni, used_prev = bs.build_universe(rows, prev)
    assert used_prev and uni == prev
    with pytest.raises(RuntimeError, match="400 rows"):
        bs.build_universe(rows, None)


def test_universe_adds_curated_etfs():
    rows = [(f"T{i}", "x", "Technology") for i in range(500)]
    uni, used_prev = bs.build_universe(rows, None)
    assert not used_prev
    assert uni["SPY"]["curated"] and uni["TLT"]["asset_class"] == "Bond"
    assert uni["T0"]["in_sp500"] and len(uni) == 500 + len(bs.CURATED_ETFS)


def test_incremental_appends_new_closes():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    fresh = {"A": series([12, 13], start="2026-01-06")}
    merged, repull = bs.merge_incremental(stored, fresh)
    assert repull == [] and list(merged["A"]) == [10, 11, 12, 13]


def test_rescaled_history_triggers_full_repull():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    fresh = {"A": series([11.88, 13], start="2026-01-06")}   # a dividend re-scaled history by 1%
    merged, repull = bs.merge_incremental(stored, fresh)
    assert repull == ["A"] and list(merged["A"]) == [10, 11, 12]


def test_no_overlap_also_triggers_repull():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    _, repull = bs.merge_incremental(stored, {"A": series([20, 21], start="2026-02-02")})
    assert repull == ["A"]


def test_failed_fetch_keeps_stored_history():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    merged, _ = bs.merge_incremental(stored, {})
    assert list(merged["A"]) == [10, 11, 12]


def test_fresh_closes_with_duplicate_dates_and_disorder_merge_cleanly():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    messy = pd.Series([13.0, 12.0, 14.0, 13.0],
                      index=pd.to_datetime(["2026-01-07", "2026-01-06", "2026-01-08", "2026-01-07"]))
    merged, repull = bs.merge_incremental(stored, {"A": messy})
    assert repull == []
    assert merged.index.is_monotonic_increasing and merged.index.is_unique
    assert list(merged["A"]) == [10, 11, 12, 13, 14]       # the later duplicate wins


def test_empty_fresh_series_keeps_stored_history():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    merged, repull = bs.merge_incremental(stored, {"A": pd.Series(dtype=float)})
    assert repull == [] and list(merged["A"]) == [10, 11, 12]


def test_bad_print_that_reverses_is_quarantined_but_real_crash_is_not():
    prices = pd.DataFrame({
        "BADTICK": series([100, 101, 300, 102, 103, 104, 105]),
        "CRASH": series([100, 101, 50, 49, 48, 50, 51]),
        "OK": series([100, 101, 102, 103, 104, 105, 106]),
    })
    q = bs.quality_gate(prices)
    assert set(q) == {"BADTICK"} and "bad print" in q["BADTICK"]


def test_stale_ticker_is_quarantined():
    vals = [100.0] * 10
    prices = pd.DataFrame({"LIVE": series(vals), "DEAD": series(vals[:4] + [np.nan] * 6)})
    q = bs.quality_gate(prices)
    assert set(q) == {"DEAD"} and "no new close" in q["DEAD"]


def test_gate_handles_a_column_that_is_nan_except_one_value():
    live = series([100.0] * 10)
    only_recent = pd.Series(np.nan, index=live.index)
    only_recent.iloc[-1] = 50.0
    only_old = pd.Series(np.nan, index=live.index)
    only_old.iloc[1] = 50.0
    prices = pd.DataFrame({"LIVE": live, "ONE_NEW": only_recent, "ONE_OLD": only_old,
                           "EMPTY": pd.Series(np.nan, index=live.index)})
    q = bs.quality_gate(prices)
    assert set(q) == {"ONE_OLD", "EMPTY"}
    assert "no new close" in q["ONE_OLD"] and q["EMPTY"] == "no prices"


def test_gate_does_not_flag_or_warn_when_a_close_repeats():
    prices = pd.DataFrame({"FLAT": series([100.0] * 8),
                           "REPEAT": series([100, 100, 100, 101, 101, 102, 102, 102])})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert bs.quality_gate(prices) == {}


def test_gate_on_a_table_with_no_tickers_or_rows_is_empty():
    assert bs.quality_gate(pd.DataFrame()) == {}
    assert bs.quality_gate(pd.DataFrame({"A": pd.Series(dtype=float)})) == {"A": "no prices"}


def _frame(closes):
    """A yfinance-shaped download: (field, ticker) columns."""
    return pd.concat({"Close": closes, "Volume": closes * 0 + 1}, axis=1)


def _install_fake_yfinance(monkeypatch, frame):
    module = types.ModuleType("yfinance")
    module.download = lambda tickers, **kwargs: frame
    monkeypatch.setitem(sys.modules, "yfinance", module)


def test_fetch_yf_reads_closes_and_retries_a_miss_over_urllib(monkeypatch):
    idx = pd.bdate_range("2026-01-02", periods=4, tz="America/New_York")
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": np.nan}, index=idx)
    _install_fake_yfinance(monkeypatch, _frame(closes))
    monkeypatch.setattr(bs, "_fetch_urllib", lambda t, start: series([7, 8], start="2026-01-02").rename(t))
    got, failed = bs.fetch_yf(["A", "B"], "2026-01-01")
    assert failed == {}
    assert list(got["A"]) == [1, 2, 3, 4] and got["A"].index.tz is None
    assert list(got["B"]) == [7, 8]


def test_fetch_yf_names_the_reason_when_the_retry_also_fails(monkeypatch):
    idx = pd.bdate_range("2026-01-02", periods=3)
    closes = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": np.nan}, index=idx)
    _install_fake_yfinance(monkeypatch, _frame(closes))

    def refuse(t, start):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(bs, "_fetch_urllib", refuse)
    got, failed = bs.fetch_yf(["A", "B"], "2026-01-01")
    assert set(got) == {"A"} and "URLError" in failed["B"] and "connection refused" in failed["B"]


def test_fetch_yf_survives_an_empty_download_and_stops_retrying_during_an_outage(monkeypatch):
    _install_fake_yfinance(monkeypatch, pd.DataFrame())
    tried = []

    def refuse(t, start):
        tried.append(t)
        raise urllib.error.URLError("network is unreachable")

    monkeypatch.setattr(bs, "_fetch_urllib", refuse)
    tickers = [f"T{i}" for i in range(40)]
    got, failed = bs.fetch_yf(tickers, "2026-01-01")
    assert got == {} and set(failed) == set(tickers)
    assert len(tried) == bs.FALLBACK_GIVE_UP
    assert "consecutive" in failed["T39"] and all(failed.values())


def test_fetch_yf_with_no_tickers_does_not_call_yfinance(monkeypatch):
    monkeypatch.delitem(sys.modules, "yfinance", raising=False)
    assert bs.fetch_yf([], "2026-01-01") == ({}, {})


def _french_zip(body):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.CSV", body)
    return buf.getvalue()


FRENCH = """This file was created by using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
20171229,    0.10,    0.20,    0.30,    0.01
20180102,    0.80,   -0.10,    0.20,    0.01
20180103,  -99.99,    0.10,    0.20,    0.01
20180104,    0.50,    0.10,    0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def test_fetch_factors_scales_to_decimals_filters_to_2018_and_marks_missing(monkeypatch):
    monkeypatch.setattr(bs, "_http", lambda url, timeout=30: _french_zip(FRENCH))
    f = bs.fetch_factors()
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert list(f.index) == list(pd.to_datetime(["2018-01-02", "2018-01-03", "2018-01-04"]))
    assert f.loc["2018-01-02", "Mkt-RF"] == pytest.approx(0.008)
    assert np.isnan(f.loc["2018-01-03", "Mkt-RF"])


def test_fetch_factors_names_the_problem_when_the_file_changes_shape(monkeypatch):
    monkeypatch.setattr(bs, "_http", lambda url, timeout=30: _french_zip("no table here\n"))
    with pytest.raises(RuntimeError, match="Mkt-RF"):
        bs.fetch_factors()


def _fake_world(n_good=100, n_fail=0, benchmark_fails=False):
    rows = [(f"T{i}", "x", "Technology") for i in range(n_good + n_fail)]
    html = ("<table id='constituents'><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>"
            + "".join(f"<tr><td>{t}</td><td>x</td><td>Information Technology</td></tr>"
                      for t, _, _ in rows) + "</table>")
    failing = {f"T{i}" for i in range(n_good, n_good + n_fail)}
    if benchmark_fails:
        failing.add("SPY")

    def fetch(tickers, start):
        idx = pd.bdate_range("2018-01-02", "2026-06-18")
        got = {t: pd.Series(np.linspace(50, 150, len(idx)), index=idx, name=t)
               for t in tickers if t not in failing}
        return got, {t: "no data returned" for t in tickers if t in failing}

    factors = pd.DataFrame({"Mkt-RF": [0.0], "SMB": [0.0], "HML": [0.0], "RF": [0.0]},
                           index=pd.to_datetime(["2026-05-29"]))
    return html, fetch, factors


@pytest.fixture
def small_bounds(monkeypatch):
    monkeypatch.setattr(bs, "SP500_MIN", 1)
    monkeypatch.setattr(bs, "SP500_MAX", 10_000)


def test_full_run_writes_a_readable_bundle(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    report = bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["as_of"] == "2026-06-18" and report["failed"] == {}
    assert report["sp500_rows_skipped"] == 0 and report["unmapped_sectors"] == []
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert "T0" in snap.prices.columns and "SPY" in snap.universe
    assert not (tmp_path / "out" / "blocked_report.json").exists()


def test_too_many_failures_blocks_publishing(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(n_good=100, n_fail=5)     # ~4% of ~132 tickers
    with pytest.raises(SystemExit, match="not publishing"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)


def test_benchmark_failure_always_blocks_publishing(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(benchmark_fails=True)
    with pytest.raises(SystemExit, match="benchmark"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)


def test_incremental_without_previous_bundle_runs_full(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    report = bs.run("incremental", str(tmp_path / "none"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["mode"] == "full"


def _run_full(tmp_path, html, fetch, factors, out="out"):
    return bs.run("full", str(tmp_path / "none"), str(tmp_path / out),
                  fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)


def test_every_ticker_failing_stops_with_a_reason_not_a_traceback(tmp_path, small_bounds):
    html, _, factors = _fake_world()

    def nothing(tickers, start):
        return {}, {t: "no data returned" for t in tickers}

    with pytest.raises(SystemExit, match="benchmark SPY failed"):
        _run_full(tmp_path, html, nothing, factors)


def test_fetch_that_drops_tickers_silently_still_names_them_in_the_report(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()

    def forgetful(tickers, start):
        got, failed = fetch(tickers, start)
        got.pop("T5")
        return got, failed

    report = _run_full(tmp_path, html, forgetful, factors)
    assert report["failed"] == {"T5": "no data returned"}


def test_empty_universe_stops_with_a_reason(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    no_columns = pd.DataFrame(index=pd.bdate_range("2026-06-15", periods=3))
    bs.write_bundle(str(tmp_path / "prev"), no_columns, {}, factors,
                    {"as_of": "2026-06-18", "mode": "full"})
    headers_only = "<table id='constituents'><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr></table>"
    with pytest.raises(SystemExit, match="universe is empty"):
        bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: headers_only, get_factors=lambda: factors)


def test_wikipedia_outage_stops_with_a_reason(tmp_path, small_bounds):
    _, fetch, factors = _fake_world()

    def down():
        raise urllib.error.URLError("name resolution failed")

    with pytest.raises(SystemExit, match="S&P 500 list.*name resolution failed"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=down, get_factors=lambda: factors)


def test_unparseable_wikipedia_page_stops_with_a_reason(tmp_path, small_bounds):
    _, fetch, factors = _fake_world()
    with pytest.raises(SystemExit, match="S&P 500 list"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: "<html>layout changed</html>", get_factors=lambda: factors)


def test_table_of_the_wrong_size_without_a_previous_universe_stops_with_a_reason(tmp_path):
    html, fetch, factors = _fake_world(n_good=20)
    with pytest.raises(SystemExit, match="20 rows"):
        _run_full(tmp_path, html, fetch, factors)


def _factors(rows):
    return pd.DataFrame(rows, columns=["Mkt-RF", "SMB", "HML", "RF"],
                        index=pd.to_datetime(["2026-05-27", "2026-05-28", "2026-05-29"][:len(rows)]))


def test_factor_rows_with_missing_values_are_dropped_and_counted(tmp_path, small_bounds):
    html, fetch, _ = _fake_world()
    nan = np.nan
    factors = _factors([[0.01, 0.0, 0.0, 0.0], [nan, 0.0, 0.0, 0.0], [0.02, 0.0, np.inf, 0.0]])
    report = _run_full(tmp_path, html, fetch, factors)
    assert report["factor_rows_dropped"] == 2 and report["factors_through"] == "2026-05-27"
    published = data.read_bundle(str(tmp_path / "out"), "release").factors
    assert len(published) == 1 and not published.isna().any().any()


def test_factors_without_a_single_complete_row_block_publishing(tmp_path, small_bounds):
    html, fetch, _ = _fake_world()
    factors = _factors([[np.nan, 0.0, 0.0, 0.0], [0.0, np.nan, 0.0, 0.0]])
    with pytest.raises(SystemExit, match="factor.*not publishing"):
        _run_full(tmp_path, html, fetch, factors)


def test_factors_with_the_wrong_columns_block_publishing(tmp_path, small_bounds):
    html, fetch, _ = _fake_world()
    factors = pd.DataFrame({"Mkt-RF": [0.0], "SMB": [0.0], "HML": [0.0]},
                           index=pd.to_datetime(["2026-05-29"]))
    with pytest.raises(SystemExit, match="Mkt-RF.*not publishing"):
        _run_full(tmp_path, html, fetch, factors)


def test_factor_download_failure_blocks_publishing_with_its_reason(tmp_path, small_bounds):
    html, fetch, _ = _fake_world()

    def down():
        raise RuntimeError("Fama-French file has no header")

    with pytest.raises(SystemExit, match="factors.*Fama-French file has no header"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: html, get_factors=down)


def _dated_world(end, rescale_on_incremental=(), calls=None, n=30,
                 fail_incremental=(), fail_full=()):
    """Closes are a function of the date alone, so any window of history agrees with
    any other, except where a ticker is listed in rescale_on_incremental. Tickers in
    fail_incremental / fail_full fail on the short pull / the pull from START."""
    html, _, factors = _fake_world(n_good=n)

    def fetch(tickers, start):
        if calls is not None:
            calls.append((sorted(tickers), start))
        incremental = start != bs.START
        idx = pd.bdate_range(start, end)
        got, failed = {}, {}
        for t in tickers:
            if t in (fail_incremental if incremental else fail_full):
                failed[t] = "no data returned"
                continue
            values = 50 + 0.01 * (idx - pd.Timestamp("2018-01-01")).days
            if t in rescale_on_incremental and incremental:
                values = values * 0.98
            got[t] = pd.Series(values, index=idx, name=t)
        return got, failed

    return html, fetch, factors


def _build_previous(tmp_path):
    html, fetch, factors = _dated_world("2026-06-17")
    bs.run("full", str(tmp_path / "none"), str(tmp_path / "prev"),
           fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    return html, factors


def test_incremental_run_extends_the_previous_bundle(tmp_path, small_bounds):
    html, factors = _build_previous(tmp_path)
    calls = []
    _, fetch, _ = _dated_world("2026-06-19", calls=calls)

    def must_not_refetch_factors():
        raise AssertionError("incremental runs reuse the stored factors")

    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=must_not_refetch_factors)
    assert report["mode"] == "incremental" and report["as_of"] == "2026-06-19"
    assert report["repulled"] == [] and report["failed"] == {}
    assert len(calls) == 1 and calls[0][1] == "2026-06-03"     # ten business days before the last stored close
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert snap.prices.index.max() == pd.Timestamp("2026-06-19")
    assert snap.prices.index.is_unique and snap.prices.index.min() == pd.Timestamp(bs.START)
    assert snap.prices["T0"].notna().all()


def test_incremental_run_repulls_a_rescaled_ticker_and_fetches_new_ones_in_full(tmp_path, small_bounds):
    html, _ = _build_previous(tmp_path)
    html = html.replace("</table>",
                        "<tr><td>NEWCO</td><td>x</td><td>Information Technology</td></tr></table>")
    calls = []
    _, fetch, factors = _dated_world("2026-06-19", rescale_on_incremental={"T3"}, calls=calls)
    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["repulled"] == ["T3"]
    assert len(calls) == 2 and calls[1] == (["NEWCO", "T3"], bs.START)
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert snap.prices["NEWCO"].dropna().index.min() == pd.Timestamp(bs.START)
    assert snap.prices["T3"].iloc[-1] == pytest.approx(snap.prices["T4"].iloc[-1])


def test_main_passes_its_arguments_to_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(bs, "run", lambda mode, prev, out: seen.update(mode=mode, prev=prev, out=out))
    monkeypatch.setattr(sys, "argv", ["build_snapshot.py", "--mode", "full",
                                      "--prev-dir", "p", "--out-dir", "o"])
    bs.main()
    assert seen == {"mode": "full", "prev": "p", "out": "o"}


def test_report_and_universe_are_plain_json(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    _run_full(tmp_path, html, fetch, factors)
    for name in ("refresh_report.json", "universe.json"):
        with open(tmp_path / "out" / name) as f:
            assert isinstance(json.load(f), dict)


def _spiking(fetch, tickers):
    """Wrap a fetch so each listed ticker has one wild close that the next day reverses."""
    def spiking_fetch(requested, start):
        got, failed = fetch(requested, start)
        for t in tickers:
            if t in got:
                got[t] = got[t].copy()
                got[t].iloc[1000] *= 3
        return got, failed
    return spiking_fetch


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def test_failed_ticker_with_old_stored_history_is_quarantined_not_published(tmp_path, small_bounds):
    html, _ = _build_previous(tmp_path)
    _, fetch, factors = _dated_world("2026-06-26", fail_incremental={"T7"})
    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["failed"] == {"T7": "no data returned"}
    assert report["quarantined"] == {"T7": "no new close since 2026-06-17"}
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert "T7" not in snap.prices.columns and "T7" in snap.quarantined
    assert snap.as_of == "2026-06-26"


def test_repulled_ticker_whose_full_repull_fails_is_quarantined_not_published(tmp_path, small_bounds):
    html, _ = _build_previous(tmp_path)
    _, fetch, factors = _dated_world("2026-06-26", rescale_on_incremental={"T3"}, fail_full={"T3"})
    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["repulled"] == ["T3"] and "T3" in report["failed"]
    assert "no new close" in report["quarantined"]["T3"]
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert "T3" not in snap.prices.columns and "T3" in snap.quarantined


def test_a_bad_print_is_quarantined_and_left_out_of_the_published_prices(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    report = _run_full(tmp_path, html, _spiking(fetch, ["T5"]), factors)
    assert report["failed"] == {} and set(report["quarantined"]) == {"T5"}
    assert "bad print" in report["quarantined"]["T5"]
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert "T5" not in snap.prices.columns and "T4" in snap.prices.columns
    assert "bad print" in snap.quarantined["T5"]


def test_a_bad_print_on_the_benchmark_blocks_publishing(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    with pytest.raises(SystemExit, match="benchmark SPY held back"):
        _run_full(tmp_path, html, _spiking(fetch, ["SPY"]), factors)


def test_quarantine_alone_can_pass_the_two_percent_cap(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    with pytest.raises(SystemExit, match="not publishing") as exit_info:
        _run_full(tmp_path, html, _spiking(fetch, ["T1", "T2", "T3", "T4", "T5"]), factors)
    assert "T1: suspected bad print" in str(exit_info.value)


def test_a_fifty_percent_jump_that_reverses_is_a_bad_print_but_thirty_five_is_not():
    prices = pd.DataFrame({"UP_50": series([100, 100, 150, 100, 100, 100, 100]),
                           "UP_35": series([100, 100, 135, 100, 100, 100, 100])})
    assert set(bs.quality_gate(prices)) == {"UP_50"}


def test_the_next_close_must_undo_half_the_jump_to_make_a_bad_print():
    prices = pd.DataFrame({"UNDOES_60": series([100, 100, 200, 140, 140, 140, 140]),
                           "UNDOES_40": series([100, 100, 200, 160, 160, 160, 160])})
    assert set(bs.quality_gate(prices)) == {"UNDOES_60"}


def test_a_blocked_build_publishes_nothing_but_a_report_naming_the_offenders(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(n_good=100, n_fail=5)
    with pytest.raises(SystemExit, match="not publishing") as exit_info:
        _run_full(tmp_path, html, fetch, factors)
    out = tmp_path / "out"
    assert os.listdir(out) == ["blocked_report.json"]
    blocked = _read_json(out / "blocked_report.json")
    assert set(blocked["failed"]) == {f"T{i}" for i in range(100, 105)}
    assert "tickers failed or held back" in blocked["blocked_reason"]
    assert all(f"T{i}: no data returned" in str(exit_info.value) for i in range(100, 105))
    with pytest.raises(data.SnapshotUnavailable):
        data.read_bundle(str(out), "release")


def test_a_blocked_build_names_at_most_ten_offenders(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(n_good=100, n_fail=15)
    with pytest.raises(SystemExit) as exit_info:
        _run_full(tmp_path, html, fetch, factors)
    message = str(exit_info.value)
    assert len(re.findall(r"^\s*T\d+: no data returned", message, re.M)) == 10
    assert "5 more" in message


def test_a_failed_benchmark_blocks_with_a_report_and_no_bundle(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(benchmark_fails=True)
    with pytest.raises(SystemExit, match="SPY: no data returned"):
        _run_full(tmp_path, html, fetch, factors)
    out = tmp_path / "out"
    assert os.listdir(out) == ["blocked_report.json"]
    blocked = _read_json(out / "blocked_report.json")
    assert blocked["blocked_reason"].startswith("benchmark SPY failed to download")
    assert blocked["as_of"] is None


def test_a_benchmark_bad_print_blocks_with_a_report_and_no_bundle(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    with pytest.raises(SystemExit, match="SPY: suspected bad print"):
        _run_full(tmp_path, html, _spiking(fetch, ["SPY"]), factors)
    out = tmp_path / "out"
    assert os.listdir(out) == ["blocked_report.json"]
    assert "held back" in _read_json(out / "blocked_report.json")["blocked_reason"]


def test_report_records_that_there_was_no_previous_bundle(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    assert _run_full(tmp_path, html, fetch, factors)["previous_bundle"] == "none"


def test_report_records_that_the_previous_bundle_was_used(tmp_path, small_bounds):
    html, _ = _build_previous(tmp_path)
    _, fetch, factors = _dated_world("2026-06-19")
    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["previous_bundle"] == "used"


def test_report_gives_the_reason_a_previous_bundle_was_rejected(tmp_path, small_bounds):
    html, _ = _build_previous(tmp_path)
    (tmp_path / "prev" / "prices.csv.gz").write_bytes(b"not a gzip file")
    _, fetch, factors = _dated_world("2026-06-19")
    report = bs.run("incremental", str(tmp_path / "prev"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["mode"] == "full"
    assert report["previous_bundle"].startswith("rejected:")
    assert "prices.csv.gz" in report["previous_bundle"]


def test_fetch_yf_turns_a_download_exception_into_a_reasoned_exit(monkeypatch):
    module = types.ModuleType("yfinance")

    def refuse(*args, **kwargs):
        raise ConnectionError("proxy refused")

    module.download = refuse
    monkeypatch.setitem(sys.modules, "yfinance", module)
    with pytest.raises(SystemExit, match="yfinance download failed.*ConnectionError.*proxy refused"):
        bs.fetch_yf(["A"], "2026-01-01")


def test_parse_skips_rows_without_a_symbol():
    html = WIKI.replace("</table>", "<tr><td></td><td>Ghost Corp</td><td>Financials</td></tr></table>")
    assert [r[0] for r in bs.parse_sp500(html)] == ["AAPL", "BRK-B"]


def test_run_counts_skipped_rows_and_lists_unmapped_sectors(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    html = html.replace("</table>",
                        "<tr><td></td><td>Ghost Corp</td><td>Financials</td></tr>"
                        "<tr><td>XYZ</td><td>Coin Corp</td><td>Digital Assets</td></tr></table>")
    report = _run_full(tmp_path, html, fetch, factors)
    assert report["sp500_rows_skipped"] == 1 and report["unmapped_sectors"] == ["XYZ"]
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert snap.universe["XYZ"]["sector"] == "Unknown" and "nan" not in snap.universe


def test_refresh_requirements_are_all_pinned():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements-refresh.txt")
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    assert lines and all("==" in line for line in lines)
