import json
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from risk_engine import config, data, importer
from risk_engine import factors as fac
from risk_engine.config import CRISES

# Resolve app.py against the repo root, not the caller's cwd or the test-file
# directory. AppTest.from_file() resolves a *relative* path against the file
# that calls it (tests/), which breaks in CI; an absolute path is portable
# across streamlit versions and working directories.
APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


def test_model_validation_section_present():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Model validation" in body


def test_factor_section_present():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Factor" in body and ("exposure" in body.lower() or "tilt" in body.lower())


def _sidebar_text(at):
    return " ".join(str(m.value) for m in at.sidebar.markdown) + " ".join(
        str(c.value) for c in at.sidebar.caption)


def test_example_portfolio_and_freshness_line_show_on_first_visit():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.session_state["is_example"] is True
    assert "example portfolio" in _sidebar_text(at)
    assert "dollars" in _sidebar_text(at)                 # amounts are $, not share counts
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Prices as of 2026-06-18 close" in body


def test_stale_data_shows_a_banner():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("refresh looks behind" in str(w.value) for w in at.warning)


def test_paste_with_a_bad_ticker_lists_it_as_not_included():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nMSFT 5000\nXYZQQ 3000").run()
    at.button(key="load_paste").click().run()
    assert not at.exception
    text = _sidebar_text(at)
    assert "XYZQQ" in text and "$3,000 excluded" in text
    assert at.session_state["holdings"] == {"AAPL": 10000.0, "MSFT": 5000.0, "XYZQQ": 3000.0}


def test_pasted_html_is_escaped():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nMSFT 5000\n<img src=x onerror=alert(1)> 100").run()
    at.button(key="load_paste").click().run()
    text = _sidebar_text(at)
    assert "&lt;img" in text and "<img" not in text


def test_zero_amounts_never_become_equal_weights():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 0\nMSFT 0").run()
    at.button(key="load_paste").click().run()
    assert not at.exception
    assert any("at least 2 holdings" in str(w.value) for w in at.warning)


def test_shared_link_restores_the_portfolio():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "AAPL:50000,TLT:50000"
    at.run()
    assert not at.exception
    assert at.session_state["holdings"] == {"AAPL": 50000.0, "TLT": 50000.0}
    assert at.session_state["is_example"] is False


def test_mangled_share_link_is_reported_not_silently_shrunk():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "AAPL:50000,TLT:50000,MSFT"
    at.run()
    assert not at.exception
    assert at.session_state["holdings"] == {"AAPL": 50000.0, "TLT": 50000.0}
    assert "MSFT" in _sidebar_text(at) and "Import notes" in _sidebar_text(at)


def _body(at):
    return " ".join(str(m.value) for m in at.markdown)


def _paste(at, text):
    at.text_area(key="paste_box").input(text).run()
    at.button(key="load_paste").click().run()
    return at


def _fake_live(sessions=None, listed=None, **meta):
    """Stand-in for data.fetch_live: a USD ticker with `sessions` of history (all of it by
    default); `listed` maps a symbol to the date its history starts."""
    profile = {"name": "Fake Co", "sector": "Technology", "type": "stock",
               "asset_class": "Equity", "in_sp500": False, "curated": False, **meta}

    def fetch(sym):
        index = data.load_snapshot().prices.index
        index = index if sessions is None else index[-sessions:]
        if listed and sym in listed:
            index = index[index >= listed[sym]]
        steps = np.random.default_rng(sum(map(ord, sym))).normal(0.0004, 0.01, len(index))
        walk = pd.Series(100 * np.exp(np.cumsum(steps)), index=index, name=sym)
        return data.LiveResult(sym, "ok", "", walk, profile)

    return fetch


@pytest.fixture
def fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


class _Upload:
    """The part of st.file_uploader's return value the app reads."""

    def __init__(self, text, size=None):
        self._bytes = text.encode()
        self.size = len(self._bytes) if size is None else size

    def getvalue(self):
        return self._bytes


def _with_upload(monkeypatch, text, size=None):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: _Upload(text, size))


def test_short_history_shows_backtest_numbers_without_a_verdict(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "fetch_live", _fake_live(300))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 10000\nNEWCO 5000")
    assert not at.exception
    body = _body(at)
    assert "too few" in body
    assert "PASS</td>" not in body and "FAIL</td>" not in body
    assert any("trading days" in str(w.value) for w in at.warning)      # the short-window warning


def test_crisis_a_holding_missed_is_named_not_dropped(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "fetch_live", _fake_live(300))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 10000\nNEWCO 5000")
    notes = " ".join(str(c.value) for c in at.caption) + " ".join(str(i.value) for i in at.info)
    assert "COVID-19 Crash" in notes and "NEWCO" in notes


def test_thin_factor_overlap_gives_a_message_not_numbers(tmp_path, monkeypatch):
    bundle = tmp_path / "snapshot"
    shutil.copytree(Path(__file__).parent / "fixtures" / "snapshot", bundle)
    factors = pd.read_csv(bundle / "factors.csv", index_col=0, parse_dates=True)
    factors.loc[:"2018-02-28"].to_csv(bundle / "factors.csv")
    monkeypatch.setenv("MARKETPLUG_DATA_DIR", str(bundle))
    st.cache_data.clear()
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    body = _body(at)
    notes = " ".join(str(w.value) for w in at.warning) + " ".join(str(i.value) for i in at.info)
    assert "at least 60" in notes
    assert "Your portfolio in one sentence" not in body


def _stress_traces(at):
    for chart in at.get("plotly_chart"):
        traces = json.loads(chart.proto.spec)["data"]
        if any(t.get("name") == "COVID-19 Crash" for t in traces):
            return traces
    raise AssertionError("no stress chart on the page")


def test_every_crisis_gets_its_own_line_and_colour():
    at = AppTest.from_file(APP, default_timeout=60).run()
    traces = _stress_traces(at)
    assert [t["name"] for t in traces] == [crisis[0] for crisis in CRISES]
    assert len({t["line"]["color"] for t in traces}) == len(CRISES)


def test_stress_section_includes_tariff_shock():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert "2025 Tariff Shock" in _body(at)


def _missed_by_crisis(at):
    (note,) = [str(c.value) for c in at.caption if "not yet trading" in str(c.value)]
    entries = (entry.split(" (", 1) for entry in note.split(": ", 1)[1].split("; "))
    return {label: set(tickers.rstrip(")").split(", ")) for label, tickers in entries}


def test_crisis_note_pairs_each_crisis_with_the_holdings_that_missed_it(monkeypatch, fresh_cache):
    monkeypatch.setattr(data, "fetch_live", _fake_live(
        listed={"LATEA": "2021-06-01", "LATEB": "2022-06-24"}))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 10000\nLATEA 5000\nLATEB 5000")
    assert _missed_by_crisis(at) == {
        "2018 Q4 Selloff": {"LATEA", "LATEB"},
        "COVID-19 Crash": {"LATEA", "LATEB"},
        "2022 Bear Market": {"LATEB"},
    }
    assert "2025 Tariff Shock" in _body(at)


def _share_param(at):
    value = at.query_params["p"]
    return value[0] if isinstance(value, list) else value


def test_share_link_matches_the_table_and_is_written_only_when_it_changes(monkeypatch):
    from streamlit.runtime.state.query_params_proxy import QueryParamsProxy
    writes = []
    original = QueryParamsProxy.__setitem__

    def counting(self, key, value):
        writes.append((key, value))
        return original(self, key, value)

    monkeypatch.setattr(QueryParamsProxy, "__setitem__", counting)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert len(writes) == 1
    at.sidebar.select_slider[0].set_value("90%").run()
    assert len(writes) == 1
    _paste(at, "AAPL 10000\nMSFT 5000")
    assert _share_param(at) == importer.encode_share({"AAPL": 10000.0, "MSFT": 5000.0})
    assert len(writes) == 2


def test_add_holding_keeps_table_edits_and_deletions():
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 10000\nMSFT 5000\nJPM 2000")
    key = f"amounts_{at.session_state['editor_v']}"
    at.session_state[key] = {"edited_rows": {0: {"Amount $": 12345.0}}, "added_rows": [],
                             "deleted_rows": [2]}
    at.run()
    at.text_input(key="search_other").input("NVDA")
    at.button(key="add_holding").click().run()
    assert not at.exception
    assert at.session_state["holdings"] == {"AAPL": 12345.0, "MSFT": 5000.0, "NVDA": 10000.0}


def test_the_holdings_table_warns_that_the_share_link_carries_the_holdings():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert ("The page address now contains these holdings and amounts — share it only with "
            "people who should see them.") in _sidebar_text(at)


def test_add_holding_after_editing_the_example_table_keeps_the_edited_rows():
    at = AppTest.from_file(APP, default_timeout=60).run()
    key = f"amounts_{at.session_state['editor_v']}"
    at.session_state[key] = {"edited_rows": {0: {"Amount $": 12345.0}}, "added_rows": [],
                             "deleted_rows": [2, 3, 4]}
    at.run()
    assert at.session_state["is_example"] is False
    assert "example portfolio" not in _sidebar_text(at)
    at.text_input(key="search_other").input("NVDA")
    at.button(key="add_holding").click().run()
    assert not at.exception
    assert at.session_state["holdings"] == {"AAPL": 12345.0, "MSFT": 20000.0, "NVDA": 10000.0}


def test_add_holding_to_the_untouched_example_starts_a_new_portfolio():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_input(key="search_other").input("NVDA")
    at.button(key="add_holding").click().run()
    assert at.session_state["holdings"] == {"NVDA": 10000.0}


def test_factor_section_reads_the_loaded_snapshot_not_a_second_load(monkeypatch):
    st.cache_data.clear()
    loaded = data.load_snapshot().factors
    monkeypatch.setattr(fac, "load_factors", lambda: loaded.iloc[:100])
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any(f"Factor data through {loaded.index.max().date()}" in str(c.value) for c in at.caption)


def test_ninety_percent_confidence_reads_ten_percent_of_days():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.select_slider[0].set_value("90%").run()
    body = _body(at)
    assert "worst 10% of days" in body and "worst 9%" not in body


SECTORLESS_QUOTES = {
    "fund": {"quoteType": "ETF"},
    "stock with no sector": {"quoteType": "EQUITY"},
    "stock with a blank sector": {"quoteType": "EQUITY", "sector": "  "},
    "index": {"quoteType": "INDEX", "typeDisp": "Index"},
}
# Equities' share of a portfolio split evenly between AAPL and one holding of each kind.
EQUITY_PCT_BESIDE_AAPL = {"fund": 50, "stock with no sector": 100,
                          "stock with a blank sector": 100, "index": 50}


def _sectorless_live(kind, symbol="X"):
    return _fake_live(**data.live_meta({**SECTORLESS_QUOTES[kind], "symbol": symbol}))


def _diversification_box(at):
    return next(str(m.value) for m in at.markdown if "Diversification check" in str(m.value))


@pytest.mark.parametrize("kind", SECTORLESS_QUOTES)
def test_portfolio_of_holdings_with_no_known_sector_is_not_called_concentrated(
        monkeypatch, fresh_cache, kind):
    monkeypatch.setattr(data, "fetch_live", _sectorless_live(kind))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "ZZA 6000\nZZB 4000")
    assert not at.exception
    box = _diversification_box(at)
    assert "concentrated in" not in box
    assert "insight danger" not in box
    assert "100% is in holdings whose sector isn't known here" in box


@pytest.mark.parametrize("kind", SECTORLESS_QUOTES)
def test_holdings_with_no_known_sector_are_left_out_of_the_sector_check_but_counted(
        monkeypatch, fresh_cache, kind):
    monkeypatch.setattr(data, "fetch_live", _sectorless_live(kind))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 5000\nZZA 5000")
    box = _diversification_box(at)
    assert "50% concentrated in Technology" in box
    assert f"{EQUITY_PCT_BESIDE_AAPL[kind]}% in equities overall" in box
    assert "50% is in holdings whose sector isn't known here" in box


def test_a_broad_market_fund_beside_bonds_is_not_called_a_concentration():
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "SPY 60000\nTLT 40000")
    assert not at.exception
    box = _diversification_box(at)
    assert "concentrated in Broad Market" not in box
    assert "insight danger" not in box
    assert "60% is in broad-market or international funds, which spread across many sectors" in box
    assert "60% in equities overall" in box


@pytest.mark.parametrize("label", ["Broad Market", "International"])
def test_funds_that_span_sectors_are_left_out_of_the_sector_check_but_counted_as_equities(
        monkeypatch, fresh_cache, label):
    monkeypatch.setattr(data, "fetch_live", _fake_live(sector=label, type="etf"))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 5000\nZZA 5000")
    box = _diversification_box(at)
    assert "50% concentrated in Technology" in box and f"concentrated in {label}" not in box
    assert "100% in equities overall" in box
    assert "50% is in broad-market or international funds, which spread across many sectors" in box


def test_sector_text_from_yahoo_is_escaped(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "fetch_live", _fake_live(sector="<img src=x onerror=alert(1)>"))
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 1000\nEVIL 9000")
    body = _body(at)
    assert "&lt;img src=x" in body and "<img src=x" not in body


def test_a_yahoo_outage_is_not_retried_on_every_rerun(monkeypatch):
    st.cache_data.clear()
    calls = []

    def down(sym):
        calls.append(sym)
        return data.LiveResult(sym, "unavailable", "data source unavailable (URLError)")

    monkeypatch.setattr(data, "fetch_live", down)
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(), "AAPL 10000\nMSFT 5000\nQQQQZ 100")
    assert calls == ["QQQQZ"]
    at.sidebar.select_slider[0].set_value("90%").run()
    assert calls == ["QQQQZ"]
    assert "data source unavailable" in _sidebar_text(at)


def test_oversized_csv_is_refused_with_a_plain_message(monkeypatch):
    _with_upload(monkeypatch, "Symbol,Value\n" + "AAPL,1\n" * 400_000)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("2 MB" in str(e.value) for e in at.sidebar.error)
    assert not [b for b in at.button if b.key == "load_csv"]


def _quantity_csv(*extra):
    return "Symbol,Quantity\nAAPL,10\n" + "".join(f"{t},1\n" for t in extra)


def test_csv_share_counts_are_valued_with_a_capped_number_of_lookups(monkeypatch):
    st.cache_data.clear()
    calls = []
    fetch = _fake_live()
    monkeypatch.setattr(data, "fetch_live", lambda sym: (calls.append(sym), fetch(sym))[1])
    monkeypatch.setattr(config, "LIVE_MAX_UI", 3)
    _with_upload(monkeypatch, _quantity_csv("ZZA", "ZZB", "ZZC", "ZZD", "ZZE"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="load_csv").click().run()
    assert not at.exception
    assert calls == ["ZZA", "ZZB", "ZZC"]
    assert "no price for ZZD" in _sidebar_text(at)


def test_csv_share_count_lookups_stop_when_the_time_budget_is_spent(monkeypatch):
    st.cache_data.clear()
    calls = []
    fetch = _fake_live()

    def slow(sym):
        calls.append(sym)
        time.sleep(0.3)
        return fetch(sym)

    monkeypatch.setattr(data, "fetch_live", slow)
    monkeypatch.setattr(config, "LIVE_BUDGET_UI_S", 0.2)
    _with_upload(monkeypatch, _quantity_csv("ZZA", "ZZB", "ZZC"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.button(key="load_csv").click().run()
    assert calls == ["ZZA"]
    assert "no price for ZZB" in _sidebar_text(at)


def test_not_included_and_import_notes_are_separate_labelled_lists():
    at = _paste(AppTest.from_file(APP, default_timeout=60).run(),
                "AAPL 10000\nMSFT 5000\nXYZQQ 3000\nbadline")
    text = _sidebar_text(at)
    order = [text.index(label) for label in ("Not included", "XYZQQ", "Import notes", "badline")]
    assert order == sorted(order)


def test_beginners_guide_describes_the_paste_upload_search_flow():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio[0].set_value("📖 Beginner's Guide").run()
    body = _body(at)
    assert "paste, upload or search" in body and "pick the stocks and bonds" not in body


def test_dark_theme_is_pinned_so_the_sidebar_is_readable_in_light_mode():
    config_toml = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    assert tomllib.loads(config_toml.read_text())["theme"]["base"] == "dark"


@pytest.mark.parametrize("bytes_over_server_limit,refused", [(0, False), (1, True)])
def test_app_csv_cap_sits_exactly_at_the_servers_upload_limit(
        monkeypatch, bytes_over_server_limit, refused):
    config_toml = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"
    limit = tomllib.loads(config_toml.read_text())["server"]["maxUploadSize"] * 1024 * 1024
    _with_upload(monkeypatch, "Symbol,Value\nAAPL,1\n", size=limit + bytes_over_server_limit)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert bool(at.sidebar.error) is refused
    assert bool([b for b in at.button if b.key == "load_csv"]) is not refused
