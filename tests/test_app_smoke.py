import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from risk_engine import data

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
    assert "MSFT" in _sidebar_text(at) and "Not included" in _sidebar_text(at)


def _recent_listing(sym):
    """A live-looked-up ticker that has traded for only 300 sessions."""
    snap = data.load_snapshot()
    index = snap.prices.index[-300:]
    walk = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.0004, 0.01, len(index))))
    meta = {"name": "NewCo", "sector": "Technology", "type": "stock", "asset_class": "Equity",
            "in_sp500": False, "curated": False}
    return data.LiveResult(sym, "ok", "", pd.Series(walk, index=index, name=sym), meta)


def test_short_history_shows_backtest_numbers_without_a_verdict(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "fetch_live", _recent_listing)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nNEWCO 5000").run()
    at.button(key="load_paste").click().run()
    assert not at.exception
    body = " ".join(str(m.value) for m in at.markdown)
    assert "too few" in body
    assert "PASS</td>" not in body and "FAIL</td>" not in body
    assert any("trading days" in str(w.value) for w in at.warning)      # the short-window warning


def test_crisis_a_holding_missed_is_named_not_dropped(monkeypatch):
    st.cache_data.clear()
    monkeypatch.setattr(data, "fetch_live", _recent_listing)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nNEWCO 5000").run()
    at.button(key="load_paste").click().run()
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
    body = " ".join(str(m.value) for m in at.markdown)
    notes = " ".join(str(w.value) for w in at.warning) + " ".join(str(i.value) for i in at.info)
    assert "at least 60" in notes
    assert "Your portfolio in one sentence" not in body
