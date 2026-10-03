import dataclasses
import json
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from risk_engine import data, portfolio

client = TestClient(app)
GOOD = {"holdings": {"AAPL": 20000, "MSFT": 20000, "JPM": 20000}}


def test_health_reports_data_freshness():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_as_of"] == "2026-06-18"
    assert body["stale"] is True and body["data_source"] == "pinned"


def test_health_degrades_when_no_snapshot_is_readable(monkeypatch, caplog):
    def unavailable():
        raise data.SnapshotUnavailable("no bundle could be read")
    monkeypatch.setattr(data, "load_snapshot", unavailable)
    with caplog.at_level(logging.ERROR, logger="api.main"):
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "degraded", "detail": "no bundle could be read"}
    assert any(rec.exc_info and "no bundle could be read" in str(rec.exc_info[1])
               for rec in caplog.records), "the failure must reach the server log"


def test_analyze_ok():
    r = client.post("/analyze", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert {"metrics", "factors", "optimizer"} <= set(body)
    assert body["backtest"] is not None


def test_analyze_no_backtest():
    r = client.post("/analyze", json={**GOOD, "include_backtest": False})
    assert r.status_code == 200 and r.json()["backtest"] is None


def test_analyze_bad_input_422():
    r = client.post("/analyze", json={"holdings": {"AAPL": 20000}})   # fewer than 2
    assert r.status_code == 422


@pytest.mark.parametrize("raw", [
    '{"holdings": {"AAPL": 1e400, "MSFT": 5000}}',    # inf (JSON overflow)
    '{"holdings": {"AAPL": -1e400, "MSFT": 5000}}',   # -inf
    '{"holdings": {"AAPL": NaN, "MSFT": 5000}}',      # NaN
])
def test_non_finite_amount_returns_422_not_500(raw):
    """Over the wire (not the Python-object path), a 422 that echoes an inf/NaN
    input must still serialize — it previously 500'd in Starlette's JSON encoder."""
    r = client.post("/analyze", content=raw, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, f"expected 422, got {r.status_code}: {r.text[:120]}"
    assert "detail" in r.json()


def test_portfolio_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB", str(tmp_path / "api.db"))
    created = client.post("/portfolios",
                          json={"name": "p1", "holdings": {"AAPL": 1000, "MSFT": 2000}})
    assert created.status_code == 200
    pid = created.json()["id"]
    got = client.get(f"/portfolios/{pid}")
    assert got.status_code == 200 and got.json()["name"] == "p1"
    assert any(r["id"] == pid for r in client.get("/portfolios").json())


def test_get_missing_portfolio_404(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB", str(tmp_path / "api2.db"))
    assert client.get("/portfolios/999").status_code == 404


def _ok_live(monkeypatch, rows=None):
    base = data.load_snapshot(use_memo=False).prices["AAPL"].dropna() * 0.5
    if rows is not None:
        base = base.iloc[-rows:]

    def fake(sym, **_):
        if sym == "XYZQQ":
            return data.LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
        if sym == "RELIANCE.NS":
            return data.LiveResult(sym, "non_usd", "priced in INR, not USD")
        meta = {"name": sym, "sector": "Technology", "type": "stock", "asset_class": "Equity",
                "in_sp500": False, "curated": False}
        return data.LiveResult(sym, "ok", "", base.rename(sym), meta)
    monkeypatch.setattr(data, "fetch_live", fake)


def _post(body):
    return client.post("/analyze", content=json.dumps(body),
                       headers={"Content-Type": "application/json"})


def test_unknown_real_ticker_is_fetched_live(monkeypatch):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 10000, "ZZZZ": 5000}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"]["as_of"] == "2026-06-18"
    assert "ZZZZ" in body["optimizer"]["max_sharpe_weights"]


@pytest.mark.parametrize("bad,fragment", [
    ("XYZQQ", "no such ticker"), ("RELIANCE.NS", "INR"),
])
def test_unresolvable_ticker_is_422_with_per_ticker_reason(monkeypatch, bad, fragment):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 10000, bad: 5000}})
    assert r.status_code == 422
    problems = r.json()["detail"]["problems"]
    assert problems[0]["ticker"] == bad and fragment in problems[0]["reason"]


def test_more_than_five_live_lookups_is_422(monkeypatch):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 1, **{f"ZZ{i}": 1 for i in range(6)}}})
    assert r.status_code == 422
    assert any("max 5" in p["reason"] for p in r.json()["detail"]["problems"])


@pytest.mark.parametrize("holdings,fragment", [
    ({f"T{i}": 1 for i in range(51)}, "between 2 and 50"),
    ({"AAPL": 1, "A/B": 1}, "valid ticker"),
    ({"AAPL": 1, "": 1}, "valid ticker"),
])
def test_shape_errors_are_rejected_by_the_request_schema(holdings, fragment):
    """The pydantic layer answers before any lookup: `detail` is a list of
    validation errors, not the engine's {"problems": [...]} object."""
    r = _post({"holdings": holdings})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail, list)
    assert any(fragment in item["msg"] for item in detail)


def test_analyze_is_503_with_a_reason_when_no_snapshot_is_readable(monkeypatch):
    def unavailable():
        raise data.SnapshotUnavailable("no bundle could be read")
    monkeypatch.setattr(data, "load_snapshot", unavailable)
    r = _post(GOOD)
    assert r.status_code == 503 and "no bundle could be read" in r.json()["detail"]


def test_saved_portfolio_analysis_reports_unresolvable_tickers(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB", str(tmp_path / "api3.db"))
    _ok_live(monkeypatch)
    pid = client.post("/portfolios", json={"name": "p2",
                                           "holdings": {"AAPL": 1000, "XYZQQ": 500}}).json()["id"]
    r = client.get(f"/portfolios/{pid}", params={"analyze": "true"})
    assert r.status_code == 422
    assert r.json()["detail"]["problems"][0]["ticker"] == "XYZQQ"


def test_saved_portfolio_can_be_analysed(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB", str(tmp_path / "api4.db"))
    pid = client.post("/portfolios", json={"name": "p3", **GOOD}).json()["id"]
    r = client.get(f"/portfolios/{pid}", params={"analyze": "true"})
    assert r.status_code == 200, r.text
    assert r.json()["analysis"]["data"]["as_of"] == "2026-06-18"


def test_full_history_portfolio_has_a_graded_backtest():
    body = _post(GOOD).json()
    assert body["data"]["window_status"] == "ok" and body["data"]["window_message"] == ""
    assert body["factors"]["observations"] > 1000
    for row in body["backtest"].values():
        assert row["observations"] >= 100
        assert isinstance(row["passed"], bool)


def test_short_window_is_reported_and_its_backtest_is_not_graded(monkeypatch):
    _ok_live(monkeypatch, rows=253)
    r = _post({"holdings": {"AAPL": 10000, "ZZZZ": 5000}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["data"]["window_status"] == "short" and body["data"]["window_days"] == 252
    assert "ZZZZ" in body["data"]["window_message"]
    for row in body["backtest"].values():
        assert row["observations"] == 2
        assert row["passed"] is None
        assert row["breaches"] >= 0 and 0 <= row["kupiec_p"] <= 1


def test_window_under_a_year_is_422_naming_the_limiting_ticker(monkeypatch):
    _ok_live(monkeypatch, rows=252)
    r = _post({"holdings": {"AAPL": 10000, "ZZZZ": 5000}})
    assert r.status_code == 422
    problem = r.json()["detail"]["problems"][0]
    assert problem["ticker"] == "ZZZZ" and "251" in problem["reason"]


def _snapshot_with_factor_overlap(overlap_days):
    """The pinned snapshot with factor data cut so that exactly `overlap_days`
    of GOOD's return days have a factor row."""
    snap = data.load_snapshot(use_memo=False)
    returns = portfolio.portfolio_window(snap.prices, list(GOOD["holdings"]),
                                         snap.prices["SPY"]).returns
    if overlap_days == 0:
        cut = returns.index[0] - pd.Timedelta(days=1)
    else:
        cut = returns.index[overlap_days - 1]
    snap = dataclasses.replace(snap, factors=snap.factors.loc[:cut])
    assert len(returns.index.intersection(snap.factors.index)) == overlap_days
    return snap


@pytest.mark.parametrize("overlap_days", [0, 1, 30])
def test_too_little_factor_overlap_is_422_not_500(monkeypatch, overlap_days):
    snap = _snapshot_with_factor_overlap(overlap_days)
    monkeypatch.setattr(data, "load_snapshot", lambda *a, **k: snap)
    r = _post(GOOD)
    assert r.status_code == 422, r.text
    problem = r.json()["detail"]["problems"][0]
    assert problem["ticker"] == "(factors)"
    assert str(snap.factors.index.max().date()) in problem["reason"]


def test_openapi_documents_the_engine_422_and_the_503s():
    spec = client.get("/openapi.json").json()
    analyze = spec["paths"]["/analyze"]["post"]["responses"]
    assert {"200", "422", "503"} <= set(analyze)
    assert "problems" in analyze["422"]["description"]
    assert "503" in spec["paths"]["/health"]["get"]["responses"]
