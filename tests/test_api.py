import json

import pytest
from fastapi.testclient import TestClient

from api.main import app
from risk_engine import data

client = TestClient(app)
GOOD = {"holdings": {"AAPL": 20000, "MSFT": 20000, "JPM": 20000}}


def test_health_reports_data_freshness():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_as_of"] == "2026-06-18"
    assert body["stale"] is True and body["data_source"] == "pinned"


def test_health_degrades_when_no_snapshot_is_readable(monkeypatch):
    def unavailable():
        raise data.SnapshotUnavailable("no bundle could be read")
    monkeypatch.setattr(data, "load_snapshot", unavailable)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "degraded", "detail": "no bundle could be read"}


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


def _ok_live(monkeypatch):
    base = data.load_snapshot(use_memo=False).prices["AAPL"].dropna() * 0.5

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
    assert r.json()["data"]["as_of"] == "2026-06-18"


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


@pytest.mark.parametrize("holdings", [
    {f"T{i}": 1 for i in range(51)}, {"AAPL": 1, "A/B": 1}, {"AAPL": 1, "": 1},
])
def test_shape_errors_are_422(holdings):
    assert _post({"holdings": holdings}).status_code == 422


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
