from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
GOOD = {"holdings": {"AAPL": 20000, "MSFT": 20000, "JPM": 20000}}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


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


import pytest


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
