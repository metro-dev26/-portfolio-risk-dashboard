import dataclasses

import pytest
from fastapi.testclient import TestClient

from api import main
from api.main import app
from api.schemas import ReportResponse
from risk_engine import data, optimize
from risk_engine.report import build_report

client = TestClient(app)
BODY = {"holdings": {"AAPL": 30000, "JPM": 20000, "TLT": 40000, "IEF": 10000}}
BONDS_ONLY = {"TLT": 50000, "IEF": 30000, "AGG": 20000}
EVEN_SPLIT = {"AAPL": 50000, "MSFT": 50000}


@pytest.fixture(scope="module")
def resp():
    return client.post("/report", json=BODY)


def test_report_returns_every_section(resp):
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {"score", "stress", "monte_carlo", "optimizer", "data"} <= set(body)
    assert body["score"]["horizon_days"] == 252


def test_same_request_twice_is_byte_identical(resp):
    again = client.post("/report", json=BODY)
    assert again.content == resp.content
    shrunk = [client.post("/report", json={**BODY, "shrinkage": True}).content for _ in range(2)]
    assert shrunk[0] == shrunk[1]


def test_report_is_gzipped_when_the_client_accepts_it():
    r = client.post("/report", json=BODY, headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


def test_unknown_ticker_is_a_422_with_problems():
    r = client.post("/report", json={"holdings": {"AAPL": 1000, "XYZQQ": 500}})
    assert r.status_code == 422
    assert r.json()["detail"]["problems"][0]["ticker"] == "XYZQQ"


def test_confidence_out_of_range_is_a_422():
    assert client.post("/report", json={**BODY, "confidence": 0.5}).status_code == 422


def test_reallocation_rows_carry_the_asset_class(resp):
    rows = resp.json()["optimizer"]["reallocation"]
    assert {r["ticker"]: r["asset_class"] for r in rows} == {
        "AAPL": "Equity", "JPM": "Equity", "TLT": "Bond", "IEF": "Bond"}


def test_stress_dollars_are_a_signed_change_not_a_loss_magnitude(resp):
    body = resp.json()
    value = body["portfolio"]["value"]
    ok = [c for c in body["stress"] if c["status"] == "ok"]
    assert ok and all(c["change_dollars"] == pytest.approx(c["total"] * value) for c in ok)
    assert any(c["change_dollars"] < 0 for c in ok)       # a crash shows as a negative number
    assert body["score"]["portfolio"]["loss_dollars"] >= 0    # the score's loss is a magnitude
    assert all("loss_dollars" not in c for c in body["stress"])


def test_flat_priced_holding_serves_a_null_correlation_cell(monkeypatch):
    snap = data.load_snapshot(use_memo=False)
    flat = dataclasses.replace(snap, prices=snap.prices.assign(JPM=100.0))
    monkeypatch.setattr(data, "load_snapshot", lambda *a, **k: flat)
    r = client.post("/report", json=BODY)
    assert r.status_code == 200, r.text
    corr = r.json()["correlation"]
    i = corr["tickers"].index("JPM")
    assert corr["matrix"][i][corr["tickers"].index("AAPL")] is None


def test_an_unusable_snapshot_is_a_503(monkeypatch):
    def unavailable():
        raise data.SnapshotUnavailable("no bundle could be read")
    monkeypatch.setattr(data, "load_snapshot", unavailable)
    r = client.post("/report", json=BODY)
    assert r.status_code == 503
    assert "no bundle could be read" in r.json()["detail"]


def test_an_internal_value_error_stays_a_500(monkeypatch):
    def misaligned(*a, **k):
        raise ValueError("portfolio and S&P 500 returns must share the same dates")
    monkeypatch.setattr(main, "build_report", misaligned)
    r = TestClient(app, raise_server_exceptions=False).post("/report", json=BODY)
    assert r.status_code == 500


def _flat_jpm(snap):
    return dataclasses.replace(snap, prices=snap.prices.assign(JPM=100.0))


def _jpm_listed_late(snap):
    prices = snap.prices.copy()
    prices.loc[prices.index < "2022-06-01", "JPM"] = float("nan")
    return dataclasses.replace(snap, prices=prices)


def _thin_factors(snap):
    return dataclasses.replace(snap, factors=snap.factors.iloc[:30])


@pytest.mark.parametrize("variant", [None, _flat_jpm, _jpm_listed_late, _thin_factors],
                         ids=["full", "flat-holding", "crises-not-covered", "no-factor-data"])
def test_response_models_keep_every_key_the_report_builds(variant):
    # A response_model silently drops any key it doesn't declare, so a 200 alone can't show
    # the models mirror the report dict. Round-tripping it can.
    snap = data.load_snapshot(use_memo=False)
    rep = build_report(BODY["holdings"], snapshot=variant(snap) if variant else snap)
    assert ReportResponse.model_validate(rep).model_dump() == rep


def test_response_models_keep_every_key_with_no_top_sector():
    rep = build_report(BONDS_ONLY)
    assert rep["optimizer"]["sector_mix"]["top_sector"] is None
    assert ReportResponse.model_validate(rep).model_dump() == rep


def test_response_models_keep_every_key_with_no_diversifier():
    rep = build_report(EVEN_SPLIT)
    assert rep["risk_contribution"]["diversifier"] is None
    assert ReportResponse.model_validate(rep).model_dump() == rep


def test_response_models_keep_every_key_when_the_optimizer_is_unavailable(monkeypatch):
    def boom(*a, **k):
        raise optimize.OptimizerError("did not converge (test)")
    monkeypatch.setattr(optimize, "max_sharpe_weights", boom)
    rep = build_report(BODY["holdings"])
    assert rep["optimizer"]["available"] is False
    assert ReportResponse.model_validate(rep).model_dump() == rep
