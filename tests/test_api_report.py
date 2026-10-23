import dataclasses
import json

import pandas as pd
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


def test_universe_lists_tickers_with_names_sorted():
    rows = client.get("/universe").json()
    tickers = [r["ticker"] for r in rows]
    assert tickers == sorted(tickers) and "AAPL" in tickers and "TLT" in tickers
    tlt = next(r for r in rows if r["ticker"] == "TLT")
    assert tlt == {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF",
                   "sector": "Govt Bonds", "asset_class": "Bond", "in_sp500": False}


def test_universe_leaves_out_quarantined_and_priceless_tickers(monkeypatch):
    snap = data.load_snapshot()
    held_back = dataclasses.replace(
        snap, report={**snap.report, "quarantined": {"AAPL": "stale price"}},
        prices=snap.prices.drop(columns=["MSFT"]))
    monkeypatch.setattr(data, "load_snapshot", lambda: held_back)
    tickers = [r["ticker"] for r in client.get("/universe").json()]
    assert "AAPL" not in tickers and "MSFT" not in tickers and "JPM" in tickers


def test_import_paste_returns_holdings_and_notes():
    r = client.post("/import", json={"format": "paste", "text": "AAPL 10000\nMSFT 5,000\ngarbage"})
    assert r.status_code == 200
    body = r.json()
    assert body["holdings"] == {"AAPL": 10000.0, "MSFT": 5000.0}
    assert any("line 3" in n for n in body["notes"])


def test_import_csv_values_share_counts_at_the_snapshot_close():
    close = float(data.load_snapshot().prices["AAPL"].dropna().iloc[-1])
    r = client.post("/import", json={"format": "csv", "text": "Symbol,Quantity\nAAPL,10\n"})
    assert r.json()["holdings"]["AAPL"] == pytest.approx(10 * close)


def test_import_csv_skips_a_share_count_it_has_no_price_for():
    r = client.post("/import", json={"format": "csv", "text": "Symbol,Quantity\nZZZZ,10\nAAPL,1\n"})
    body = r.json()
    assert set(body["holdings"]) == {"AAPL"}
    assert any("ZZZZ" in n and "no price" in n for n in body["notes"])


def test_import_rejects_oversized_text_and_unknown_format():
    assert client.post("/import", json={"format": "csv", "text": "x" * 200_001}).status_code == 422
    assert client.post("/import", json={"format": "xlsx", "text": "a"}).status_code == 422


@pytest.mark.parametrize("body", [
    {"format": "paste"},
    {"text": "AAPL 1"},
    {"format": "paste", "text": None},
    {"format": "paste", "text": 12345},
    {"format": "paste", "text": ["AAPL 1"]},
    ["paste", "AAPL 1"],
], ids=["no-text", "no-format", "null-text", "number-text", "list-text", "array-body"])
def test_import_malformed_bodies_are_422s(body):
    assert client.post("/import", json=body).status_code == 422


def test_import_body_that_is_not_json_is_a_422():
    r = client.post("/import", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_import_text_at_the_limit_is_accepted():
    r = client.post("/import", json={"format": "paste", "text": "x" * 200_000})
    assert r.status_code == 200
    assert r.json()["holdings"] == {}


@pytest.mark.parametrize("text", [
    "Symbol,Value\n" + "A" * 150_000 + ",1\n",
    "Symbol,Value\nAAPL,\x00\x00\n",
    'Symbol,Value\nAAPL,"unterminated\n',
    "Symbol,Quantity\nAAPL," + "9" * 5_000 + "\n",
    "\ufeff\ufeff",
], ids=["field-over-csv-limit", "nul-bytes", "open-quote", "overflowing-quantity", "bom-only"])
def test_import_hostile_csv_is_a_200_with_notes_never_a_500(text):
    r = TestClient(app, raise_server_exceptions=False).post("/import", json={"format": "csv", "text": text})
    assert r.status_code == 200, r.text
    assert r.json()["holdings"] == {}


def _unavailable():
    raise data.SnapshotUnavailable("no bundle could be read")


def test_universe_is_a_503_when_the_snapshot_is_unusable(monkeypatch):
    monkeypatch.setattr(data, "load_snapshot", _unavailable)
    r = client.get("/universe")
    assert r.status_code == 503
    assert "no bundle could be read" in r.json()["detail"]


def test_import_csv_is_a_503_when_the_snapshot_is_unusable_but_paste_needs_none(monkeypatch):
    monkeypatch.setattr(data, "load_snapshot", _unavailable)
    csv_resp = client.post("/import", json={"format": "csv", "text": "Symbol,Quantity\nAAPL,10\n"})
    assert csv_resp.status_code == 503
    assert "no bundle could be read" in csv_resp.json()["detail"]
    assert client.post("/import", json={"format": "paste", "text": "AAPL 10"}).status_code == 200


class _CountingPrices(pd.DataFrame):
    column_reads = 0

    def __getitem__(self, key):
        type(self).column_reads += 1
        return super().__getitem__(key)


def test_import_csv_looks_a_repeated_symbol_up_once(monkeypatch):
    snap = data.load_snapshot()
    close = float(snap.prices["AAPL"].dropna().iloc[-1])
    counting = dataclasses.replace(snap, prices=_CountingPrices(snap.prices))
    _CountingPrices.column_reads = 0
    monkeypatch.setattr(data, "load_snapshot", lambda: counting)
    text = "Symbol,Quantity\n" + "AAPL,1\n" * 500
    body = client.post("/import", json={"format": "csv", "text": text}).json()
    assert body["holdings"]["AAPL"] == pytest.approx(500 * close)
    assert _CountingPrices.column_reads == 1


@pytest.mark.parametrize("path, body", [
    ("/import", rb'{"format": "paste", "text": "\ud800 10"}'),
    ("/import", rb'{"format": "csv", "text": "Symbol,Value\n\ud800,10\n"}'),
    ("/import", rb'{"format": "paste", "text": "AAPL \ud800"}'),
    ("/import", rb'{"format": "x\ud800", "text": "a"}'),
    ("/report", rb'{"holdings": {"\ud800": 100}}'),
], ids=["paste-ticker", "csv-ticker", "paste-amount", "format-echo", "report-holdings-key"])
def test_a_lone_surrogate_is_a_422_never_a_500(path, body):
    r = TestClient(app, raise_server_exceptions=False).post(
        path, content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 422, r.text[:200]
    assert "\N{REPLACEMENT CHARACTER}" in r.content.decode("utf-8")


def test_a_422_does_not_echo_an_oversized_input_back():
    r = client.post("/import", json={"format": "csv", "text": "x" * 300_000})
    assert r.status_code == 422
    assert len(r.content) < 2_000
    echoed = r.json()["detail"][0]["input"]
    assert len(echoed) == 200 and echoed.endswith("…")


def test_a_422_truncates_an_oversized_ticker_in_the_message_and_the_echo():
    # 60,000 characters: still 300 times the echo cap, but under /report's 64 KB body cap.
    # A 300,000-character ticker is now a 413 before it reaches validation.
    r = client.post("/report", json={"holdings": {"A" * 60_000: 100}})
    assert r.status_code == 422
    assert len(r.content) < 2_000


def test_import_caps_the_notes_it_returns():
    r = client.post("/import", json={"format": "paste", "text": "x\n" * 100_000})
    notes = r.json()["notes"]
    assert len(notes) == 51
    assert notes[0].startswith("line 1:") and notes[49].startswith("line 50:")
    assert notes[50] == "…and 99950 more lines were skipped"
    assert len(r.content) < 10_000


def test_import_returns_exactly_fifty_notes_uncapped():
    r = client.post("/import", json={"format": "paste", "text": "x\n" * 50})
    notes = r.json()["notes"]
    assert len(notes) == 50 and not any("more lines" in n for n in notes)


def test_a_huge_holdings_object_is_refused_before_it_is_parsed():
    """One 20 MB POST of 1.5M holdings once pushed the process from 302 to 799 MiB."""
    body = json.dumps({"holdings": {f"T{i}": 1 for i in range(100_000)}}).encode()
    assert len(body) > 1_000_000
    r = client.post("/report", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert len(r.content) < 1_000


def test_a_422_echoes_only_the_first_twenty_keys_of_a_big_holdings_object():
    r = client.post("/report", json={"holdings": {f"T{i}": 1 for i in range(3_000)}})
    assert r.status_code == 422
    assert len(r.content) < 3_000
    echoed = r.json()["detail"][0]["input"]
    assert len(echoed) == 21 and echoed["…"] == "2980 more"
    assert list(echoed)[:20] == [f"T{i}" for i in range(20)]


def test_json_safe_keeps_the_first_twenty_list_items_and_counts_the_rest():
    assert main._json_safe(list(range(100))) == list(range(20)) + ["…80 more"]
    assert main._json_safe(list(range(20))) == list(range(20))


def test_json_safe_keeps_the_first_twenty_dict_keys_and_counts_the_rest():
    out = main._json_safe({f"k{i}": i for i in range(100)})
    assert list(out) == [f"k{i}" for i in range(20)] + ["…"]
    assert out["…"] == "80 more"
    exact = {f"k{i}": i for i in range(20)}
    assert main._json_safe(exact) == exact


def test_json_safe_caps_containers_inside_containers():
    out = main._json_safe([{"input": list(range(50))}])
    assert out == [{"input": list(range(20)) + ["…30 more"]}]


def test_import_accepts_200000_characters_even_when_json_escapes_every_one():
    """Six bytes per character is the worst case: 200,000 of them are about 1.2 MB."""
    body = json.dumps({"format": "paste", "text": "é" * 200_000}).encode("ascii")
    assert len(body) > 1_200_000
    r = client.post("/import", content=body, headers={"Content-Type": "application/json"})
    assert r.status_code == 200
