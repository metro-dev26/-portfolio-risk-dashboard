import threading
import time

import pytest
from fastapi.testclient import TestClient

from api import limits, main


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_fixed_window_allows_the_limit_then_says_when_to_retry():
    clock = Clock()
    rl = limits.RateLimiter({"/report": 2}, default=5, clock=clock)
    assert rl.check("1.1.1.1", "/report") is None
    assert rl.check("1.1.1.1", "/report") is None
    clock.t = 20.0
    assert rl.check("1.1.1.1", "/report") == pytest.approx(40.0)
    assert rl.check("2.2.2.2", "/report") is None          # per IP
    assert rl.check("1.1.1.1", "/health") is None          # per bucket
    clock.t = 60.0
    assert rl.check("1.1.1.1", "/report") is None          # new window


def test_limiter_memory_is_bounded_under_many_ips():
    clock = Clock()
    rl = limits.RateLimiter({}, default=5, clock=clock, max_keys=100)
    for i in range(1000):
        rl.check(f"10.0.{i // 256}.{i % 256}", "/x")
        clock.t += 0.1
    assert len(rl._hits) <= 101


def test_client_ip_uses_the_proxy_added_last_entry():
    class Req:
        headers = {"x-forwarded-for": "6.6.6.6, 9.9.9.9"}
        client = None
    assert limits.client_ip(Req()) == "9.9.9.9"

    class Direct:
        headers = {}
        client = type("C", (), {"host": "127.0.0.1"})()
    assert limits.client_ip(Direct()) == "127.0.0.1"


def test_compute_gate_raises_busy_when_full():
    gate = limits.ComputeGate(slots=1, wait_s=0.01)
    with gate.slot():
        with pytest.raises(limits.Busy):
            with gate.slot():
                pass
    with gate.slot():
        pass                                                  # released


def test_compute_gate_turns_away_new_arrivals_once_enough_are_waiting():
    gate = limits.ComputeGate(slots=1, wait_s=10.0, max_waiting=1)
    outcome = []

    def waiter():
        try:
            with gate.slot():
                outcome.append("ran")
        except limits.Busy:
            outcome.append("busy")

    with gate.slot():
        t = threading.Thread(target=waiter, daemon=True)
        t.start()
        deadline = time.monotonic() + 5
        while gate._waiting < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert gate._waiting == 1
        started = time.monotonic()
        with pytest.raises(limits.Busy):
            with gate.slot():
                pass
        assert time.monotonic() - started < 1.0       # refused outright, did not queue for 10 s
    t.join(timeout=5)
    assert outcome == ["ran"]                          # the queued one still got its turn
    assert gate._waiting == 0


def test_compute_gate_stops_counting_a_waiter_that_timed_out():
    gate = limits.ComputeGate(slots=1, wait_s=0.01, max_waiting=1)
    with gate.slot():
        for _ in range(3):
            with pytest.raises(limits.Busy):
                with gate.slot():
                    pass
            assert gate._waiting == 0


ORIGIN = {"Origin": "http://localhost:5173"}


def test_over_the_limit_is_a_429_that_still_carries_cors(monkeypatch):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/health": 1}, default=100))
    client = TestClient(main.app)
    assert client.get("/health", headers=ORIGIN).status_code == 200
    r = client.get("/health", headers=ORIGIN)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_preflight_allows_the_dev_origin_and_not_others():
    client = TestClient(main.app)
    ok = client.options("/report", headers={**ORIGIN, "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_a_full_compute_gate_returns_503_busy(monkeypatch):
    gate = limits.ComputeGate(slots=1, wait_s=0.01)
    monkeypatch.setattr(main, "gate", gate)
    client = TestClient(main.app)
    with gate.slot():
        r = client.post("/report", json={"holdings": {"AAPL": 1000, "MSFT": 1000}}, headers=ORIGIN)
    assert r.status_code == 503 and r.json() == {"detail": "busy"}
    assert r.headers["retry-after"] == "10"
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_a_429_exposes_retry_after_to_the_browser(monkeypatch):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/health": 1}, default=100))
    client = TestClient(main.app)
    client.get("/health", headers=ORIGIN)
    r = client.get("/health", headers=ORIGIN)
    assert r.status_code == 429
    exposed = [h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")]
    assert "retry-after" in exposed


@pytest.fixture
def saved_portfolio(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTFOLIO_DB", str(tmp_path / "limits.db"))
    body = TestClient(main.app).post(
        "/portfolios", json={"name": "p", "holdings": {"AAPL": 1000, "MSFT": 2000}}).json()
    return body["id"]


def test_analysing_a_saved_portfolio_goes_through_the_compute_gate(monkeypatch, saved_portfolio):
    gate = limits.ComputeGate(slots=1, wait_s=0.01)
    monkeypatch.setattr(main, "gate", gate)
    client = TestClient(main.app)
    with gate.slot():
        busy = client.get(f"/portfolios/{saved_portfolio}", params={"analyze": "true"})
        plain = client.get(f"/portfolios/{saved_portfolio}")
    assert busy.status_code == 503 and busy.json() == {"detail": "busy"}
    assert busy.headers["retry-after"] == "10"
    assert plain.status_code == 200                       # reading it back needs no slot


@pytest.fixture
def analysis_is_stubbed(monkeypatch):
    monkeypatch.setattr(main, "_analyze_or_http_error", lambda holdings, **kw: {"stub": True})


@pytest.mark.parametrize("flag", ["true", "TRUE", "1", "yes", "on"])
def test_analyse_true_on_a_saved_portfolio_spends_the_analyze_limit(
        monkeypatch, saved_portfolio, analysis_is_stubbed, flag):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/analyze": 1}, default=100))
    client = TestClient(main.app)
    url = f"/portfolios/{saved_portfolio}"
    assert client.get(url, params={"analyze": flag}).status_code == 200
    r = client.get(url, params={"analyze": flag})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_a_second_truthy_analyze_value_cannot_hide_behind_a_false_one(
        monkeypatch, saved_portfolio, analysis_is_stubbed):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/analyze": 1}, default=100))
    client = TestClient(main.app)
    url = f"/portfolios/{saved_portfolio}?analyze=false&analyze=true"
    assert client.get(url).status_code == 200
    assert client.get(url).status_code == 429


@pytest.mark.parametrize("query", ["", "?analyze=false", "?analyze=0"])
def test_reading_a_saved_portfolio_without_analysis_keeps_the_general_limit(
        monkeypatch, saved_portfolio, query):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/analyze": 1}, default=100))
    client = TestClient(main.app)
    for _ in range(3):
        assert client.get(f"/portfolios/{saved_portfolio}{query}").status_code == 200
