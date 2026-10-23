import asyncio
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


def test_a_declared_body_over_the_cap_is_a_413_that_still_carries_cors():
    r = TestClient(main.app).post("/report", content=b" " * (64 * 1024 + 1),
                                  headers={**ORIGIN, "Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_import_has_a_larger_cap_than_every_other_path():
    client = TestClient(main.app)
    body = b" " * (100 * 1024)
    headers = {"Content-Type": "application/json"}
    assert client.post("/report", content=body, headers=headers).status_code == 413
    assert client.post("/portfolios", content=body, headers=headers).status_code == 413
    assert client.post("/import", content=body, headers=headers).status_code == 422  # reached the schema
    over = client.post("/import", content=b" " * (1_500_000 + 1), headers=headers)
    assert over.status_code == 413


def test_a_chunked_body_over_the_cap_is_a_413_that_still_carries_cors():
    """No Content-Length to check (httpx sends a generator as Transfer-Encoding: chunked)."""
    chunks = (b" " * 1024 for _ in range(200))
    r = TestClient(main.app).post("/report", content=chunks,
                                  headers={**ORIGIN, "Content-Type": "application/json"})
    assert r.status_code == 413
    assert r.json() == {"detail": "request body too large"}
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_a_chunked_body_is_not_read_past_the_cap():
    """Drives the real app one 1 KB message at a time; TestClient would buffer the lot first."""
    pulled = []

    async def receive():
        pulled.append(1)
        return {"type": "http.request", "body": b" " * 1024, "more_body": len(pulled) < 2_000}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
             "scheme": "http", "path": "/report", "raw_path": b"/report", "query_string": b"",
             "root_path": "", "client": ("203.0.113.9", 5000), "server": ("testserver", 80),
             "headers": [(b"origin", b"http://localhost:5173"), (b"content-type", b"application/json"),
                         (b"transfer-encoding", b"chunked")]}
    asyncio.run(main.app(scope, receive, send))
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    assert (b"access-control-allow-origin", b"http://localhost:5173") in start["headers"]
    assert len(pulled) == 65                  # 64 KB is 64 chunks; the 65th crosses the cap


def test_a_declared_body_over_the_cap_is_refused_without_reading_any_of_it():
    reads = []

    async def receive():
        reads.append(1)
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    async def never_called(scope, receive, send):
        raise AssertionError("the app must not run")

    scope = {"type": "http", "method": "POST", "path": "/report", "headers": [(b"content-length", b"999999999")]}
    asyncio.run(limits.BodyLimit(never_called, caps={}, default=65_536)(scope, receive, send))
    assert reads == []
    assert sent[0]["status"] == 413


def test_a_body_exactly_at_the_cap_goes_through():
    got = []

    async def app(scope, receive, send):
        got.append((await receive())["body"])

    async def receive():
        return {"type": "http.request", "body": b"x" * 100, "more_body": False}

    async def send(message):
        pass

    scope = {"type": "http", "method": "POST", "path": "/x", "headers": [(b"content-length", b"100")]}
    asyncio.run(limits.BodyLimit(app, caps={}, default=100)(scope, receive, send))
    assert got == [b"x" * 100]
