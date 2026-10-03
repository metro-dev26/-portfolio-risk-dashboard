import pytest
import urllib.error

from risk_engine import data

SEARCH_PLTR = {"quotes": [{"symbol": "PLTR", "quoteType": "EQUITY", "sector": "Technology",
                           "longname": "Palantir Technologies Inc."}]}
SEARCH_VOO = {"quotes": [{"symbol": "VOO", "quoteType": "ETF", "shortname": "Vanguard S&P 500"}]}


def chart(currency="USD", closes=(10.0, 11.0, None, 12.0)):
    ts = [1735828200, 1735914600, 1736173800, 1736260200][: len(closes)]
    return {"chart": {"result": [{"meta": {"currency": currency}, "timestamp": ts,
                                  "indicators": {"adjclose": [{"adjclose": list(closes)}]}}]}}


def fake_get(search, chart_payload=None, fail=None):
    def get_json(url, timeout):
        if fail:
            raise fail
        return search if "/search" in url else chart_payload
    return get_json


@pytest.fixture(autouse=True)
def allow_live(monkeypatch):
    monkeypatch.delenv("MARKETPLUG_NO_LIVE", raising=False)


@pytest.mark.parametrize("raw,expected", [
    ("brk.b", "BRK-B"), (" aapl ", "AAPL"), ("BF.B", "BF-B"), ("RELIANCE.NS", "RELIANCE.NS"),
])
def test_normalize_ticker(raw, expected):
    assert data.normalize_ticker(raw) == expected


@pytest.mark.parametrize("bad", ["", "AAPL/../X", "<SCRIPT>", "TOOLONGTICKER1", "A B"])
def test_invalid_symbols_never_reach_the_network(bad):
    def explode(url, timeout):
        raise AssertionError("network called for an invalid symbol")
    r = data.fetch_live(bad, get_json=explode)
    assert r.status == "invalid"


def test_ok_stock_returns_prices_and_meta():
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, chart()))
    assert r.status == "ok"
    assert list(r.prices) == [10.0, 11.0, 12.0]          # the null close is dropped, not filled
    assert r.meta["sector"] == "Technology" and r.meta["asset_class"] == "Equity"
    assert r.meta["in_sp500"] is False


def test_unknown_etf_is_labeled_as_a_black_box():
    r = data.fetch_live("VOO", get_json=fake_get(SEARCH_VOO, chart()))
    assert r.meta["type"] == "etf"
    assert r.meta["sector"] == "Fund (holdings unknown)"


def test_search_miss_is_not_found():
    r = data.fetch_live("XYZQQ", get_json=fake_get({"quotes": []}))
    assert r.status == "not_found"


def test_fuzzy_search_hit_for_a_different_symbol_is_not_found():
    r = data.fetch_live("APPL", get_json=fake_get({"quotes": [{"symbol": "AAPL"}]}))
    assert r.status == "not_found"


def test_non_usd_is_rejected_with_the_currency_named():
    search = {"quotes": [{"symbol": "RELIANCE.NS", "quoteType": "EQUITY"}]}
    r = data.fetch_live("RELIANCE.NS", get_json=fake_get(search, chart(currency="INR")))
    assert r.status == "non_usd" and "INR" in r.reason


def test_network_failure_is_unavailable_with_reason():
    r = data.fetch_live("PLTR", get_json=fake_get(None, fail=TimeoutError("slow")))
    assert r.status == "unavailable" and "TimeoutError" in r.reason


def test_malformed_chart_is_unavailable():
    broken = {"chart": {"result": [{"meta": {"currency": "USD"}}]}}
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, broken))
    assert r.status == "unavailable"


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("MARKETPLUG_NO_LIVE", "1")
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, chart()))
    assert r.status == "unavailable" and "disabled" in r.reason


def test_fetch_live_with_non_string_sym():
    """fetch_live never raises; non-string sym should return invalid without network call."""
    def explode(url, timeout):
        raise AssertionError("network called for invalid sym type")
    r = data.fetch_live(5, get_json=explode)
    assert r.status == "invalid"


def test_normalize_ticker_none():
    """normalize_ticker(None) should return empty string to fail validation."""
    assert data.normalize_ticker(None) == ""


def test_normalize_ticker_nan():
    """normalize_ticker(NaN) should return empty string to fail validation."""
    assert data.normalize_ticker(float('nan')) == ""


def test_fetch_live_length_mismatch_chart():
    """Mismatched timestamp and adjclose lengths should be unavailable, not raise."""
    bad_chart = {"chart": {"result": [{
        "meta": {"currency": "USD"},
        "timestamp": [1735828200, 1735914600],  # 2 timestamps
        "indicators": {"adjclose": [{"adjclose": [10.0, 11.0, 12.0]}]}  # 3 closes
    }]}}
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, bad_chart))
    assert r.status == "unavailable" and "unexpected format" in r.reason


def test_fetch_live_list_shaped_json():
    """JSON body that is a list [] instead of dict should be unavailable, not raise."""
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, []))
    assert r.status == "unavailable" and "unexpected format" in r.reason


def test_zero_and_negative_closes_dropped():
    """Zero and negative closes should be dropped along with NaN."""
    good_chart = chart(closes=(10.0, 0.0, -1.0, 12.0))
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, good_chart))
    assert r.status == "ok"
    assert list(r.prices) == [10.0, 12.0]


def test_timestamps_sorted_after_dedup():
    """Out-of-order timestamps should be sorted after dedup."""
    ts = [1735914600, 1735828200, 1736173800]  # Not in order: 2nd, 1st, 3rd
    out_of_order_chart = {"chart": {"result": [{
        "meta": {"currency": "USD"},
        "timestamp": ts,
        "indicators": {"adjclose": [{"adjclose": [11.0, 10.0, 12.0]}]}
    }]}}
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, out_of_order_chart))
    assert r.status == "ok"
    assert list(r.prices.index) == sorted(r.prices.index)


def test_ticker_regex_rejects_newline():
    """TICKER_RE should reject symbols with trailing newline."""
    assert data.TICKER_RE.match("AAPL\n") is None
    assert data.TICKER_RE.match("AAPL") is not None


def test_http_error_includes_status_code():
    """HTTPError exceptions should include the status code in the reason."""
    def get_json_http_error(url, timeout):
        raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
    r = data.fetch_live("PLTR", get_json=get_json_http_error)
    assert r.status == "unavailable" and "HTTP 429" in r.reason


def test_fetch_live_with_integer_sym_no_network_call():
    """Integer sym should return invalid without making network call."""
    def explode(url, timeout):
        raise AssertionError("network called for integer sym")
    r = data.fetch_live(5, get_json=explode)
    assert r.status == "invalid" and r.ticker == "5"


def test_live_meta_with_list_quotetype():
    """live_meta should handle non-string quoteType without raising."""
    r = data.fetch_live("PLTR", get_json=fake_get(
        {"quotes": [{"symbol": "PLTR", "quoteType": ["EQUITY"]}]},
        chart()
    ))
    assert r.status == "ok"


def test_search_response_as_list():
    """Search response that is a list should return human-readable error."""
    r = data.fetch_live("PLTR", get_json=fake_get([]))
    assert r.status == "unavailable" and "search results came back in an unexpected format" in r.reason


def test_scalar_adjclose():
    """Scalar adjclose instead of list should return unavailable, not raise."""
    bad_chart = {"chart": {"result": [{
        "meta": {"currency": "USD"},
        "timestamp": [1735828200, 1735914600, 1736173800],
        "indicators": {"adjclose": [{"adjclose": 5.0}]}  # Scalar, not list
    }]}}
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, bad_chart))
    assert r.status == "unavailable"


def test_infinite_closes_dropped():
    """Infinite closes should be dropped along with non-positive values."""
    inf_chart = chart(closes=(10.0, float('inf'), 12.0))
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, inf_chart))
    assert r.status == "ok"
    assert list(r.prices) == [10.0, 12.0]


def test_bad_start_parameter_no_network():
    """Bad start parameter should return unavailable with 'start' in reason, no network call."""
    def explode(url, timeout):
        raise AssertionError("network called with bad start")
    r = data.fetch_live("PLTR", start=None, get_json=explode)
    assert r.status == "unavailable" and "start" in r.reason
