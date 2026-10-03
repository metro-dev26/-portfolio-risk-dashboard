import math

import pytest

from risk_engine import importer


@pytest.mark.parametrize("text,value", [
    ("10000", 10000.0), ("$25,000", 25000.0), ("8,500.50", 8500.5), ("$ 1 000", 1000.0),
    ("(1,234.00)", -1234.0), ("-5", -5.0), ("abc", None), ("", None), ("1.2.3", None),
])
def test_parse_amount(text, value):
    assert importer.parse_amount(text) == value


def test_paste_forgives_case_dollars_commas_tabs_and_semicolons():
    r = importer.parse_paste("aapl 10000\nVOO, $25,000\nmsft\t8,500.50\nTLT;5000\n\n")
    assert r.holdings == {"AAPL": 10000.0, "VOO": 25000.0, "MSFT": 8500.5, "TLT": 5000.0}
    assert r.notes == []


def test_paste_merges_duplicates_and_says_so():
    r = importer.parse_paste("AAPL 10000\naapl 8000")
    assert r.holdings == {"AAPL": 18000.0}
    assert r.notes == ["AAPL appeared 2 times — merged into $18,000"]


def test_paste_class_share_dot_becomes_dash():
    assert importer.parse_paste("brk.b 5000").holdings == {"BRK-B": 5000.0}


@pytest.mark.parametrize("line,fragment", [
    ("AAPL", "expected 'TICKER AMOUNT'"),
    ("AAPL lots", "no dollar amount"),
    ("AAPL 0", "$0 amount"),
    ("AAPL (1,234.00)", "short positions are not supported"),
    ("TSLA -500", "short positions are not supported"),
    ("<img src=x> 100", "not a valid ticker"),
])
def test_paste_skips_bad_lines_with_a_reason(line, fragment):
    r = importer.parse_paste(line)
    assert r.holdings == {}
    assert len(r.notes) == 1 and fragment in r.notes[0]


SCHWAB_LIKE = (
    '"Positions for account Individual ...123 as of 10/03/2026"\n'
    '\n'
    '"Symbol","Description","Quantity","Price","Market Value"\n'
    '"AAPL","APPLE INC","50","$230.00","$11,500.00"\n'
    '"VOO","VANGUARD S&P 500","20","$600.00","$12,000.00"\n'
    '"Cash & Cash Investments","--","--","--","$1,250.00"\n'
    '"Account Total","--","--","--","$24,750.00"\n'
)


def test_csv_finds_header_below_account_info_and_skips_junk_rows():
    r = importer.parse_csv(SCHWAB_LIKE)
    assert r.holdings == {"AAPL": 11500.0, "VOO": 12000.0}
    assert sum("not a valid ticker" in n for n in r.notes) == 2


def test_csv_handles_byte_order_mark_and_lowercase_headers():
    r = importer.parse_csv("﻿ticker,amount\nmsft,9000\n".lstrip("﻿"))
    assert r.holdings == {"MSFT": 9000.0}


def test_csv_quantity_only_is_valued_at_latest_close_and_labeled():
    r = importer.parse_csv("Symbol,Quantity\nAAPL,10\nZZZZ,5\n",
                           last_price=lambda t: {"AAPL": 200.0}.get(t))
    assert r.holdings == {"AAPL": 2000.0}
    assert any("quantity × latest close" in n for n in r.notes)
    assert any("no price for ZZZZ" in n for n in r.notes)


def test_csv_without_ticker_column_names_the_expected_headers():
    r = importer.parse_csv("Name,Value\nApple,100\n")
    assert r.holdings == {} and "Symbol or Ticker" in r.notes[0]


def test_csv_without_value_or_quantity_column():
    r = importer.parse_csv("Symbol,Description\nAAPL,Apple\n")
    assert r.holdings == {} and "Market Value" in r.notes[0]


def test_from_rows_ignores_blank_editor_rows():
    r = importer.from_rows([("AAPL", 1000.0), (None, math.nan), ("", None), ("msft", 500)])
    assert r.holdings == {"AAPL": 1000.0, "MSFT": 500.0} and r.notes == []


def test_share_link_round_trip():
    h = {"AAPL": 10000.4, "BRK-B": 5000.0}
    assert importer.encode_share(h) == "AAPL:10000,BRK-B:5000"
    assert importer.decode_share("AAPL:10000,BRK-B:5000") == {"AAPL": 10000.0, "BRK-B": 5000.0}
    assert importer.decode_share("") == {}
    assert importer.decode_share("garbage") == {}
