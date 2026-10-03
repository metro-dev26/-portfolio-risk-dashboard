"""Turn what people paste, type or export from a broker into {ticker: dollars}.
Nothing is dropped silently: every skipped line or merged duplicate becomes a
note the UI shows. Pure — no UI."""
import csv
import io
import math
import re
from dataclasses import dataclass, field

from risk_engine.data import TICKER_RE, normalize_ticker

TICKER_HEADERS = ("symbol", "ticker")
VALUE_HEADERS = ("market value", "current value", "value", "amount")
QTY_HEADERS = ("quantity", "shares", "qty")

_AMOUNT = re.compile(r"^\(?-?\$?-?[\d,]*\.?\d+\)?$")


@dataclass
class ParseResult:
    holdings: dict = field(default_factory=dict)   # ticker -> dollars, duplicates summed
    notes: list = field(default_factory=list)      # skipped input and merges, for the user


def parse_amount(text):
    """'$25,000.50' -> 25000.5; '(1,234)' and '-5' -> negative; anything else -> None."""
    t = str(text).strip().replace(" ", "")
    if not t or not _AMOUNT.match(t):
        return None
    value = float(re.sub(r"[^\d.]", "", t))
    return -value if (t.startswith("(") or "-" in t) else value


class _Collector:
    def __init__(self):
        self.holdings, self.notes, self.counts = {}, [], {}

    def add(self, raw_ticker, amount, where):
        t = normalize_ticker(raw_ticker)
        if not TICKER_RE.fullmatch(t):
            self.notes.append(f"{where}: '{raw_ticker}' is not a valid ticker — skipped")
        elif amount is None:
            self.notes.append(f"{where}: no dollar amount for {t} — skipped")
        elif amount < 0:
            self.notes.append(f"{where}: {t} has a negative amount "
                              f"(short positions are not supported) — skipped")
        elif amount == 0:
            self.notes.append(f"{where}: {t} has a $0 amount — skipped")
        else:
            self.counts[t] = self.counts.get(t, 0) + 1
            self.holdings[t] = self.holdings.get(t, 0.0) + float(amount)

    def result(self):
        merges = [f"{t} appeared {n} times — merged into ${self.holdings[t]:,.0f}"
                  for t, n in self.counts.items() if n > 1]
        return ParseResult(self.holdings, self.notes + merges)


def parse_paste(text):
    c = _Collector()
    for i, line in enumerate(str(text).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,;]+", line, maxsplit=1) if p]
        if len(parts) != 2:
            c.notes.append(f"line {i}: expected 'TICKER AMOUNT', got '{line}' — skipped")
            continue
        c.add(parts[0], parse_amount(parts[1]), f"line {i}")
    return c.result()


def _find(header, wanted):
    by_name = {h.strip().lower(): h for h in header if h}
    return next((by_name[w] for w in wanted if w in by_name), None)


def parse_csv(text, *, last_price=None):
    """Broker exports often put account details above the header row, so the
    header is the first row that names a ticker column. `last_price(ticker)` values
    quantity-only files; it may return None when no price is known."""
    rows = list(csv.reader(io.StringIO(text.lstrip("﻿"))))
    hdr_i = next((i for i, r in enumerate(rows) if _find(r, TICKER_HEADERS)), None)
    if hdr_i is None:
        return ParseResult({}, ["no ticker column found — expected a header named Symbol or Ticker"])
    header = rows[hdr_i]
    t_col = header.index(_find(header, TICKER_HEADERS))
    v_name, q_name = _find(header, VALUE_HEADERS), _find(header, QTY_HEADERS)
    if v_name is None and (q_name is None or last_price is None):
        return ParseResult({}, ["no value column found — expected Market Value, Current Value, "
                                "Value or Amount (or Quantity)"])
    c = _Collector()
    if v_name is None:
        c.notes.append("no value column — each holding valued as quantity × latest close")
    col = header.index(v_name if v_name is not None else q_name)
    for i, r in enumerate(rows[hdr_i + 1:], hdr_i + 2):
        if not any(cell.strip() for cell in r):
            continue
        if len(r) <= max(t_col, col):
            c.notes.append(f"row {i}: too few columns — skipped")
            continue
        sym, number = r[t_col].strip(), parse_amount(r[col])
        if v_name is None and number is not None and number > 0:
            price = last_price(normalize_ticker(sym))
            if price is None:
                c.notes.append(f"row {i}: no price for {normalize_ticker(sym)} to value its shares — skipped")
                continue
            number *= price
        c.add(sym, number, f"row {i}")
    return c.result()


def from_rows(rows):
    """Rows from the editable holdings table. A row with no ticker and no amount is
    an empty line the user added, not a mistake, so it is skipped quietly."""
    c = _Collector()
    for i, (ticker, amount) in enumerate(rows, 1):
        blank_ticker = ticker is None or str(ticker).strip() == ""
        blank_amount = amount is None or (isinstance(amount, float) and math.isnan(amount))
        if blank_ticker and blank_amount:
            continue
        c.add("" if blank_ticker else ticker, None if blank_amount else float(amount),
              f"table row {i}")
    return c.result()


def encode_share(holdings):
    return ",".join(f"{t}:{round(a)}" for t, a in holdings.items())


def decode_share(param):
    if not param:
        return {}
    return parse_paste("\n".join(p.replace(":", " ", 1) for p in param.split(","))).holdings
