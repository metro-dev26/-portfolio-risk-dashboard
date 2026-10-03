"""Resolve a user's holdings against the snapshot (or live Yahoo for anything
outside it), then find the date window they can honestly be analysed over.
Pure — no UI."""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import data
from risk_engine.config import (BENCHMARK, CRISES, LIVE_BUDGET_API_S, LIVE_MAX_API,
                                WINDOW_MIN_DAYS, WINDOW_OK_DAYS)

# A live series that stopped this many business days before the snapshot is
# treated as delisted rather than analysed on an old, frozen price.
LIVE_STALE_BUSINESS_DAYS = 5


@dataclass(frozen=True)
class Rejection:
    ticker: str
    amount: float
    reason: str


@dataclass
class Resolution:
    holdings: dict        # accepted ticker -> dollars, input order kept
    prices: pd.DataFrame  # accepted tickers' prices; NaN before each listed
    meta: dict            # ticker -> universe-style metadata
    source: dict          # ticker -> "snapshot" | "live"
    rejected: list        # Rejection, input order

    @property
    def tickers(self):
        return list(self.holdings)


def _has_prices(series):
    return series is not None and series.first_valid_index() is not None


def _stale_reason(series, as_of):
    last = series.last_valid_index()
    if int(np.busday_count(last.date(), as_of)) > LIVE_STALE_BUSINESS_DAYS:
        return f"no recent prices (last close {last.date()}) — possibly delisted"
    return None


def resolve_holdings(holdings, snap, *, live_fetch=None, max_live=LIVE_MAX_API,
                     budget_s=LIVE_BUDGET_API_S, clock=time.monotonic):
    live_fetch = live_fetch or data.fetch_live
    accepted, meta, source, cols, rejected = {}, {}, {}, {}, {}
    started, n_live = clock(), 0
    for raw, amount in holdings.items():
        t = data.normalize_ticker(raw)
        if t in accepted:
            accepted[t] += amount
            continue
        if t in rejected:
            prior = rejected[t]
            rejected[t] = Rejection(t, prior.amount + amount, prior.reason)
            continue
        if t in snap.quarantined:
            rejected[t] = Rejection(t, amount,
                                    f"held back by today's data checks ({snap.quarantined[t]})")
            continue
        if t in snap.universe and t in snap.prices.columns:
            if not _has_prices(snap.prices[t]):
                rejected[t] = Rejection(t, amount, "no price history in today's dataset")
                continue
            accepted[t], meta[t], source[t] = amount, snap.universe[t], "snapshot"
            cols[t] = snap.prices[t]
            continue
        if not data.TICKER_RE.fullmatch(t):
            rejected[t] = Rejection(t, amount, "not a valid ticker symbol")
            continue
        if n_live >= max_live:
            rejected[t] = Rejection(t, amount, f"too many tickers outside the dataset "
                                               f"in one request (max {max_live})")
            continue
        if clock() - started > budget_s:
            rejected[t] = Rejection(t, amount, "lookup time budget used up — try again")
            continue
        n_live += 1
        res = live_fetch(t)
        if res.status != "ok":
            rejected[t] = Rejection(t, amount, res.reason)
            continue
        if not _has_prices(res.prices):
            rejected[t] = Rejection(t, amount, "no price history on Yahoo Finance")
            continue
        stale = _stale_reason(res.prices, snap.as_of)
        if stale:
            rejected[t] = Rejection(t, amount, stale)
            continue
        accepted[t], meta[t], source[t] = amount, res.meta, "live"
        cols[t] = res.prices
    prices = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    return Resolution(accepted, prices, meta, source, list(rejected.values()))


@dataclass
class Window:
    returns: pd.DataFrame  # log returns over the window; holdings (+ benchmark), no gaps
    start: pd.Timestamp    # NaT when no date has a price for every column
    end: pd.Timestamp
    first_dates: dict      # column -> first date with a price; absent for one with none
    n_days: int
    status: str            # "ok" | "short" | "too_short"
    limiting: str          # the column whose first price date sets the start
    message: str


def _refusal(columns, first, limiting, message):
    empty = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([]), dtype=float)
    return Window(empty, pd.NaT, pd.NaT, first, 0, "too_short", limiting, message)


def _no_history_message(bare):
    one = len(bare) == 1
    return (f"{', '.join(bare)} {'has' if one else 'have'} no price history, so there is "
            f"nothing to analyse. Remove {'it' if one else 'them'} and try again.")


def _no_overlap_message(frame, limiting, listed):
    ended = [c for c in frame.columns if frame[c].last_valid_index() < listed]
    why = (f"{', '.join(ended)} stopped trading before {limiting} was listed ({listed.date()})."
           if ended else f"These holdings' price dates never line up after {limiting} was "
                         f"listed ({listed.date()}).")
    return f"No trading day has a price for every holding. {why} Remove one of them to continue."


def _grade(n, limiting, since):
    if n >= WINDOW_OK_DAYS:
        return "ok", ""
    if n >= WINDOW_MIN_DAYS:
        return "short", (f"Results use {n} trading days (since {since}), limited by "
                         f"{limiting}'s listing date. With under two years of shared history, "
                         f"VaR is less reliable.")
    return "too_short", (f"{limiting} has only {n} trading days of shared history (since "
                         f"{since}). That is under a year, too little for these risk numbers "
                         f"to mean anything. Remove {limiting} or analyse it once it has a "
                         f"year of prices.")


def _with_benchmark(held, benchmark):
    """The benchmark joins the alignment as one more column, unless it is held."""
    if benchmark is None:
        return held
    name = BENCHMARK if benchmark.name is None else benchmark.name
    if name in held.columns:
        return held
    return pd.concat([held, benchmark.rename(name)], axis=1).sort_index()


def portfolio_window(prices, tickers, benchmark=None):
    tickers = list(dict.fromkeys(tickers))
    frame = _with_benchmark(prices[tickers], benchmark)
    columns = list(frame.columns)
    if not tickers:
        return _refusal(columns, {}, "", "There are no holdings to analyse.")
    first = {c: frame[c].first_valid_index() for c in columns}
    bare = [c for c in columns if first[c] is None]
    if bare:
        known = {c: d for c, d in first.items() if d is not None}
        return _refusal(columns, known, bare[0], _no_history_message(bare))
    limiting = max(columns, key=lambda c: first[c])
    # Align on dates where every column has a price, THEN difference: a gap is
    # never filled with an invented price, and a multi-day move lands on one row.
    aligned = frame.loc[first[limiting]:].dropna()
    if aligned.empty:
        return _refusal(columns, first, limiting,
                        _no_overlap_message(frame, limiting, first[limiting]))
    returns = np.log(aligned / aligned.shift(1)).iloc[1:].copy()
    n = len(returns)
    status, message = _grade(n, limiting, aligned.index[0].date())
    return Window(returns, aligned.index[0], aligned.index[-1], first, n, status, limiting, message)


def crisis_coverage(prices, tickers, crises=CRISES):
    """For each crisis, which holdings weren't trading yet. A crisis a holding
    missed is reported, never silently dropped."""
    firsts = {t: prices[t].first_valid_index() for t in dict.fromkeys(tickers)}
    out = []
    for label, start, end, context in crises:
        began = pd.Timestamp(start)
        missing = [t for t, first in firsts.items() if first is None or first > began]
        out.append({"label": label, "start": start, "end": end, "context": context,
                    "missing": missing})
    return out
