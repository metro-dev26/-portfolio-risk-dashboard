"""Replay today's holdings, buy-and-hold at today's weights, through real crash
windows. Every crisis comes back with a status, so one the portfolio can't be
replayed through is reported, never silently dropped. Pure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import portfolio
from risk_engine.config import CRISES

MIN_CRISIS_DAYS = 5


@dataclass(frozen=True)
class CrisisResult:
    label: str
    context: str
    start: str
    end: str
    status: str            # "ok" | "not_covered" | "too_few_days"
    missing: list          # holdings not yet trading when the crisis began
    total: float | None
    worst_day: float | None
    max_dd: float | None
    path: pd.Series | None  # portfolio value, starts at 1.0


def replay(prices, tickers, weights, crises=CRISES):
    tickers, weights = list(tickers), np.asarray(weights, dtype=float)
    out = []
    for c in portfolio.crisis_coverage(prices, tickers, crises):
        head = (c["label"], c["context"], c["start"], c["end"])
        if c["missing"]:
            out.append(CrisisResult(*head, "not_covered", c["missing"], None, None, None, None))
            continue
        wp = prices.loc[c["start"]:c["end"], tickers].dropna()
        if len(wp) < MIN_CRISIS_DAYS:
            out.append(CrisisResult(*head, "too_few_days", [], None, None, None, None))
            continue
        path = (wp / wp.iloc[0] * weights).sum(axis=1)
        out.append(CrisisResult(*head, "ok", [], float(path.iloc[-1] - 1.0),
                                float(path.pct_change().min()),
                                float((path / path.cummax() - 1.0).min()), path))
    return out
