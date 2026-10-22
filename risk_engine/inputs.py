"""Holdings + snapshot -> everything every section reads: resolved holdings,
weights, the shared window, daily log returns for the portfolio and the S&P 500.
Built once per request. Pure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import data, metrics, portfolio
from risk_engine.config import BENCHMARK, LIVE_BUDGET_API_S, LIVE_MAX_API, MIN_HOLDINGS

HOLDINGS_PROBLEM = "(holdings)"


class PortfolioError(ValueError):
    """The portfolio can't be analysed as given; `problems` says why, per ticker."""

    def __init__(self, problems, rejected=()):
        super().__init__("; ".join(f"{p['ticker']}: {p['reason']}" for p in problems))
        self.problems = problems
        self.rejected = list(rejected)   # [{ticker, amount, reason}] dropped before the refusal


def _rejected(res):
    return [{"ticker": r.ticker, "amount": r.amount, "reason": r.reason} for r in res.rejected]


@dataclass(frozen=True)
class Inputs:
    snap: object
    res: portfolio.Resolution
    tickers: list
    weights: np.ndarray
    value: float
    window: portfolio.Window
    lr: pd.DataFrame     # holdings + benchmark daily log returns over the window
    pr: pd.Series        # portfolio daily log returns
    spy_r: pd.Series     # benchmark daily log returns, same dates as pr


def prepare(holdings, snap, *, live_fetch=None, max_live=LIVE_MAX_API,
            budget_s=LIVE_BUDGET_API_S, allow_rejected=False):
    res = portfolio.resolve_holdings(holdings, snap, live_fetch=live_fetch or data.fetch_live,
                                     max_live=max_live, budget_s=budget_s)
    if res.rejected and not allow_rejected:
        raise PortfolioError([{"ticker": r.ticker, "reason": r.reason} for r in res.rejected],
                             _rejected(res))
    tickers = res.tickers
    if len(tickers) < MIN_HOLDINGS:
        raise PortfolioError([{"ticker": HOLDINGS_PROBLEM,
                               "reason": f"at least {MIN_HOLDINGS} holdings with a price history "
                                         f"are needed; {len(tickers)} could be used"}],
                             _rejected(res))
    win = portfolio.portfolio_window(res.prices, tickers, benchmark=snap.prices[BENCHMARK])
    if win.status == "too_short":
        raise PortfolioError([{"ticker": win.limiting, "reason": win.message}], _rejected(res))
    amounts = np.array([res.holdings[t] for t in tickers], dtype=float)
    value = float(amounts.sum())
    weights = amounts / value
    lr = win.returns
    return Inputs(snap, res, tickers, weights, value, win, lr,
                  metrics.portfolio_returns(lr, tickers, weights), lr[BENCHMARK])
