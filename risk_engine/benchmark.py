"""Portfolio vs the S&P 500 (SPY) over the same days. Pure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import metrics
from risk_engine.config import TRADING_DAYS


@dataclass(frozen=True)
class Comparison:
    rows: list                    # label, portfolio, benchmark, higher_is_better, kind
    beta: float
    growth_portfolio: pd.Series   # value of `value` invested, daily-rebalanced
    growth_benchmark: pd.Series


def beta(pr, spy_r):
    var = float(np.var(spy_r, ddof=1))
    return float(np.cov(pr, spy_r, ddof=1)[0, 1] / var) if var > 0 else 0.0


def compare(pr, spy_r, value):
    common = pr.index.intersection(spy_r.index)
    p, b = pr.loc[common], spy_r.loc[common]
    rows = [
        {"label": "Annualized return", "portfolio": float(p.mean() * TRADING_DAYS),
         "benchmark": float(b.mean() * TRADING_DAYS), "higher_is_better": True, "kind": "pct"},
        {"label": "Annualized volatility (risk)", "portfolio": metrics.annualized_vol(p),
         "benchmark": metrics.annualized_vol(b), "higher_is_better": False, "kind": "pct"},
        {"label": "Sharpe ratio", "portfolio": metrics.sharpe_ratio(p),
         "benchmark": metrics.sharpe_ratio(b), "higher_is_better": True, "kind": "ratio"},
        # Higher (less negative) is better.
        {"label": "Maximum drawdown", "portfolio": metrics.max_drawdown(p),
         "benchmark": metrics.max_drawdown(b), "higher_is_better": True, "kind": "pct"},
    ]
    return Comparison(rows, beta(p, b), value * np.exp(p.cumsum()), value * np.exp(b.cumsum()))
