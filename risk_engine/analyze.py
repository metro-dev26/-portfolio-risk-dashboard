"""Pure portfolio-analysis orchestration for the API: resolve holdings against
the snapshot (bounded live lookups for anything outside it), choose the window
the holdings share, run the engine, return a JSON-serializable dict.
No UI, no transport."""
import numpy as np

from risk_engine import backtest, data, metrics, optimize, portfolio
from risk_engine import factors as fac
from risk_engine.config import BENCHMARK, LIVE_BUDGET_API_S, LIVE_MAX_API


class PortfolioError(ValueError):
    """The portfolio can't be analysed as given; `problems` says why, per ticker."""

    def __init__(self, problems):
        super().__init__("; ".join(f"{p['ticker']}: {p['reason']}" for p in problems))
        self.problems = problems


def analyze_portfolio(holdings, *, confidence=0.95, include_backtest=True,
                      snapshot=None, live_fetch=None):
    snap = snapshot or data.load_snapshot()
    res = portfolio.resolve_holdings(holdings, snap, live_fetch=live_fetch or data.fetch_live,
                                     max_live=LIVE_MAX_API, budget_s=LIVE_BUDGET_API_S)
    if res.rejected:
        raise PortfolioError([{"ticker": r.ticker, "reason": r.reason} for r in res.rejected])
    tickers = res.tickers
    win = portfolio.portfolio_window(res.prices, tickers, benchmark=snap.prices[BENCHMARK])
    if win.status == "too_short":
        raise PortfolioError([{"ticker": win.limiting, "reason": win.message}])

    lr = win.returns
    amounts = np.array([res.holdings[t] for t in tickers], dtype=float)
    weights = amounts / amounts.sum()
    pr = metrics.portfolio_returns(lr, tickers, weights)

    hist_var, hist_cvar = metrics.historical_var_cvar(pr, confidence)
    gauss_var, gauss_cvar = metrics.gaussian_var_cvar(pr, confidence)
    result = {
        "metrics": {
            "hist_var": hist_var,
            "hist_cvar": hist_cvar,
            "gaussian_var": gauss_var,
            "gaussian_cvar": gauss_cvar,
            "sharpe": metrics.sharpe_ratio(pr),
            "annual_vol": metrics.annualized_vol(pr),
            "max_drawdown": metrics.max_drawdown(pr),
        },
    }

    reg = fac.factor_regression(pr, snap.factors)
    result["factors"] = {
        "betas": reg["betas"],
        "alpha_annual": reg["alpha_annual"],
        "r2": reg["r2"],
        "variance_split": fac.variance_attribution(reg),
    }

    mu = optimize.annualized_mean(lr, tickers)
    cov = optimize.sample_cov(lr, tickers)
    w_ms = optimize.max_sharpe_weights(mu, cov)
    w_mv = optimize.min_variance_weights(cov)
    _, _, cur_sharpe = optimize.perf(weights, mu, cov)
    _, _, ms_sharpe = optimize.perf(w_ms, mu, cov)
    sector_pct = {}
    for t, w in zip(tickers, weights):
        s = res.meta[t]["sector"]
        sector_pct[s] = sector_pct.get(s, 0.0) + float(w)
    top_sector = max(sector_pct, key=sector_pct.get)
    result["optimizer"] = {
        "current_sharpe": float(cur_sharpe),
        "max_sharpe_value": float(ms_sharpe),
        "max_sharpe_weights": {t: float(w) for t, w in zip(tickers, w_ms)},
        "min_variance_weights": {t: float(w) for t, w in zip(tickers, w_mv)},
        "top_sector": top_sector,
        "top_sector_pct": float(sector_pct[top_sector]),
    }

    result["data"] = {
        "as_of": snap.as_of,
        "window_start": str(win.start.date()),
        "window_days": int(win.n_days),
        "window_status": win.status,
        "crisis_coverage": {c["label"]: c["missing"]
                            for c in portfolio.crisis_coverage(res.prices, tickers)},
    }

    if include_backtest:
        rows = backtest.backtest_var(pr, confidence)
        result["backtest"] = {
            r["method"]: {
                "breaches": int(r["breaches"]),
                "expected": float(r["expected"]),
                "kupiec_p": float(r["kupiec_p"]),
                "christoffersen_p": float(r["christoffersen_p"]),
                "passed": bool(r["passed"]),
            }
            for r in rows
        }

    return result
