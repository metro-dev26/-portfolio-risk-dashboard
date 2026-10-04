"""Pure portfolio-analysis orchestration for the API: resolve holdings against
the snapshot (bounded live lookups for anything outside it), choose the window
the holdings share, run the engine, return a JSON-serializable dict.
No UI, no transport."""
import numpy as np

from risk_engine import backtest, data, metrics, optimize, portfolio
from risk_engine import factors as fac
from risk_engine.config import (BACKTEST_MIN_OBS, BENCHMARK, FACTOR_MIN_OBS,
                                LIVE_BUDGET_API_S, LIVE_MAX_API)


class PortfolioError(ValueError):
    """The portfolio can't be analysed as given; `problems` says why, per ticker."""

    def __init__(self, problems):
        super().__init__("; ".join(f"{p['ticker']}: {p['reason']}" for p in problems))
        self.problems = problems


FACTORS_PROBLEM = "(factors)"
NO_IDENTIFIABLE_SECTOR = "None identifiable"


def _factor_gap_reason(factors, shared, window_days):
    ends = f"ends {factors.index.max().date()}" if len(factors) else "is empty"
    return (f"The Fama-French factor data {ends}, so only {shared} of this portfolio's "
            f"{window_days} trading days have factors; at least {FACTOR_MIN_OBS} are needed "
            f"to estimate factor exposure.")


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
    shared = len(lr.index.intersection(snap.factors.index))
    if shared < FACTOR_MIN_OBS:
        raise PortfolioError([{"ticker": FACTORS_PROBLEM,
                               "reason": _factor_gap_reason(snap.factors, shared, win.n_days)}])
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
        "observations": reg["n_obs"],
    }

    mu = optimize.annualized_mean(lr, tickers)
    cov = optimize.sample_cov(lr, tickers)
    w_ms = optimize.max_sharpe_weights(mu, cov)
    w_mv = optimize.min_variance_weights(cov)
    _, _, cur_sharpe = optimize.perf(weights, mu, cov)
    _, _, ms_sharpe = optimize.perf(w_ms, mu, cov)
    sector_pct = portfolio.sector_mix(tickers, weights, res.meta).sectors
    if sector_pct:
        top_sector = max(sector_pct, key=sector_pct.get)
        top_sector_pct = sector_pct[top_sector]
    else:
        top_sector, top_sector_pct = NO_IDENTIFIABLE_SECTOR, 0.0
    result["optimizer"] = {
        "current_sharpe": float(cur_sharpe),
        "max_sharpe_value": float(ms_sharpe),
        "max_sharpe_weights": {t: float(w) for t, w in zip(tickers, w_ms)},
        "min_variance_weights": {t: float(w) for t, w in zip(tickers, w_mv)},
        "top_sector": top_sector,
        "top_sector_pct": float(top_sector_pct),
    }

    result["data"] = {
        "as_of": snap.as_of,
        "window_start": str(win.start.date()),
        "window_days": int(win.n_days),
        "window_status": win.status,
        "window_message": win.message,
        "crisis_coverage": {c["label"]: c["missing"]
                            for c in portfolio.crisis_coverage(res.prices, tickers)},
    }

    if include_backtest:
        rows = backtest.backtest_var(pr, confidence)
        result["backtest"] = {
            r["method"]: {
                "observations": int(r["observations"]),
                "breaches": int(r["breaches"]),
                "expected": float(r["expected"]),
                "kupiec_p": float(r["kupiec_p"]),
                "christoffersen_p": float(r["christoffersen_p"]),
                # Too few out-of-sample days to grade: report the numbers, not a verdict.
                "passed": bool(r["passed"]) if r["observations"] >= BACKTEST_MIN_OBS else None,
            }
            for r in rows
        }

    return result
