"""Pure portfolio-analysis orchestration shared by the API (and, later, the UI).
Loads the committed price snapshot ONLY — never a live fetch in a request path —
runs the engine, and returns a JSON-serializable dict. No UI, no transport."""
import numpy as np

from risk_engine import data, metrics, optimize, backtest
from risk_engine import factors as fac
from risk_engine.config import SECTOR


def analyze_portfolio(holdings, *, confidence=0.95, include_backtest=True):
    tickers = list(holdings.keys())
    amounts = np.array([holdings[t] for t in tickers], dtype=float)
    weights = amounts / amounts.sum()

    # Snapshot only — prefer_live=True would hit Yahoo per ticker in the request path.
    _, lr, _ = data.load_prices(prefer_live=False)
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

    f = fac.load_factors()
    reg = fac.factor_regression(pr, f)
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
        s = SECTOR.get(t, "Other")
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
