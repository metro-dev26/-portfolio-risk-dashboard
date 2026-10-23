"""Portfolio risk metrics. Pure functions over log-return series."""
import numpy as np
from scipy.stats import norm, t as _t
from risk_engine.config import TRADING_DAYS


def portfolio_returns(log_returns, holdings, weights):
    """Daily log return of a portfolio rebalanced to `weights` every day: log(sum(w * exp(r))).
    The weighted sum of log returns is not it: it understates a diversified portfolio's growth
    by about half the gap between the holdings' average variance and the portfolio's."""
    ret_sel = log_returns[holdings].dropna()
    return np.log((np.exp(ret_sel) * weights).sum(axis=1))


def historical_var_cvar(port_returns, conf):
    var = float(np.percentile(port_returns, (1 - conf) * 100))
    cvar = float(port_returns[port_returns <= var].mean())
    return var, cvar


def gaussian_var_cvar(port_returns, conf):
    mu, std = port_returns.mean(), port_returns.std()
    var = float(mu + std * norm.ppf(1 - conf))
    cvar = float(mu - std * norm.pdf(norm.ppf(1 - conf)) / (1 - conf))
    return var, cvar


def annualized_vol(port_returns):
    return float(port_returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(port_returns):
    mu, std = port_returns.mean(), port_returns.std()
    return float((mu / std) * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0


def drawdown_series(port_returns):
    wealth = np.exp(port_returns.cumsum())
    return wealth / wealth.cummax() - 1.0


def max_drawdown(port_returns):
    return float(drawdown_series(port_returns).min())


def student_t_var_cvar(port_returns, conf):
    """VaR/CVaR from a fitted Student-t. Falls back to historical if df <= 2."""
    alpha = 1 - conf
    df, loc, scale = _t.fit(port_returns.to_numpy())
    if df <= 2:
        return historical_var_cvar(port_returns, conf)
    var = float(_t.ppf(alpha, df, loc, scale))
    q = _t.ppf(alpha, df)  # standardized quantile (negative)
    es_std = -(df + q ** 2) / (df - 1) * _t.pdf(q, df) / alpha
    cvar = float(loc + scale * es_std)
    return var, cvar


def to_dollars(log_r, value):
    """The dollar change a log return means on `value`: exp(r) - 1, not r."""
    return float(np.expm1(log_r) * value)


def return_histogram(port_returns, bins=80):
    """Daily returns binned as a probability density (area sums to 1)."""
    density, edges = np.histogram(np.asarray(port_returns, dtype=float), bins=bins, density=True)
    return {"edges": edges.tolist(), "density": density.tolist()}


def correlation(log_returns, tickers):
    tickers = list(tickers)
    return {"tickers": tickers, "matrix": log_returns[tickers].corr().to_numpy().tolist()}
