"""Portfolio risk metrics. Pure functions over log-return series."""
import numpy as np
import pandas as pd
from scipy.stats import norm, t as _t
from risk_engine.config import TRADING_DAYS


def portfolio_returns(log_returns, holdings, weights):
    ret_sel = log_returns[holdings].dropna()
    return (ret_sel * weights).sum(axis=1)


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
