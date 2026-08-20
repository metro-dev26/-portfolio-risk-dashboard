"""Mean-variance optimization, risk contribution, and covariance estimators. Pure."""
import numpy as np
from scipy.optimize import minimize
from risk_engine.config import TRADING_DAYS


def annualized_mean(log_returns, holdings):
    return (log_returns[holdings].mean() * TRADING_DAYS).to_numpy()


def sample_cov(log_returns, holdings):
    return (log_returns[holdings].cov() * TRADING_DAYS).to_numpy()


def ledoit_wolf_cov(log_returns, holdings):
    """Shrinkage toward a scaled identity (Ledoit & Wolf, 2004). Annualized."""
    X = log_returns[holdings].to_numpy()
    X = X - X.mean(axis=0)
    T, N = X.shape
    S = (X.T @ X) / T
    m = np.trace(S) / N
    d2 = np.sum((S - m * np.eye(N)) ** 2) / N
    b2 = 0.0
    for t in range(T):
        xt = X[t][:, None]
        b2 += np.sum((xt @ xt.T - S) ** 2)
    b2 = b2 / (T ** 2 * N)
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 0.0
    shrunk = delta * m * np.eye(N) + (1 - delta) * S
    return shrunk * TRADING_DAYS


def perf(weights, mu_v, cov_m):
    r = float(weights @ mu_v)
    v = float(np.sqrt(weights @ cov_m @ weights))
    return r, v, (r / v if v > 0 else 0.0)


def risk_contribution(cov_m, weights):
    pv = float(np.sqrt(weights @ cov_m @ weights))
    if pv <= 0:
        return np.asarray(weights, dtype=float)
    mcr = (cov_m @ weights) / pv
    ccr = weights * mcr
    return ccr / ccr.sum()


def _solve(objective, n):
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    w = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons).x
    w = np.clip(w, 0, None)
    return w / w.sum()


def max_sharpe_weights(mu_v, cov_m):
    return _solve(lambda w: -perf(w, mu_v, cov_m)[2], len(mu_v))


def min_variance_weights(cov_m):
    return _solve(lambda w: float(w @ cov_m @ w), cov_m.shape[0])


def efficient_frontier(mu_v, cov_m, n_points=40):
    n = len(mu_v)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    mv_r = float(min_variance_weights(cov_m) @ mu_v)
    vols, rets = [], []
    for tr in np.linspace(mv_r, float(mu_v.max()), n_points):
        c = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},
             {"type": "eq", "fun": lambda w, tr=tr: float(w @ mu_v) - tr})
        res = minimize(lambda w: float(w @ cov_m @ w), w0, method="SLSQP",
                       bounds=bnds, constraints=c)
        if res.success:
            vols.append(float(np.sqrt(res.fun)) * 100)
            rets.append(tr * 100)
    return vols, rets
