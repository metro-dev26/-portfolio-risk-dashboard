"""One-off: capture app outputs for a fixed portfolio as the golden master, computed from
the frozen price fixture tests/fixtures/snapshot/prices.csv.gz.
Run from the repo root: python tools/snapshot_reference.py
Regenerate ONLY if the reference behavior is intentionally changed."""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

HOLDINGS = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
CONF = 0.95
TRADING_DAYS = 252

prices = pd.read_csv("tests/fixtures/snapshot/prices.csv.gz", index_col=0, parse_dates=True).ffill().dropna()
lr = np.log(prices / prices.shift(1)).dropna()

weights = np.ones(len(HOLDINGS)) / len(HOLDINGS)
ret_sel = lr[HOLDINGS].dropna()
pr = (ret_sel * weights).sum(axis=1)
mu, std = pr.mean(), pr.std()

h_var = np.percentile(pr, (1 - CONF) * 100)
h_cvar = pr[pr <= h_var].mean()
g_var = mu + std * norm.ppf(1 - CONF)
g_cvar = mu - std * norm.pdf(norm.ppf(1 - CONF)) / (1 - CONF)
ann_vol = std * np.sqrt(TRADING_DAYS)
sharpe = (mu / std) * np.sqrt(TRADING_DAYS) if std > 0 else 0.0
wealth = np.exp(pr.cumsum())
max_dd = (wealth / wealth.cummax() - 1.0).min()

spy = lr["SPY"]
common = pr.index.intersection(spy.index)
prc, spyc = pr.loc[common], spy.loc[common]
beta = float(np.cov(prc, spyc)[0, 1] / np.var(spyc))

cov_rc = (lr[HOLDINGS].cov() * TRADING_DAYS).to_numpy()
pv = float(np.sqrt(weights @ cov_rc @ weights))
mcr = (cov_rc @ weights) / pv
ccr = weights * mcr
risk_pct = (ccr / ccr.sum()).tolist()

mu_v = (lr[HOLDINGS].mean() * TRADING_DAYS).to_numpy()
cov_m = (lr[HOLDINGS].cov() * TRADING_DAYS).to_numpy()
n = len(HOLDINGS)
cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
bnds = tuple((0.0, 1.0) for _ in range(n))
w0 = np.ones(n) / n


def perf(w):
    r = float(w @ mu_v); v = float(np.sqrt(w @ cov_m @ w))
    return r, v, (r / v if v > 0 else 0.0)


w_ms = minimize(lambda w: -perf(w)[2], w0, method="SLSQP", bounds=bnds, constraints=cons).x
w_mv = minimize(lambda w: float(w @ cov_m @ w), w0, method="SLSQP", bounds=bnds, constraints=cons).x
w_ms = np.clip(w_ms, 0, None); w_ms /= w_ms.sum()
w_mv = np.clip(w_mv, 0, None); w_mv /= w_mv.sum()

golden = {
    "hist_var": float(h_var), "hist_cvar": float(h_cvar),
    "gauss_var": float(g_var), "gauss_cvar": float(g_cvar),
    "ann_vol": float(ann_vol), "sharpe": float(sharpe), "max_dd": float(max_dd),
    "beta": beta, "risk_pct": risk_pct,
    "max_sharpe": perf(w_ms)[2], "min_var_sharpe": perf(w_mv)[2],
}

os.makedirs("tests/fixtures", exist_ok=True)
with open("tests/fixtures/golden.json", "w") as f:
    json.dump(golden, f, indent=2)
print(json.dumps(golden, indent=2))
