"""VaR backtesting: rolling breaches + Kupiec POF and Christoffersen coverage tests.
Pure — no UI. References: Kupiec (1995), Christoffersen (1998)."""
import numpy as np
from scipy.stats import norm, chi2, t as _t


def rolling_var_breaches(port_returns, conf, window=250, method="historical"):
    r = np.asarray(port_returns, dtype=float)
    idx = list(port_returns.index) if hasattr(port_returns, "index") else list(range(len(r)))
    alpha = 1 - conf
    dates, var, realized, breach = [], [], [], []
    for i in range(window, len(r)):
        w = r[i - window:i]
        if method == "historical":
            v = np.percentile(w, alpha * 100)
        elif method == "gaussian":
            v = w.mean() + w.std() * norm.ppf(alpha)
        elif method == "student_t":
            df, loc, scale = _t.fit(w)
            v = _t.ppf(alpha, df, loc, scale) if df > 2 else np.percentile(w, alpha * 100)
        else:
            raise ValueError(f"unknown method: {method}")
        dates.append(idx[i]); var.append(v); realized.append(r[i])
        breach.append(1 if r[i] < v else 0)
    return {"dates": dates, "var": np.array(var),
            "realized": np.array(realized), "breach": np.array(breach, dtype=int)}


def kupiec_pof(n_obs, n_breaches, conf):
    """Unconditional coverage LR test. Chi-square(1)."""
    p = 1 - conf
    n, x = n_obs, n_breaches
    pi = x / n if n else 0.0

    def _ln(v):
        return np.log(v) if v > 0 else 0.0

    ln_null = (n - x) * _ln(1 - p) + x * _ln(p)
    ln_alt = (n - x) * _ln(1 - pi) + x * _ln(pi)
    lr = -2 * (ln_null - ln_alt)
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen_cc(breaches, conf):
    """Conditional coverage LR = Kupiec + independence. Chi-square(2)."""
    b = np.asarray(breaches, dtype=int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        prev, cur = b[i - 1], b[i]
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    def _ln(v):
        return np.log(v) if v > 0 else 0.0

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if len(b) > 1 else 0.0
    ln_null = (n00 + n10) * _ln(1 - pi) + (n01 + n11) * _ln(pi)
    ln_alt = (n00 * _ln(1 - pi01) + n01 * _ln(pi01)
              + n10 * _ln(1 - pi11) + n11 * _ln(pi11))
    lr_ind = -2 * (ln_null - ln_alt)

    x = int(b.sum())
    lr_pof, _ = kupiec_pof(len(b), x, conf)
    lr_cc = lr_pof + lr_ind
    return float(lr_cc), float(1 - chi2.cdf(lr_cc, 2))


def backtest_var(port_returns, conf, window=250, methods=("historical", "gaussian")):
    rows = []
    for method in methods:
        res = rolling_var_breaches(port_returns, conf, window, method)
        n, x = len(res["breach"]), int(res["breach"].sum())
        _, kp = kupiec_pof(n, x, conf)
        _, cp = christoffersen_cc(res["breach"], conf)
        rows.append({
            "method": method, "observations": n, "breaches": x,
            "expected": round(n * (1 - conf), 1),
            "kupiec_p": kp, "christoffersen_p": cp,
            "passed": bool(kp > 0.05 and cp > 0.05),
        })
    return rows
