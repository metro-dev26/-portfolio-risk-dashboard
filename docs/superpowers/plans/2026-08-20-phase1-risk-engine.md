# Phase 1 — Credible Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the inline math in `app.py` into a pure, tested `risk_engine/` package, then add VaR backtesting (Kupiec + Christoffersen), Student-t VaR/CVaR, and Ledoit-Wolf shrinkage — with a new "Model Validation" UI section — while the live Streamlit URL keeps working.

**Architecture:** Extract inline computations into pure functions (no Streamlit imports), guarded every step by a golden-master fixture captured from the current code so behavior is provably unchanged. New methods land as pure functions with property-based tests, then wire into a thin Streamlit skin. The engine is the single source of truth the UI, tests, and (later phases) the API all import.

**Tech Stack:** Python 3, numpy, pandas, scipy.stats (`t`, `norm`, `chi2`), scipy.optimize (`minimize`), Streamlit + Plotly for UI, pytest for tests.

## Global Constraints

- **`risk_engine/` is pure** — no `import streamlit`, no Plotly, no UI, no I/O side effects in engine functions (except `data.py`, whose job IS I/O).
- **No new heavy dependencies** — no scikit-learn, no statsmodels. Hand-roll Ledoit-Wolf; use `scipy.stats`/`numpy` for everything else.
- **Behavior-preserving refactor** — every extracted function must reproduce the golden-master fixture. The live URL must not break.
- **Convention: log returns** — the whole app uses log returns (`np.log(prices/prices.shift(1))`). All engine functions consume log returns unless stated.
- **Standard deviation ddof** — the current app uses `pandas.Series.std()` (ddof=1). Extracted functions MUST match to preserve golden values.
- **Annualization** — `TRADING_DAYS = 252`. Volatility ×√252, mean ×252, covariance ×252, exactly as the current code does.
- **Commits** — frequent, one per task step group. **No `Co-Authored-By` trailer** (this is Tam's portfolio repo). All work on branch `phase1-risk-engine`.
- **Framing** — educational risk tool, never "financial advice."

---

### Task 1: Branch, package skeleton, config module, CI

**Files:**
- Create: `risk_engine/__init__.py`
- Create: `risk_engine/config.py`
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: nothing.
- Produces: `risk_engine.config.TICKERS: list[str]`, `SECTOR: dict[str,str]`, `BONDS: set[str]`, `ASSET_CLASS: dict[str,str]`, `TRADING_DAYS: int`.

- [ ] **Step 1: Create the branch**

Run: `git checkout -b phase1-risk-engine`
Expected: `Switched to a new branch 'phase1-risk-engine'`

- [ ] **Step 2: Create the dev-dependency file**

Create `requirements-dev.txt`:

```
pytest>=8.0
```

- [ ] **Step 3: Create the package skeleton**

Create `risk_engine/__init__.py` (empty file).
Create `tests/__init__.py` (empty file).

- [ ] **Step 4: Write the failing test for config**

Create `tests/test_config.py`:

```python
from risk_engine import config


def test_tickers_are_unique_and_nonempty():
    assert len(config.TICKERS) == 24
    assert len(set(config.TICKERS)) == len(config.TICKERS)


def test_bonds_are_subset_of_tickers():
    assert config.BONDS.issubset(set(config.TICKERS))


def test_every_ticker_has_a_sector_and_asset_class():
    for t in config.TICKERS:
        assert t in config.SECTOR
        assert config.ASSET_CLASS[t] in ("Bond", "Equity")


def test_trading_days_constant():
    assert config.TRADING_DAYS == 252
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'risk_engine.config'`

- [ ] **Step 6: Create config.py**

Create `risk_engine/config.py` (values copied verbatim from `app.py:187-206`):

```python
"""Universe definition and shared constants for the risk engine."""

TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
           "JPM", "GS", "BAC", "MS", "XOM", "CVX", "COP",
           "JNJ", "PFE", "UNH", "ABBV", "TSLA", "WMT", "BA",
           "TLT", "IEF", "AGG", "LQD"]

SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology",
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials", "MS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "BA": "Industrials",
    "TLT": "Govt Bonds", "IEF": "Govt Bonds", "AGG": "Aggregate Bonds", "LQD": "Corp Bonds",
}

BONDS = {"TLT", "IEF", "AGG", "LQD"}
ASSET_CLASS = {t: ("Bond" if t in BONDS else "Equity") for t in TICKERS}

TRADING_DAYS = 252
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (4 passed)

- [ ] **Step 8: Add CI workflow**

Create `.github/workflows/ci.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: python -m pytest -v
```

- [ ] **Step 9: Commit**

```bash
git add risk_engine/__init__.py risk_engine/config.py tests/__init__.py tests/test_config.py requirements-dev.txt .github/workflows/ci.yml
git commit -m "Add risk_engine package skeleton, config module, and CI"
```

---

### Task 2: Golden-master reference fixture

**Purpose:** Capture the current app's exact numeric outputs for a fixed portfolio BEFORE touching anything, so later refactors can prove they changed nothing. The reference is produced by running the current inline formulas verbatim.

**Fixed reference portfolio:** holdings `["AAPL", "MSFT", "JPM", "XOM", "TLT"]`, equal dollars ($20,000 each → equal weights), `conf = 0.95`, data = `prices.csv` snapshot.

**Files:**
- Create: `tools/snapshot_reference.py`
- Create: `tests/fixtures/golden.json` (generated by the script)

**Interfaces:**
- Consumes: `risk_engine.config`.
- Produces: `tests/fixtures/golden.json` with keys — `hist_var`, `hist_cvar`, `gauss_var`, `gauss_cvar`, `ann_vol`, `sharpe`, `max_dd`, `beta`, `risk_pct` (list, order = holdings), `max_sharpe`, `min_var_sharpe`.

- [ ] **Step 1: Write the reference-snapshot script**

Create `tools/snapshot_reference.py` (formulas copied verbatim from `app.py`; this is the anchor for behavior-preservation):

```python
"""One-off: capture current app outputs for a fixed portfolio as the golden master.
Run once from the repo root: python tools/snapshot_reference.py
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

prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True).ffill().dropna()
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
```

- [ ] **Step 2: Run the script to generate the fixture**

Run: `python tools/snapshot_reference.py`
Expected: prints a JSON block and writes `tests/fixtures/golden.json`. Sanity-check: `hist_var` is a small negative number (≈ -0.01 to -0.02), `sharpe` is a positive float, `risk_pct` is a list of 5 numbers summing to ~1.0.

- [ ] **Step 3: Commit the fixture and the tool**

```bash
git add tools/snapshot_reference.py tests/fixtures/golden.json
git commit -m "Capture golden-master reference fixture for fixed portfolio"
```

---

### Task 3: Pure data loader (`risk_engine/data.py`)

**Files:**
- Create: `risk_engine/data.py`
- Create: `tests/test_data.py`
- Modify: `app.py:210-263` (replace inline `load()` body with a thin cached wrapper over the engine)

**Interfaces:**
- Consumes: `risk_engine.config.TICKERS`.
- Produces: `load_prices(prefer_live: bool = True) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]` returning `(prices, log_returns, source_label)`. Snapshot path is used when `prefer_live=False` or live fails.

- [ ] **Step 1: Write the failing test**

Create `tests/test_data.py`:

```python
from risk_engine.data import load_prices


def test_snapshot_load_shape():
    prices, lr, source = load_prices(prefer_live=False)
    assert prices is not None
    assert prices.shape[0] > 2000          # ~8.5 years of daily rows
    assert "SPY" in prices.columns
    assert lr.shape[0] == prices.shape[0] - 1   # one row lost to differencing
    assert "snapshot" in source


def test_log_returns_are_finite():
    _, lr, _ = load_prices(prefer_live=False)
    import numpy as np
    assert np.isfinite(lr.to_numpy()).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'risk_engine.data'`

- [ ] **Step 3: Create data.py**

Create `risk_engine/data.py` (logic lifted from `app.py:210-263`, `@st.cache_data` removed, `SNAPSHOT` path made repo-root-relative, `prefer_live` flag added):

```python
"""Price loading. Live pull via Python's own SSL stack (survives HTTPS-inspecting
networks that break curl-based clients); frozen snapshot as fallback. Pure — no UI."""
import os
import numpy as np
import pandas as pd
from risk_engine.config import TICKERS

_SNAPSHOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prices.csv")


def _fetch(sym):
    import urllib.request, json, datetime
    p1 = int(datetime.datetime(2018, 1, 1).timestamp())
    p2 = int(datetime.datetime.now().timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read())
    r = d["chart"]["result"][0]
    idx = pd.to_datetime([datetime.date.fromtimestamp(t) for t in r["timestamp"]])
    cl = r["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(cl, index=idx, name=sym)


def load_prices(prefer_live=True):
    """Return (prices, log_returns, source_label)."""
    prices, source = None, None

    if prefer_live:
        try:
            series = {}
            for t in TICKERS + ["SPY"]:
                try:
                    s = _fetch(t)
                    if s.notna().sum() > 500:
                        series[t] = s
                except Exception:
                    pass
            if len(series) >= 2:
                prices = pd.concat(series.values(), axis=1).ffill().dropna()
                source = "Yahoo Finance · live"
        except Exception:
            prices = None

    if prices is None or prices.shape[1] < 2:
        if os.path.exists(_SNAPSHOT):
            prices = pd.read_csv(_SNAPSHOT, index_col=0, parse_dates=True).ffill().dropna()
            source = f"frozen snapshot · {prices.index.max().date()}"
        else:
            return None, None, None

    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr, source
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Wire app.py to use the engine loader**

In `app.py`, replace the entire inline `load()` function (`app.py:210-263`) with a thin cached wrapper, and remove the now-unused `SNAPSHOT` constant at `app.py:207`:

```python
from risk_engine.data import load_prices


@st.cache_data(ttl=3600, show_spinner=False)
def load():
    return load_prices(prefer_live=True)
```

- [ ] **Step 6: Verify the app still boots**

Run: `python -m streamlit run app.py --server.port 8801 --server.headless true` (Ctrl-C after it prints the local URL with no traceback).
Expected: starts cleanly, no exceptions in the terminal.

- [ ] **Step 7: Commit**

```bash
git add risk_engine/data.py tests/test_data.py app.py
git commit -m "Extract pure data loader into risk_engine.data"
```

---

### Task 4: Core metrics — portfolio returns, historical & Gaussian VaR/CVaR, supporting stats

**Files:**
- Create: `risk_engine/metrics.py`
- Create: `tests/conftest.py`
- Create: `tests/test_metrics.py`
- Modify: `app.py:326-346` (replace inline math with engine calls)

**Interfaces:**
- Consumes: `risk_engine.data.load_prices` (in tests, via fixture).
- Produces:
  - `portfolio_returns(log_returns: pd.DataFrame, holdings: list[str], weights: np.ndarray) -> pd.Series`
  - `historical_var_cvar(port_returns: pd.Series, conf: float) -> tuple[float, float]`
  - `gaussian_var_cvar(port_returns: pd.Series, conf: float) -> tuple[float, float]`
  - `annualized_vol(port_returns: pd.Series) -> float`
  - `sharpe_ratio(port_returns: pd.Series) -> float`
  - `drawdown_series(port_returns: pd.Series) -> pd.Series`
  - `max_drawdown(port_returns: pd.Series) -> float`

- [ ] **Step 1: Write the shared fixtures**

Create `tests/conftest.py`:

```python
import json
import numpy as np
import pandas as pd
import pytest
from risk_engine.data import load_prices

HOLDINGS = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
CONF = 0.95


@pytest.fixture(scope="session")
def market():
    prices, lr, _ = load_prices(prefer_live=False)
    return prices, lr


@pytest.fixture(scope="session")
def ref_weights():
    return np.ones(len(HOLDINGS)) / len(HOLDINGS)


@pytest.fixture(scope="session")
def golden():
    with open("tests/fixtures/golden.json") as f:
        return json.load(f)


@pytest.fixture
def normal_returns():
    """A long, well-behaved normal return series with known parameters."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.01, 50_000))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_metrics.py`:

```python
import numpy as np
from scipy.stats import norm
from risk_engine import metrics
from tests.conftest import HOLDINGS, CONF


def test_historical_var_of_normal_matches_theory(normal_returns):
    var, cvar = metrics.historical_var_cvar(normal_returns, 0.95)
    # 95% VaR of N(0, 0.01) ≈ -1.645 * 0.01
    assert abs(var - (-1.645 * 0.01)) < 5e-4
    assert cvar < var  # expected shortfall is worse than the threshold


def test_gaussian_var_matches_closed_form(normal_returns):
    var, _ = metrics.gaussian_var_cvar(normal_returns, 0.95)
    mu, sd = normal_returns.mean(), normal_returns.std()
    assert abs(var - (mu + sd * norm.ppf(0.05))) < 1e-9


def test_metrics_reproduce_golden(market, ref_weights, golden):
    _, lr = market
    pr = metrics.portfolio_returns(lr, HOLDINGS, ref_weights)
    h_var, h_cvar = metrics.historical_var_cvar(pr, CONF)
    g_var, g_cvar = metrics.gaussian_var_cvar(pr, CONF)
    assert abs(h_var - golden["hist_var"]) < 1e-9
    assert abs(h_cvar - golden["hist_cvar"]) < 1e-9
    assert abs(g_var - golden["gauss_var"]) < 1e-9
    assert abs(g_cvar - golden["gauss_cvar"]) < 1e-9
    assert abs(metrics.annualized_vol(pr) - golden["ann_vol"]) < 1e-9
    assert abs(metrics.sharpe_ratio(pr) - golden["sharpe"]) < 1e-9
    assert abs(metrics.max_drawdown(pr) - golden["max_dd"]) < 1e-9
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'risk_engine.metrics'`

- [ ] **Step 4: Create metrics.py**

Create `risk_engine/metrics.py` (formulas match `app.py:327-346` exactly, incl. `pandas.std()` ddof=1):

```python
"""Portfolio risk metrics. Pure functions over log-return series."""
import numpy as np
import pandas as pd
from scipy.stats import norm
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Wire app.py to use the metrics engine**

In `app.py`, replace the inline risk-engine block (`app.py:326-346`, from `ret_sel = lr[selected]...` through `max_dd = drawdown.min()`) with:

```python
from risk_engine import metrics

ret_sel = lr[selected].dropna()
pr = metrics.portfolio_returns(lr, selected, weights)
mu, std = pr.mean(), pr.std()
h_var, h_cvar = metrics.historical_var_cvar(pr, conf)
g_var, g_cvar = metrics.gaussian_var_cvar(pr, conf)
gap = (h_var - g_var) * port_val
ann_vol = metrics.annualized_vol(pr)
sharpe = metrics.sharpe_ratio(pr)
drawdown = metrics.drawdown_series(pr)
max_dd = metrics.max_drawdown(pr)
```

- [ ] **Step 7: Verify the app still boots and numbers are unchanged**

Run: `python -m streamlit run app.py --server.port 8801 --server.headless true` (Ctrl-C after clean boot). Load the default portfolio in a browser and confirm the headline metrics render without error.
Expected: no traceback; metrics display as before.

- [ ] **Step 8: Commit**

```bash
git add risk_engine/metrics.py tests/conftest.py tests/test_metrics.py app.py
git commit -m "Extract core metrics into risk_engine.metrics with golden-master coverage"
```

---

### Task 5: Student-t (fat-tailed) VaR/CVaR

**Files:**
- Modify: `risk_engine/metrics.py` (add one function)
- Modify: `tests/test_metrics.py` (add tests)

**Interfaces:**
- Consumes: `port_returns: pd.Series`.
- Produces: `student_t_var_cvar(port_returns: pd.Series, conf: float) -> tuple[float, float]`. Falls back to historical when fitted `df <= 2` (ES undefined).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_metrics.py`:

```python
def test_student_t_has_fatter_tail_than_gaussian_on_heavy_data():
    from scipy.stats import t as tdist
    import pandas as pd
    rng = np.random.default_rng(1)
    heavy = pd.Series(tdist.rvs(df=3, size=200_000, random_state=rng) * 0.01)
    t_var, _ = metrics.student_t_var_cvar(heavy, 0.99)
    g_var, _ = metrics.gaussian_var_cvar(heavy, 0.99)
    # at 99%, a t-fit sees the fat tail Gaussian misses → more extreme (more negative)
    assert t_var < g_var


def test_student_t_cvar_worse_than_var():
    from scipy.stats import t as tdist
    import pandas as pd
    rng = np.random.default_rng(2)
    heavy = pd.Series(tdist.rvs(df=4, size=100_000, random_state=rng) * 0.01)
    var, cvar = metrics.student_t_var_cvar(heavy, 0.95)
    assert cvar < var
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -k student_t -v`
Expected: FAIL — `AttributeError: module 'risk_engine.metrics' has no attribute 'student_t_var_cvar'`

- [ ] **Step 3: Implement student_t_var_cvar**

Add to `risk_engine/metrics.py` (add `from scipy.stats import t as _t` at the top with the other imports):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -k student_t -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add risk_engine/metrics.py tests/test_metrics.py
git commit -m "Add Student-t fat-tailed VaR/CVaR to risk_engine.metrics"
```

---

### Task 6: Optimizer extraction + Ledoit-Wolf shrinkage

**Files:**
- Create: `risk_engine/optimize.py`
- Create: `tests/test_optimize.py`
- Modify: `app.py:664-826` (risk-contribution + optimizer blocks call the engine)

**Interfaces:**
- Consumes: `log_returns: pd.DataFrame`, `holdings: list[str]`, `weights: np.ndarray`.
- Produces:
  - `sample_cov(log_returns: pd.DataFrame, holdings: list[str]) -> np.ndarray` (annualized)
  - `ledoit_wolf_cov(log_returns: pd.DataFrame, holdings: list[str]) -> np.ndarray` (annualized, shrunk)
  - `annualized_mean(log_returns: pd.DataFrame, holdings: list[str]) -> np.ndarray`
  - `perf(weights, mu_v, cov_m) -> tuple[float, float, float]` (return, vol, sharpe)
  - `risk_contribution(cov_m, weights) -> np.ndarray` (fractions summing to 1)
  - `max_sharpe_weights(mu_v, cov_m) -> np.ndarray`
  - `min_variance_weights(cov_m) -> np.ndarray`
  - `efficient_frontier(mu_v, cov_m, n_points=40) -> tuple[list, list]` (vols%, rets%)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_optimize.py`:

```python
import numpy as np
from risk_engine import optimize
from tests.conftest import HOLDINGS


def test_ledoit_wolf_is_symmetric_and_psd(market):
    _, lr = market
    cov = optimize.ledoit_wolf_cov(lr, HOLDINGS)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10   # PSD


def test_min_variance_not_worse_than_equal_weight(market):
    _, lr = market
    cov = optimize.sample_cov(lr, HOLDINGS)
    w_mv = optimize.min_variance_weights(cov)
    eq = np.ones(len(HOLDINGS)) / len(HOLDINGS)
    v_mv = float(np.sqrt(w_mv @ cov @ w_mv))
    v_eq = float(np.sqrt(eq @ cov @ eq))
    assert v_mv <= v_eq + 1e-9
    assert abs(w_mv.sum() - 1.0) < 1e-6
    assert (w_mv >= -1e-9).all()


def test_risk_contribution_sums_to_one(market, ref_weights):
    _, lr = market
    cov = optimize.sample_cov(lr, HOLDINGS)
    rc = optimize.risk_contribution(cov, ref_weights)
    assert abs(rc.sum() - 1.0) < 1e-9


def test_optimizer_reproduces_golden(market, ref_weights, golden):
    _, lr = market
    mu_v = optimize.annualized_mean(lr, HOLDINGS)
    cov = optimize.sample_cov(lr, HOLDINGS)
    w_ms = optimize.max_sharpe_weights(mu_v, cov)
    w_mv = optimize.min_variance_weights(cov)
    assert abs(optimize.perf(w_ms, mu_v, cov)[2] - golden["max_sharpe"]) < 1e-4
    assert abs(optimize.perf(w_mv, mu_v, cov)[2] - golden["min_var_sharpe"]) < 1e-4
    rc = optimize.risk_contribution(cov, ref_weights)
    assert np.allclose(rc, golden["risk_pct"], atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'risk_engine.optimize'`

- [ ] **Step 3: Create optimize.py**

Create `risk_engine/optimize.py` (extraction of `app.py:670-796` plus hand-rolled Ledoit-Wolf shrinkage-to-identity, Ledoit & Wolf 2004):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Wire app.py risk-contribution + optimizer to the engine**

In `app.py`, replace the inline risk-contribution math (`app.py:670-677`) with:

```python
from risk_engine import optimize
_cov_rc = optimize.sample_cov(lr, selected)
risk_pct = optimize.risk_contribution(_cov_rc, weights)
```

Then replace the optimizer setup + solves (`app.py:718-796`, the `mu_v`/`cov_m`/`_perf`/`_cons`/`_bnds`/`_w0` block, the two `minimize` calls, and the frontier loop) with calls to `optimize.annualized_mean`, `optimize.sample_cov` (or `optimize.ledoit_wolf_cov` when the shrinkage toggle is on), `optimize.perf`, `optimize.max_sharpe_weights`, `optimize.min_variance_weights`, and `optimize.efficient_frontier`. Add a sidebar toggle:

```python
use_shrinkage = st.sidebar.checkbox(
    "Use Ledoit-Wolf shrinkage (robust covariance)", value=False,
    help="Shrinks the noisy sample covariance toward a stable target so the "
         "optimizer stops chasing estimation error. Standard practice on real desks.")
cov_m = optimize.ledoit_wolf_cov(lr, selected) if use_shrinkage else optimize.sample_cov(lr, selected)
mu_v = optimize.annualized_mean(lr, selected)
```

Keep the existing Plotly rendering; only the numeric sources change. `_perf(...)` call sites become `optimize.perf(..., mu_v, cov_m)`.

- [ ] **Step 6: Verify the app still boots and the optimizer renders**

Run: `python -m streamlit run app.py --server.port 8801 --server.headless true` (Ctrl-C after clean boot). In the browser: confirm the efficient frontier, the three Sharpe cards, and the reallocation table render; toggle the shrinkage checkbox and confirm the frontier updates without error.
Expected: no traceback; frontier + cards render; toggle works.

- [ ] **Step 7: Commit**

```bash
git add risk_engine/optimize.py tests/test_optimize.py app.py
git commit -m "Extract optimizer to risk_engine.optimize and add Ledoit-Wolf shrinkage"
```

---

### Task 7: VaR backtesting engine — Kupiec + Christoffersen (the headline)

**Files:**
- Create: `risk_engine/backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `port_returns: pd.Series`.
- Produces:
  - `rolling_var_breaches(port_returns, conf, window=250, method="historical") -> dict` with keys `dates` (list), `var` (np.ndarray), `realized` (np.ndarray), `breach` (np.ndarray of 0/1).
  - `kupiec_pof(n_obs: int, n_breaches: int, conf: float) -> tuple[float, float]` — (LR statistic, p-value), chi²(1).
  - `christoffersen_cc(breaches: np.ndarray, conf: float) -> tuple[float, float]` — (LR_cc statistic, p-value), chi²(2).
  - `backtest_var(port_returns, conf, window=250, methods=("historical", "gaussian")) -> list[dict]` — one row per method: `method`, `breaches`, `expected`, `kupiec_p`, `christoffersen_p`, `passed` (bool).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
import numpy as np
import pandas as pd
from risk_engine import backtest


def _normal_series(n=6000, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, sd, n))


def test_correctly_specified_var_passes_kupiec():
    """A stationary normal series backtested with historical VaR should breach
    at ~5% and Kupiec should NOT reject."""
    r = _normal_series()
    res = backtest.rolling_var_breaches(r, conf=0.95, window=500, method="historical")
    n, x = len(res["breach"]), int(res["breach"].sum())
    rate = x / n
    assert 0.03 < rate < 0.07
    _, p = backtest.kupiec_pof(n, x, 0.95)
    assert p > 0.05   # fail to reject correct model


def test_too_tight_var_is_rejected_by_kupiec():
    """If we deliberately claim a far-too-tight VaR, breaches explode and Kupiec rejects."""
    n, x = 1000, 200   # 20% breaches vs 5% expected
    lr, p = backtest.kupiec_pof(n, x, 0.95)
    assert p < 0.01


def test_christoffersen_detects_clustering():
    """Breaches that arrive in one contiguous block violate independence."""
    b = np.zeros(1000, dtype=int)
    b[100:150] = 1   # 50 breaches, all clustered
    _, p_clustered = backtest.christoffersen_cc(b, 0.95)
    # a spread-out arrangement of the same count should look far more independent
    rng = np.random.default_rng(3)
    b2 = np.zeros(1000, dtype=int)
    b2[rng.choice(1000, size=50, replace=False)] = 1
    _, p_spread = backtest.christoffersen_cc(b2, 0.95)
    assert p_clustered < p_spread


def test_backtest_var_summary_shape():
    r = _normal_series()
    rows = backtest.backtest_var(r, conf=0.95, window=500,
                                 methods=("historical", "gaussian"))
    assert {row["method"] for row in rows} == {"historical", "gaussian"}
    for row in rows:
        assert 0.0 <= row["kupiec_p"] <= 1.0
        assert isinstance(row["passed"], bool)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'risk_engine.backtest'`

- [ ] **Step 3: Create backtest.py**

Create `risk_engine/backtest.py`:

```python
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
            "method": method, "breaches": x, "expected": round(n * (1 - conf), 1),
            "kupiec_p": kp, "christoffersen_p": cp,
            "passed": bool(kp > 0.05 and cp > 0.05),
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add risk_engine/backtest.py tests/test_backtest.py
git commit -m "Add VaR backtesting engine: Kupiec POF and Christoffersen coverage tests"
```

---

### Task 8: Model Validation UI section

**Files:**
- Modify: `app.py` (add a new section after the correlation matrix, before risk contribution — around `app.py:646`)
- Create: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `risk_engine.backtest.backtest_var`, `risk_engine.backtest.rolling_var_breaches`.
- Produces: UI only.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_app_smoke.py`:

```python
from streamlit.testing.v1 import AppTest


def test_app_runs_without_exception():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception


def test_model_validation_section_present():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Model Validation" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: FAIL on `test_model_validation_section_present` — "Model Validation" not found (the first test may already pass).

- [ ] **Step 3: Add the Model Validation section to app.py**

Insert after the correlation-matrix block (after `app.py:662`, before the risk-contribution section). Uses the backtest engine; caches the (slow) rolling computation:

```python
# ── MODEL VALIDATION: DID THE VAR ACTUALLY HOLD? ──────────────
from risk_engine import backtest

st.markdown("<div class='sec'>Model validation — did the risk numbers actually hold up?</div>",
            unsafe_allow_html=True)
st.caption(f"A VaR estimate is only worth anything if it's been backtested. We roll a "
           f"250-day window across all history, and each day ask: did the real loss breach "
           f"the VaR? A {int(conf*100)}% model should be breached about {int(round((1-conf)*100))}% "
           f"of the time — no more, and not in clusters. "
           f"Kupiec tests the rate; Christoffersen tests that breaches don't bunch up in crises.")


@st.cache_data(ttl=3600, show_spinner=False)
def _run_backtest(returns_values, conf, window):
    s = pd.Series(returns_values)
    rows = backtest.backtest_var(s, conf, window, methods=("historical", "gaussian"))
    hist = backtest.rolling_var_breaches(s, conf, window, "historical")
    return rows, hist


bt_rows, bt_hist = _run_backtest(pr.to_numpy(), conf, 250)

rows_html = ""
for row in bt_rows:
    verdict = "PASS" if row["passed"] else "FAIL"
    vcol = "#05d69e" if row["passed"] else "#ff3d5a"
    rows_html += (
        f"<tr>"
        f"<td style='padding:10px 14px;color:#8a9bb8;text-transform:capitalize;'>{row['method']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['breaches']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;color:#6a849e;'>{row['expected']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['kupiec_p']:.3f}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['christoffersen_p']:.3f}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;font-weight:600;color:{vcol};'>{verdict}</td>"
        f"</tr>")
st.markdown(f"""
<table style='width:100%;border-collapse:collapse;background:var(--card);
              border:1px solid var(--border);border-radius:10px;overflow:hidden;'>
    <tr style='background:#0c1220;'>
        <th style='padding:10px 14px;text-align:left;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Method</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Breaches</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Expected</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Kupiec p</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Christoffersen p</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Verdict</th>
    </tr>
    {rows_html}
</table>""", unsafe_allow_html=True)

# Breach-timeline chart for the historical method
bt_dates = bt_hist["dates"]
breach_mask = bt_hist["breach"].astype(bool)
mvfig = go.Figure()
mvfig.add_trace(go.Scatter(x=bt_dates, y=bt_hist["realized"] * 100, mode="lines",
                           line=dict(color="#4d9fff", width=1), name="Daily return"))
mvfig.add_trace(go.Scatter(x=bt_dates, y=bt_hist["var"] * 100, mode="lines",
                           line=dict(color="#ffb347", width=1.5, dash="dash"), name="Historical VaR"))
mvfig.add_trace(go.Scatter(x=[d for d, m in zip(bt_dates, breach_mask) if m],
                           y=[r * 100 for r, m in zip(bt_hist["realized"], breach_mask) if m],
                           mode="markers", marker=dict(color="#ff3d5a", size=6, symbol="x"),
                           name="Breach"))
mvfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                    font=dict(color="#dde4f0", family="DM Sans"),
                    xaxis=dict(gridcolor="#1c2d44"),
                    yaxis=dict(title="Daily return (%)", gridcolor="#1c2d44"),
                    height=360, margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(mvfig, width="stretch")

_hist_row = next(r for r in bt_rows if r["method"] == "historical")
_gauss_row = next(r for r in bt_rows if r["method"] == "gaussian")


def _mv_rate(row):
    return (f"gets the breach <em>rate</em> right (Kupiec p={row['kupiec_p']:.2f})"
            if row["kupiec_p"] > 0.05
            else f"breaks the expected rate (Kupiec p={row['kupiec_p']:.2f})")


def _mv_clust(row):
    return ("with breaches staying independent" if row["christoffersen_p"] > 0.05
            else f"but breaches <strong>cluster in crises</strong> "
                 f"(Christoffersen p={row['christoffersen_p']:.3f})")


_both_cluster = (_hist_row["christoffersen_p"] <= 0.05
                 and _gauss_row["christoffersen_p"] <= 0.05)
_closing = (
    "Both models get the frequency about right yet fail the independence test — real losses "
    "bunch together in crises (COVID 2020, the 2022 bear), the exact stretches a static VaR "
    "never sees coming. That gap is the honest limit of any single-number risk measure, and "
    "naming it is the whole job."
    if _both_cluster else
    "A model earns trust only by passing both — the right rate <em>and</em> independent breaches. "
    "That is the test a risk desk runs before it believes any VaR at all.")
st.markdown(f"""
<div class='insight warn'>
    <div class='insight-icon'>🧪</div>
    <div class='insight-text'>
        <strong>The honest scoreboard.</strong> Over the full history, historical VaR was breached
        <strong>{_hist_row['breaches']}</strong> times vs <strong>{_hist_row['expected']}</strong> expected,
        and Gaussian <strong>{_gauss_row['breaches']}</strong> times. Historical {_mv_rate(_hist_row)}
        {_mv_clust(_hist_row)}; Gaussian {_mv_rate(_gauss_row)} {_mv_clust(_gauss_row)}.
        Kupiec asks whether the breach <em>rate</em> matches the confidence level; Christoffersen asks
        whether breaches arrive <em>independently</em> or bunch together. {_closing}
    </div>
</div>""", unsafe_allow_html=True)
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Visually verify in the browser**

Run: `python -m streamlit run app.py --server.port 8801 --server.headless true`. In the browser, scroll to "Model validation" — confirm the table (2 method rows with PASS/FAIL), the breach-timeline chart with red × markers on the big down days, and the scoreboard callout all render.
Expected: section renders; the Gaussian row typically shows more breaches than historical (the pedagogical payoff).

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Add Model Validation UI section with VaR backtest table and breach timeline"
```

---

### Task 9: README methodology update + full-suite verification

**Files:**
- Modify: `README.md` (Methodology section)
- Modify: `requirements.txt` (confirm no new runtime deps needed — scipy already present)

**Interfaces:**
- Consumes: nothing new.
- Produces: documentation only.

- [ ] **Step 1: Confirm no new runtime dependency crept in**

Run: `python -c "import scipy, numpy, pandas, streamlit, plotly; print('runtime deps OK')"`
Expected: `runtime deps OK` (scipy already in requirements.txt; no sklearn/statsmodels added).

- [ ] **Step 2: Add a Model Validation subsection to README.md**

In `README.md`, under the Methodology section, add (objective/impersonal voice — no "I", no "you", per repo convention):

```markdown
### Model Validation

Value-at-Risk is only meaningful once it has been backtested. A 250-day rolling
window is walked across the full history; each day the realized loss is compared
to the VaR estimate to flag a breach. Two standard tests then judge the model:

- **Kupiec POF** (unconditional coverage) — is the breach *rate* consistent with
  the stated confidence level?
- **Christoffersen** (conditional coverage) — do breaches arrive independently, or
  cluster together in crises?

Historical and Gaussian VaR are scored side by side, exposing where the
normal-distribution assumption underestimates real tail risk. Fat-tailed
Student-t VaR/CVaR and Ledoit-Wolf covariance shrinkage are also available, the
latter producing more robust optimizer weights by shrinking the noisy sample
covariance toward a stable target.
```

- [ ] **Step 3: Run the entire test suite**

Run: `python -m pytest -v`
Expected: ALL PASS — config (4), data (2), metrics (5), optimize (4), backtest (4), app smoke (2). No failures, no errors.

- [ ] **Step 4: Final app boot check**

Run: `python -m streamlit run app.py --server.port 8801 --server.headless true` (Ctrl-C after clean boot).
Expected: no traceback; full page renders end to end.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt
git commit -m "Document Model Validation methodology in README"
```

- [ ] **Step 6: Phase 1 done — do NOT merge yet**

Phase 1 is complete on branch `phase1-risk-engine`. Merge to `main` (which auto-deploys to Streamlit Cloud) is a separate, deliberate step — use the `superpowers:finishing-a-development-branch` skill, and only after a human eyeballs the live-preview render. Capture a fresh screenshot of the Model Validation section for the README/LinkedIn post at that point.

---

## Self-Review

**1. Spec coverage:**
- Refactor into pure `risk_engine/` package → Tasks 1, 3, 4, 6. ✅
- Golden-master behavior-preservation → Task 2 + assertions in Tasks 4, 6. ✅
- VaR backtesting (Kupiec + Christoffersen) → Task 7 + UI Task 8. ✅
- Student-t fat-tailed VaR → Task 5. ✅
- Ledoit-Wolf shrinkage → Task 6. ✅
- Model Validation UI section → Task 8. ✅
- Tests from zero + CI → Tasks 1–8. ✅
- README model-validation subsection → Task 9. ✅
- Live URL protected / no merge until verified → Tasks 3/4/6/8 boot checks + Task 9 Step 6. ✅
- No sklearn/statsmodels → hand-rolled LW (Task 6), scipy-only stats (Tasks 5, 7); confirmed Task 9 Step 1. ✅
- Stress/simulate/contribution left inline until needed → only risk-contribution numeric extracted (used by golden); stress & Monte Carlo untouched. ✅

**2. Placeholder scan:** No TBD/TODO. Every code step contains complete code; every test step contains real assertions; every run step has an exact command and expected output.

**3. Type consistency:** `port_returns` is a `pd.Series` throughout metrics/backtest. `cov_m`/`mu_v` are `np.ndarray` produced by `optimize.sample_cov`/`annualized_mean` and consumed by `perf`/`max_sharpe_weights`/`min_variance_weights`/`efficient_frontier`/`risk_contribution`. `backtest_var` returns `list[dict]` with keys consumed verbatim in Task 8. `rolling_var_breaches` returns a dict with `dates`/`var`/`realized`/`breach`, all consumed in Task 8. Names are consistent across tasks.

**Note on golden tolerances:** metrics assertions use `1e-9` (pure formula, deterministic); optimizer Sharpe assertions use `1e-4` (SLSQP is iterative — same machine/version is deterministic but cross-environment drift is possible). If a golden optimizer assertion fails only in CI, loosen to `1e-3` — do not regenerate the fixture to force a pass.
