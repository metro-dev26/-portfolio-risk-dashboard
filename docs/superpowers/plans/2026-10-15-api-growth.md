# Project A — Grow the API: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One `POST /report` call returns everything the 13 dashboard sections need, plus a 1–99 risk score for the portfolio and for the S&P 500. The math comes out of `app.py` into tested `risk_engine` functions that both the API and Streamlit use.

**Architecture:**
- New pure engine modules:
  - `inputs` builds the shared pipeline.
  - `simulate` holds the Monte Carlo and the score.
  - `stress`, `benchmark` and `report` each cover their own area.
- `report.report_from_inputs()` builds one JSON-safe dict.
- FastAPI serves it through `/report`, with CORS, rate limits and a compute gate.
- Streamlit calls the same `report_from_inputs()` and only lays the numbers out.

**Tech stack:** Python 3.11, numpy 2.2, pandas 2.2, scipy 1.15, FastAPI 0.133, pydantic 2.12, Streamlit, pytest. **No new dependencies.**

**Spec:** `docs/superpowers/specs/2026-10-15-api-growth-design.md`. Read it before starting any task.

## Global Constraints

**Work environment**
- Repo: `C:\Users\sujan\portfolio-risk-dashboard`. All work happens on branch `api-growth` (Task 1 creates it). **Never push. Never merge.**
- Commit messages: plain imperative sentence, no prefix like `feat:`. **No `Co-Authored-By` trailer** (Tam's repo rule).
- The shell is Git Bash. Run tests with `python -m pytest`. The full suite took 170 s at the 370-test baseline.
- Tests read the frozen fixture snapshot (`tests/fixtures/snapshot`, as of 2026-06-18, 25 tickers, no GLD) with live lookups off (`MARKETPLUG_NO_LIVE=1`, set in `tests/conftest.py`).

**Engine rules**
- `risk_engine` stays pure: no streamlit, plotly, fastapi or network imports.
- Every fraction stays a fraction (0.05 means 5%). Only dollars are dollars. Nothing in the engine or API is pre-multiplied by 100.
- Dollar figures from log returns use `metrics.to_dollars` (`expm1`), never `log_r * value`.
- Monte Carlo constants: `N_SIMS = 10_000`, `SEED = 42`, horizons `{126, 252, 504, 756}`, `BAND_STEP = 6`, `BAND_PCTS = (5, 25, 50, 75, 95)`, `SCORE_HORIZON = 252`, `FULL_SCALE_LOSS = 0.60`.
- Score: `score = clip(round(1 + 98 * L / 0.60), 1, 99)` with `L = max(0, 1 - P5/V)`, where P5 is the 5th percentile of simulated value at day 252.
- `/analyze` keeps its request and response shapes exactly.

**API rules**
- Rate limits per IP per 60 s: `/report` 10, `/analyze` 10, `/import` 20, everything else 60. Over the limit returns 429 with `Retry-After`.
- Compute gate: 2 slots shared by `/report` and `/analyze`. Wait up to 15 s, then return 503 `{"detail": "busy"}` with `Retry-After: 10`.
- CORS origins come from env `ALLOWED_ORIGINS` (comma-separated), defaulting to `http://localhost:5173`. Methods GET and POST. No credentials.
- Client IP is the **last** `X-Forwarded-For` entry, falling back to `request.client.host`.

**Process rule:** if any existing test fails in a way this plan does not predict, **stop and report it**. Do not loosen the test.

## Review Focus

These failure modes are implied by the spec but not covered by the obvious tests. Each one is pinned by a test in the task named:

1. **A portfolio that contains SPY itself.** The benchmark column is the holding. `inputs.prepare` must not duplicate the column, and the benchmark score must still compute. Pinned in Tasks 5 and 6.
2. **A crisis with fewer than 5 clean days, or one a holding missed.** It must appear with a status, never vanish. Pinned in Task 4.
3. **A 429 or 503 seen from the browser.** The response must still carry CORS headers, or the frontend shows a useless "CORS error" instead of "slow down". Pinned in Task 9.
4. **A rate limiter fed thousands of distinct IPs** must not grow without bound. Pinned in Task 9.
5. **Determinism across the 2-slot gate.** Two identical `/report` calls must return byte-identical JSON, even with shrinkage on. Pinned in Task 7.

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `risk_engine/metrics.py` | modify | + `to_dollars`, `return_histogram`, `correlation` |
| `risk_engine/optimize.py` | modify | `_solve` checks convergence; + `OptimizerError` |
| `risk_engine/backtest.py` | modify | each `backtest_var` row carries its rolling series |
| `risk_engine/factors.py` | modify | + `factor_gap_reason` (moved from analyze) |
| `risk_engine/config.py` | modify | + `CONCENTRATION_LIMIT = 0.50` |
| `risk_engine/stress.py` | create | crisis replay |
| `risk_engine/benchmark.py` | create | portfolio vs S&P 500 |
| `risk_engine/simulate.py` | create | bootstrap Monte Carlo + risk score |
| `risk_engine/inputs.py` | create | holdings + snapshot → `Inputs`; `PortfolioError` lives here |
| `risk_engine/analyze.py` | modify | rebuilt on `inputs.prepare`; same output |
| `risk_engine/report.py` | create | every section → one dict |
| `api/limits.py` | create | `RateLimiter`, `ComputeGate`, `client_ip` |
| `api/schemas.py` | modify | report models, universe and import models, string portfolio ids |
| `api/main.py` | modify | `/report`, `/universe`, `/import`, middleware, `/portfolios` fix |
| `api/store.py` | modify | random token ids, no list |
| `app.py` | modify | reads one report dict; no inline math |
| `tests/test_metrics_extra.py`, `test_stress.py`, `test_benchmark.py`, `test_simulate.py`, `test_inputs.py`, `test_report.py`, `test_api_report.py`, `test_limits.py` | create | |
| `tests/test_optimize.py`, `test_backtest.py`, `test_api.py`, `test_store.py`, `tests/conftest.py` | modify | |

---

### Task 1: Small engine additions: dollars, histogram, correlation, optimizer convergence, backtest series

**Files:**
- Modify: `risk_engine/metrics.py` (append)
- Modify: `risk_engine/optimize.py` (`_solve`, new `OptimizerError`)
- Modify: `risk_engine/backtest.py` (`backtest_var`)
- Modify: `risk_engine/analyze.py` (map `OptimizerError` to `PortfolioError`)
- Create: `tests/test_metrics_extra.py`
- Modify: `tests/test_optimize.py`, `tests/test_backtest.py` (append)

**Interfaces:**
- Produces:
  - `metrics.to_dollars(log_r: float, value: float) -> float`
  - `metrics.return_histogram(port_returns, bins=80) -> {"edges": list[float], "density": list[float]}`
  - `metrics.correlation(log_returns: DataFrame, tickers: list) -> {"tickers": list[str], "matrix": list[list[float]]}`
  - `optimize.OptimizerError(RuntimeError)`
  - Each `backtest_var` row dict gains key `"series"`: the `rolling_var_breaches` dict `{"dates", "var", "realized", "breach"}`.

- [ ] **Step 1: Create the branch.**

```bash
cd /c/Users/sujan/portfolio-risk-dashboard && git checkout -b api-growth
```

- [ ] **Step 2: Write the failing tests** in `tests/test_metrics_extra.py`:

```python
import numpy as np
import pandas as pd
import pytest

from risk_engine import metrics


def test_to_dollars_uses_the_simple_return_not_the_log_return():
    # A -10% log return is a 9.516% real loss, not 10%.
    assert metrics.to_dollars(-0.10, 100_000) == pytest.approx(-9516.258, abs=1e-3)
    assert metrics.to_dollars(0.0, 100_000) == 0.0


def test_return_histogram_is_a_density_over_the_returns():
    r = pd.Series([-0.02, -0.01, 0.0, 0.0, 0.01, 0.03])
    h = metrics.return_histogram(r, bins=5)
    assert len(h["edges"]) == 6 and len(h["density"]) == 5
    assert h["edges"][0] == pytest.approx(-0.02) and h["edges"][-1] == pytest.approx(0.03)
    widths = np.diff(h["edges"])
    assert float(np.sum(np.array(h["density"]) * widths)) == pytest.approx(1.0)


def test_correlation_matches_pandas_and_keeps_ticker_order(market):
    _, lr = market
    c = metrics.correlation(lr, ["MSFT", "AAPL"])
    assert c["tickers"] == ["MSFT", "AAPL"]
    assert c["matrix"][0][0] == pytest.approx(1.0)
    assert c["matrix"][0][1] == pytest.approx(lr["MSFT"].corr(lr["AAPL"]))
```

Append to `tests/test_optimize.py`:

```python
def test_a_solver_that_does_not_converge_raises_instead_of_returning_noise(monkeypatch):
    import types
    from risk_engine import optimize
    fake = types.SimpleNamespace(success=False, message="Iteration limit reached",
                                 x=np.array([0.9, 0.1]))
    monkeypatch.setattr(optimize, "minimize", lambda *a, **k: fake)
    with pytest.raises(optimize.OptimizerError, match="Iteration limit reached"):
        optimize.min_variance_weights(np.eye(2))
```

(If `tests/test_optimize.py` lacks `import numpy as np` or `import pytest` at the top, add them.)

Append to `tests/test_backtest.py`:

```python
def test_each_backtest_row_carries_its_rolling_series(normal_returns):
    from risk_engine import backtest
    r = normal_returns.iloc[:600]
    rows = backtest.backtest_var(r, 0.95, 250)
    hist = backtest.rolling_var_breaches(r, 0.95, 250, "historical")
    series = rows[0]["series"]
    assert rows[0]["method"] == "historical"
    assert list(series["breach"]) == list(hist["breach"])
    assert int(series["breach"].sum()) == rows[0]["breaches"]
```

- [ ] **Step 3: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_metrics_extra.py tests/test_optimize.py tests/test_backtest.py -q`
Expected: FAIL with `AttributeError` (`to_dollars`, `OptimizerError`) and `KeyError: 'series'`.

- [ ] **Step 4: Implement.**

Append to `risk_engine/metrics.py`:

```python
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
```

In `risk_engine/optimize.py`, add this below the imports:

```python
class OptimizerError(RuntimeError):
    """The solver did not converge, so its weights would be noise."""
```

Then replace `_solve` with:

```python
def _solve(objective, n):
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    res = minimize(objective, w0, method="SLSQP", bounds=bnds, constraints=cons)
    if not res.success:
        raise OptimizerError(f"the optimizer did not converge ({res.message})")
    w = np.clip(res.x, 0, None)
    return w / w.sum()
```

In `risk_engine/backtest.py` `backtest_var`, add `"series": res,` to the row dict, after `"passed"`.

In `risk_engine/analyze.py`:
- Wrap the two weight calls:

```python
    try:
        w_ms = optimize.max_sharpe_weights(mu, cov)
        w_mv = optimize.min_variance_weights(cov)
    except optimize.OptimizerError as e:
        raise PortfolioError([{"ticker": OPTIMIZER_PROBLEM, "reason": str(e)}]) from e
```

- Add `OPTIMIZER_PROBLEM = "(optimizer)"` next to `FACTORS_PROBLEM`.

- [ ] **Step 5: Run the task tests, then the full suite.**

Run: `python -m pytest tests/test_metrics_extra.py tests/test_optimize.py tests/test_backtest.py -q`, then `python -m pytest -q`.
Expected: all pass (373). If an existing test now raises `OptimizerError`, real SLSQP is failing on fixture data. **Stop and report** with the holdings and message.

- [ ] **Step 6: Commit.**

```bash
git add risk_engine/metrics.py risk_engine/optimize.py risk_engine/backtest.py risk_engine/analyze.py tests/test_metrics_extra.py tests/test_optimize.py tests/test_backtest.py
git commit -m "Add dollar, histogram and correlation helpers; refuse unconverged optimizer weights"
```

---

### Task 2: Benchmark comparison module

**Files:**
- Create: `risk_engine/benchmark.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `metrics.annualized_vol`, `sharpe_ratio`, `max_drawdown`; `config.TRADING_DAYS`
- Produces:
  - `benchmark.beta(pr, spy_r) -> float` (ddof=1 throughout)
  - `benchmark.compare(pr, spy_r, value) -> Comparison`
  - `Comparison` fields:
    - `rows: list[dict]`, where each dict has keys `label, portfolio, benchmark, higher_is_better, kind` (`kind` is `"pct"` or `"ratio"`)
    - `beta: float`
    - `growth_portfolio: pd.Series`
    - `growth_benchmark: pd.Series`

- [ ] **Step 1: Write the failing tests** in `tests/test_benchmark.py`. The legacy function is a verbatim copy of the app.py:576-584 math, used as the oracle:

```python
import numpy as np
import pytest

from risk_engine import benchmark, metrics
from risk_engine.config import TRADING_DAYS


def _legacy(prc, spyc):
    """app.py:576-584 before the extraction, verbatim math."""
    b_mu, b_sd = spyc.mean(), spyc.std()
    b_ret, b_vol = b_mu * TRADING_DAYS, b_sd * np.sqrt(TRADING_DAYS)
    b_sharpe = (b_mu / b_sd) * np.sqrt(TRADING_DAYS) if b_sd > 0 else 0.0
    b_wealth = np.exp(spyc.cumsum())
    b_dd = (b_wealth / b_wealth.cummax() - 1.0).min()
    beta = float(np.cov(prc, spyc)[0, 1] / np.var(spyc)) if np.var(spyc) > 0 else 0.0
    return b_ret, b_vol, b_sharpe, b_dd, beta


def _pr(lr):
    w = np.array([0.5, 0.3, 0.2])
    return metrics.portfolio_returns(lr, ["AAPL", "JPM", "TLT"], w)


def test_benchmark_column_matches_the_old_inline_math(market):
    _, lr = market
    pr, spy = _pr(lr), lr["SPY"]
    c = benchmark.compare(pr, spy, 100_000)
    b_ret, b_vol, b_sharpe, b_dd, _ = _legacy(pr, spy)
    got = {r["label"]: r["benchmark"] for r in c.rows}
    assert got["Annualized return"] == pytest.approx(b_ret)
    assert got["Annualized volatility (risk)"] == pytest.approx(b_vol)
    assert got["Sharpe ratio"] == pytest.approx(b_sharpe)
    assert got["Maximum drawdown"] == pytest.approx(b_dd)


def test_beta_uses_one_degrees_of_freedom_convention(market):
    # The old code divided an ddof=1 covariance by an ddof=0 variance, inflating beta by n/(n-1).
    _, lr = market
    pr, spy = _pr(lr), lr["SPY"]
    n = len(spy)
    legacy_beta = _legacy(pr, spy)[4]
    assert benchmark.beta(pr, spy) == pytest.approx(legacy_beta * (n - 1) / n)


def test_spy_against_itself_has_beta_one_and_identical_growth(market):
    _, lr = market
    spy = lr["SPY"]
    c = benchmark.compare(spy, spy, 10_000)
    assert c.beta == pytest.approx(1.0)
    assert c.growth_portfolio.iloc[-1] == pytest.approx(c.growth_benchmark.iloc[-1])
    assert c.growth_portfolio.iloc[0] == pytest.approx(10_000 * np.exp(spy.iloc[0]))


def test_rows_are_in_display_order_with_direction():
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    p = pd.Series([0.01, -0.02, 0.005, 0.0], idx)
    c = benchmark.compare(p, p * 0.5, 1000)
    assert [r["label"] for r in c.rows] == ["Annualized return", "Annualized volatility (risk)",
                                            "Sharpe ratio", "Maximum drawdown"]
    assert [r["higher_is_better"] for r in c.rows] == [True, False, True, True]
    assert [r["kind"] for r in c.rows] == ["pct", "pct", "ratio", "pct"]
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: FAIL with `ImportError: cannot import name 'benchmark'`.

- [ ] **Step 3: Implement** `risk_engine/benchmark.py`:

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_benchmark.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit.**

```bash
git add risk_engine/benchmark.py tests/test_benchmark.py
git commit -m "Move the S&P 500 comparison into the engine and fix the beta denominator"
```

---

### Task 3: Monte Carlo and the risk score

**Files:**
- Create: `risk_engine/simulate.py`
- Create: `tests/test_simulate.py`

**Interfaces:**
- Produces:
  - `simulate.HORIZONS: dict[str, int]` = `{"6 months": 126, "1 year": 252, "2 years": 504, "3 years": 756}`
  - Constants: `N_SIMS`, `SEED`, `BAND_STEP`, `BAND_PCTS`, `SCORE_HORIZON`, `FULL_SCALE_LOSS`
  - `draw_indices(n_obs, *, days=756, n_sims=N_SIMS, seed=SEED) -> np.ndarray[int32] (n_sims, days)`
  - `simulate(daily_log_returns, value, draws) -> Simulation`
  - `Simulation(value, band_days: list[int], bands: dict[int, list[float]], horizons: dict[int, dict])`
    - Each `horizons[d]` has keys `p5, p50, p95, prob_loss, prob_up20`.
    - `bands` is keyed by percentile `5, 25, 50, 75, 95`.
  - `score_from_loss(loss: float) -> int`
  - `risk_score(sim) -> Score(score: int, bad_year_loss: float, loss_dollars: float, gain_dollars: float)`

- [ ] **Step 1: Write the failing tests** in `tests/test_simulate.py`:

```python
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from risk_engine import metrics, simulate as sim_mod
from risk_engine.simulate import draw_indices, risk_score, score_from_loss, simulate


def _legacy_terminal(hist_r, value, draws):
    """app.py:1187-1192 math, verbatim, on the given draws."""
    sim_paths = value * np.exp(np.cumsum(hist_r[draws], axis=1))
    terminal = sim_paths[:, -1]
    t5, t50, t95 = np.percentile(terminal, [5, 50, 95])
    return t5, t50, t95, float((terminal < value).mean()), float((terminal >= value * 1.2).mean())


def test_simulation_reproduces_the_old_math_on_the_same_draws(market):
    _, lr = market
    r = lr["AAPL"].to_numpy()
    draws = np.random.default_rng(42).integers(0, len(r), size=(2000, 252))
    s = simulate(r, 50_000, draws)
    t5, t50, t95, ploss, pup = _legacy_terminal(r, 50_000, draws)
    h = s.horizons[252]
    assert (h["p5"], h["p50"], h["p95"]) == pytest.approx((t5, t50, t95))
    assert (h["prob_loss"], h["prob_up20"]) == pytest.approx((ploss, pup))


def test_every_horizon_lands_on_a_band_point_and_bands_start_at_value(market):
    _, lr = market
    r = lr["SPY"].to_numpy()
    s = simulate(r, 10_000, draw_indices(len(r), n_sims=500))
    assert s.band_days[0] == 0 and s.band_days[-1] == 756 and len(s.band_days) == 127
    assert set(sim_mod.HORIZONS.values()) <= set(s.band_days)
    for p in sim_mod.BAND_PCTS:
        assert s.bands[p][0] == 10_000
        assert len(s.bands[p]) == 127
    one_year = s.band_days.index(252)
    assert s.bands[5][one_year] == pytest.approx(s.horizons[252]["p5"])
    assert s.bands[95][one_year] == pytest.approx(s.horizons[252]["p95"])


def test_same_draws_give_identical_results(market):
    _, lr = market
    r = lr["MSFT"].to_numpy()
    a = simulate(r, 1000, draw_indices(len(r), n_sims=300))
    b = simulate(r, 1000, draw_indices(len(r), n_sims=300))
    assert a.horizons == b.horizons and a.bands == b.bands


def test_score_is_linear_bounded_and_monotone():
    assert score_from_loss(0.0) == 1
    assert score_from_loss(0.30) == 50
    assert score_from_loss(0.60) == 99
    assert score_from_loss(0.95) == 99
    losses = np.linspace(0, 0.7, 50)
    scores = [score_from_loss(x) for x in losses]
    assert scores == sorted(scores)


def test_score_reads_the_same_run_as_the_dollar_range():
    s = sim_mod.Simulation(value=100_000, band_days=[0], bands={},
                           horizons={252: {"p5": 76_000.0, "p50": 105_000.0, "p95": 131_000.0,
                                           "prob_loss": 0.3, "prob_up20": 0.2}})
    sc = risk_score(s)
    assert sc.bad_year_loss == pytest.approx(0.24)
    assert sc.loss_dollars == pytest.approx(24_000)
    assert sc.gain_dollars == pytest.approx(31_000)
    assert sc.score == score_from_loss(0.24) == 40


def test_a_bad_year_that_still_gains_scores_one_and_loses_nothing():
    s = sim_mod.Simulation(value=100, band_days=[0], bands={},
                           horizons={252: {"p5": 101.0, "p50": 104.0, "p95": 108.0,
                                           "prob_loss": 0.0, "prob_up20": 0.0}})
    sc = risk_score(s)
    assert sc.score == 1 and sc.loss_dollars == 0.0 and sc.bad_year_loss == 0.0


def _score(lr, tickers, weights):
    pr = metrics.portfolio_returns(lr, tickers, np.asarray(weights, dtype=float))
    draws = draw_indices(len(pr), n_sims=3000)
    return risk_score(simulate(pr.to_numpy(), 100_000, draws)).score


def test_volatile_stock_scores_above_spy_and_bonds_below(market):
    _, lr = market
    spy = _score(lr, ["SPY"], [1.0])
    assert _score(lr, ["TSLA"], [1.0]) > spy
    assert _score(lr, ["TLT", "IEF"], [0.5, 0.5]) < spy


def test_peak_memory_stays_under_150_mb_at_full_size(market):
    _, lr = market
    r = lr["SPY"].to_numpy()
    tracemalloc.start()
    try:
        simulate(r, 100_000, draw_indices(len(r)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 150 * 2**20, f"peak {peak / 2**20:.0f} MB"
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_simulate.py -q`
Expected: FAIL with `ModuleNotFoundError: risk_engine.simulate`.

- [ ] **Step 3: Implement** `risk_engine/simulate.py`:

```python
"""Bootstrap Monte Carlo and the 1-99 risk score. Pure.

One run draws the longest horizon once. Every shorter horizon, the fan-chart bands
and the risk score are read off that one run, so they can never disagree. Passing
the same `draws` to two portfolios (yours and the S&P 500) gives both the same
random days, so their scores differ only because the holdings do."""
from dataclasses import dataclass

import numpy as np

HORIZONS = {"6 months": 126, "1 year": 252, "2 years": 504, "3 years": 756}
SCORE_HORIZON = 252
N_SIMS = 10_000
SEED = 42
BAND_STEP = 6                     # every horizon is a multiple of 6, so each lands on a band point
BAND_PCTS = (5, 25, 50, 75, 95)
FULL_SCALE_LOSS = 0.60            # a bad-year loss this large or larger scores 99


def draw_indices(n_obs, *, days=max(HORIZONS.values()), n_sims=N_SIMS, seed=SEED):
    """Which historical day each simulated day replays. int32 halves the memory."""
    return np.random.default_rng(seed).integers(0, n_obs, size=(n_sims, days), dtype=np.int32)


@dataclass(frozen=True)
class Simulation:
    value: float
    band_days: list    # 0, 6, ..., the longest horizon
    bands: dict        # percentile -> simulated value at each band day
    horizons: dict     # days -> p5, p50, p95, prob_loss, prob_up20


@dataclass(frozen=True)
class Score:
    score: int
    bad_year_loss: float   # fraction lost at the 5th percentile, floored at 0
    loss_dollars: float
    gain_dollars: float


def simulate(daily_log_returns, value, draws):
    r = np.asarray(daily_log_returns, dtype=float)
    paths = np.take(r, draws)                  # the one n_sims x days buffer
    np.cumsum(paths, axis=1, out=paths)
    np.exp(paths, out=paths)
    paths *= value
    days = draws.shape[1]
    band_days = list(range(0, days + 1, BAND_STEP))
    at = np.percentile(paths[:, [d - 1 for d in band_days[1:]]], BAND_PCTS, axis=0)
    bands = {p: [float(value)] + row.tolist() for p, row in zip(BAND_PCTS, at)}
    horizons = {}
    for d in HORIZONS.values():
        if d > days:
            continue
        terminal = paths[:, d - 1]
        p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
        horizons[d] = {"p5": float(p5), "p50": float(p50), "p95": float(p95),
                       "prob_loss": float((terminal < value).mean()),
                       "prob_up20": float((terminal >= value * 1.2).mean())}
    return Simulation(float(value), band_days, bands, horizons)


def score_from_loss(loss):
    return int(min(99, max(1, round(1 + 98 * loss / FULL_SCALE_LOSS))))


def risk_score(sim):
    h = sim.horizons[SCORE_HORIZON]
    loss = max(0.0, 1.0 - h["p5"] / sim.value)
    return Score(score_from_loss(loss), loss, max(0.0, sim.value - h["p5"]), h["p95"] - sim.value)
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_simulate.py -q`
Expected: 8 passed.
- If the memory test fails, report the measured peak. Do not raise the limit.
- If `score_from_loss(0.30) != 50`, check the rounding: `1 + 98*0.5 = 50.0` exactly.

- [ ] **Step 5: Commit.**

```bash
git add risk_engine/simulate.py tests/test_simulate.py
git commit -m "Add one-run bootstrap Monte Carlo and the 1-99 risk score"
```

---

### Task 4: Stress test module

**Files:**
- Create: `risk_engine/stress.py`
- Create: `tests/test_stress.py`

**Interfaces:**
- Consumes: `portfolio.crisis_coverage(prices, tickers, crises)`, which returns `[{label, start, end, context, missing}]`, and `config.CRISES`
- Produces:
  - `stress.MIN_CRISIS_DAYS = 5`
  - `stress.replay(prices, tickers, weights, crises=CRISES) -> list[CrisisResult]`, in crisis order, **one entry per crisis, always**
  - `CrisisResult` fields:
    - `label, context, start, end` (str)
    - `status` (`"ok"` | `"not_covered"` | `"too_few_days"`)
    - `missing: list[str]`
    - `total, worst_day, max_dd: float | None`
    - `path: pd.Series | None` (buy-and-hold value, starts at 1.0)

- [ ] **Step 1: Write the failing tests** in `tests/test_stress.py`:

```python
import numpy as np
import pandas as pd
import pytest

from risk_engine import stress


def _prices():
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    return pd.DataFrame({"A": [100, 110, 99, 120, 120, 132, 132, 132],
                         "B": [50, 50, 50, 50, 25, 25, 25, 25]}, idx, dtype=float)


CRISIS = [("Test crash", "2020-01-01", "2020-01-10", "context")]


def test_buy_and_hold_path_worked_by_hand():
    (r,) = stress.replay(_prices(), ["A", "B"], np.array([0.5, 0.5]), CRISIS)
    # A/A0 = 1, 1.1, 0.99, 1.2, 1.2, 1.32, 1.32, 1.32 ; B/B0 = 1,1,1,1,0.5,0.5,0.5,0.5
    expected = [1.0, 1.05, 0.995, 1.1, 0.85, 0.91, 0.91, 0.91]
    assert r.status == "ok" and r.missing == []
    assert list(r.path.round(6)) == pytest.approx(expected)
    assert r.total == pytest.approx(-0.09)
    assert r.max_dd == pytest.approx(0.85 / 1.1 - 1)
    assert r.worst_day == pytest.approx(0.85 / 1.1 - 1)


def test_a_crisis_a_holding_missed_is_returned_not_dropped():
    p = _prices()
    p.loc[:"2020-01-03", "B"] = np.nan          # B lists after the crisis start
    (r,) = stress.replay(p, ["A", "B"], np.array([0.5, 0.5]), CRISIS)
    assert r.status == "not_covered" and r.missing == ["B"]
    assert r.total is None and r.path is None


def test_a_crisis_with_too_few_clean_days_is_returned_with_that_status():
    short = [("Blip", "2020-01-01", "2020-01-06", "ctx")]     # 4 business days
    (r,) = stress.replay(_prices(), ["A", "B"], np.array([0.5, 0.5]), short)
    assert r.status == "too_few_days" and r.total is None


def test_real_crises_all_come_back_in_order(market):
    prices, _ = market
    out = stress.replay(prices, ["AAPL", "TLT"], np.array([0.6, 0.4]))
    from risk_engine.config import CRISES
    assert [r.label for r in out] == [c[0] for c in CRISES]
    assert all(r.status == "ok" for r in out)
    covid = out[1]
    assert covid.total < 0 and covid.max_dd < 0
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_stress.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement** `risk_engine/stress.py`. The math is the inline `stress()` at app.py:712-723, unchanged:

```python
"""Replay today's holdings, buy-and-hold at today's weights, through real crash
windows. Every crisis comes back with a status, so one the portfolio can't be
replayed through is reported, never silently dropped. Pure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import portfolio
from risk_engine.config import CRISES

MIN_CRISIS_DAYS = 5


@dataclass(frozen=True)
class CrisisResult:
    label: str
    context: str
    start: str
    end: str
    status: str            # "ok" | "not_covered" | "too_few_days"
    missing: list          # holdings not yet trading when the crisis began
    total: float | None
    worst_day: float | None
    max_dd: float | None
    path: pd.Series | None  # portfolio value, starts at 1.0


def replay(prices, tickers, weights, crises=CRISES):
    tickers, weights = list(tickers), np.asarray(weights, dtype=float)
    out = []
    for c in portfolio.crisis_coverage(prices, tickers, crises):
        head = (c["label"], c["context"], c["start"], c["end"])
        if c["missing"]:
            out.append(CrisisResult(*head, "not_covered", c["missing"], None, None, None, None))
            continue
        wp = prices.loc[c["start"]:c["end"], tickers].dropna()
        if len(wp) < MIN_CRISIS_DAYS:
            out.append(CrisisResult(*head, "too_few_days", [], None, None, None, None))
            continue
        path = (wp / wp.iloc[0] * weights).sum(axis=1)
        out.append(CrisisResult(*head, "ok", [], float(path.iloc[-1] - 1.0),
                                float(path.pct_change().min()),
                                float((path / path.cummax() - 1.0).min()), path))
    return out
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_stress.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit.**

```bash
git add risk_engine/stress.py tests/test_stress.py
git commit -m "Move the crisis replay into the engine and report every crisis with a status"
```

---

### Task 5: Shared inputs pipeline; rebuild `analyze` on it

**Files:**
- Create: `risk_engine/inputs.py`
- Create: `tests/test_inputs.py`
- Modify: `risk_engine/analyze.py` (use `prepare`; re-export `PortfolioError`)
- Modify: `risk_engine/factors.py` (add `factor_gap_reason`, moved from analyze)

**Interfaces:**
- Consumes: `portfolio.resolve_holdings`, `portfolio.portfolio_window`, `metrics.portfolio_returns`
- Produces:
  - `inputs.PortfolioError(ValueError)` with `.problems: list[{"ticker", "reason"}]` and `.rejected: list[{"ticker", "amount", "reason"}]` (holdings dropped before the refusal; empty when none). Moved here; `from risk_engine.analyze import PortfolioError` must keep working.
  - `inputs.HOLDINGS_PROBLEM = "(holdings)"`
  - `inputs.prepare(holdings, snap, *, live_fetch=None, max_live=LIVE_MAX_API, budget_s=LIVE_BUDGET_API_S, allow_rejected=False) -> Inputs`
  - `Inputs` (frozen) fields:
    - `snap`, `res` (`portfolio.Resolution`)
    - `tickers: list[str]`, `weights: np.ndarray`, `value: float`
    - `window` (`portfolio.Window`)
    - `lr` (DataFrame), `pr` (Series), `spy_r` (Series)
  - `factors.factor_gap_reason(factors, shared, window_days) -> str`

- [ ] **Step 1: Write the failing tests** in `tests/test_inputs.py`:

```python
import numpy as np
import pytest

from risk_engine import data
from risk_engine.inputs import HOLDINGS_PROBLEM, PortfolioError, prepare


def _snap():
    return data.load_snapshot()


def test_prepare_builds_weights_returns_and_benchmark():
    inp = prepare({"AAPL": 30000, "TLT": 10000}, _snap())
    assert inp.tickers == ["AAPL", "TLT"] and inp.value == 40000.0
    assert inp.weights.tolist() == [0.75, 0.25]
    assert inp.pr.index.equals(inp.spy_r.index)
    assert inp.pr.iloc[0] == pytest.approx(0.75 * inp.lr["AAPL"].iloc[0] + 0.25 * inp.lr["TLT"].iloc[0])
    assert inp.window.status == "ok"


def test_holding_spy_itself_does_not_duplicate_the_benchmark_column():
    inp = prepare({"SPY": 50000, "AAPL": 50000}, _snap())
    assert list(inp.lr.columns).count("SPY") == 1
    assert inp.spy_r.equals(inp.lr["SPY"])


def test_rejected_tickers_raise_unless_allowed():
    with pytest.raises(PortfolioError) as e:
        prepare({"AAPL": 1000, "MSFT": 1000, "XYZQQ": 5}, _snap())
    assert e.value.problems[0]["ticker"] == "XYZQQ"
    inp = prepare({"AAPL": 1000, "MSFT": 1000, "XYZQQ": 5}, _snap(), allow_rejected=True)
    assert inp.tickers == ["AAPL", "MSFT"] and [r.ticker for r in inp.res.rejected] == ["XYZQQ"]


def test_fewer_than_two_usable_holdings_is_refused():
    with pytest.raises(PortfolioError) as e:
        prepare({"AAPL": 1000, "XYZQQ": 5}, _snap(), allow_rejected=True)
    assert e.value.problems[0]["ticker"] == HOLDINGS_PROBLEM
    assert [(r["ticker"], r["amount"]) for r in e.value.rejected] == [("XYZQQ", 5)]


def test_analyze_still_exports_portfolio_error():
    from risk_engine.analyze import PortfolioError as Reexported
    assert Reexported is PortfolioError
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_inputs.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `risk_engine/inputs.py`:

```python
"""Holdings + snapshot -> everything every section reads: resolved holdings,
weights, the shared window, daily log returns for the portfolio and the S&P 500.
Built once per request. Pure."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import data, metrics, portfolio
from risk_engine.config import BENCHMARK, LIVE_BUDGET_API_S, LIVE_MAX_API, MIN_HOLDINGS

HOLDINGS_PROBLEM = "(holdings)"


class PortfolioError(ValueError):
    """The portfolio can't be analysed as given; `problems` says why, per ticker."""

    def __init__(self, problems, rejected=()):
        super().__init__("; ".join(f"{p['ticker']}: {p['reason']}" for p in problems))
        self.problems = problems
        self.rejected = list(rejected)   # [{ticker, amount, reason}] dropped before the refusal


def _rejected(res):
    return [{"ticker": r.ticker, "amount": r.amount, "reason": r.reason} for r in res.rejected]


@dataclass(frozen=True)
class Inputs:
    snap: object
    res: portfolio.Resolution
    tickers: list
    weights: np.ndarray
    value: float
    window: portfolio.Window
    lr: pd.DataFrame     # holdings + benchmark daily log returns over the window
    pr: pd.Series        # portfolio daily log returns
    spy_r: pd.Series     # benchmark daily log returns, same dates as pr


def prepare(holdings, snap, *, live_fetch=None, max_live=LIVE_MAX_API,
            budget_s=LIVE_BUDGET_API_S, allow_rejected=False):
    res = portfolio.resolve_holdings(holdings, snap, live_fetch=live_fetch or data.fetch_live,
                                     max_live=max_live, budget_s=budget_s)
    if res.rejected and not allow_rejected:
        raise PortfolioError([{"ticker": r.ticker, "reason": r.reason} for r in res.rejected],
                             _rejected(res))
    tickers = res.tickers
    if len(tickers) < MIN_HOLDINGS:
        raise PortfolioError([{"ticker": HOLDINGS_PROBLEM,
                               "reason": f"at least {MIN_HOLDINGS} holdings with a price history "
                                         f"are needed; {len(tickers)} could be used"}],
                             _rejected(res))
    win = portfolio.portfolio_window(res.prices, tickers, benchmark=snap.prices[BENCHMARK])
    if win.status == "too_short":
        raise PortfolioError([{"ticker": win.limiting, "reason": win.message}], _rejected(res))
    amounts = np.array([res.holdings[t] for t in tickers], dtype=float)
    value = float(amounts.sum())
    weights = amounts / value
    lr = win.returns
    return Inputs(snap, res, tickers, weights, value, win, lr,
                  metrics.portfolio_returns(lr, tickers, weights), lr[BENCHMARK])
```

In `risk_engine/factors.py`, add `from risk_engine.config import FACTOR_MIN_OBS` to the imports (TRADING_DAYS is already imported; extend that import), then append:

```python
def factor_gap_reason(factors, shared, window_days):
    ends = f"ends {factors.index.max().date()}" if len(factors) else "is empty"
    return (f"The Fama-French factor data {ends}, so only {shared} of this portfolio's "
            f"{window_days} trading days have factors; at least {FACTOR_MIN_OBS} are needed "
            f"to estimate factor exposure.")
```

Rewrite the top of `risk_engine/analyze.py` up to the `result = {` line:
- Delete the local `PortfolioError` class and `_factor_gap_reason`.
- Import them instead:
  - `from risk_engine.inputs import PortfolioError, prepare`
  - Use `fac.factor_gap_reason`.

```python
def analyze_portfolio(holdings, *, confidence=0.95, include_backtest=True,
                      snapshot=None, live_fetch=None):
    snap = snapshot or data.load_snapshot()
    inp = prepare(holdings, snap, live_fetch=live_fetch)
    res, tickers, win, lr, pr, weights = inp.res, inp.tickers, inp.window, inp.lr, inp.pr, inp.weights
    shared = len(lr.index.intersection(snap.factors.index))
    if shared < FACTOR_MIN_OBS:
        raise PortfolioError([{"ticker": FACTORS_PROBLEM,
                               "reason": fac.factor_gap_reason(snap.factors, shared, win.n_days)}])

    hist_var, hist_cvar = metrics.historical_var_cvar(pr, confidence)
```

The rest of the function is unchanged. Remove the now-unused imports `BENCHMARK`, `LIVE_BUDGET_API_S`, `LIVE_MAX_API` from analyze if nothing else uses them. Keep `portfolio`, because `sector_mix` and `crisis_coverage` still use it.

- [ ] **Step 4: Run the task tests and the analyze/API tests.**

Run: `python -m pytest tests/test_inputs.py tests/test_analyze.py tests/test_api.py -q`
Expected: all pass, with `/analyze` output unchanged.

- [ ] **Step 5: Commit.**

```bash
git add risk_engine/inputs.py risk_engine/analyze.py risk_engine/factors.py tests/test_inputs.py
git commit -m "Build every analysis on one shared inputs pipeline"
```

---

### Task 6: The report builder

**Files:**
- Modify: `risk_engine/config.py` (append `CONCENTRATION_LIMIT = 0.50  # a stock sector this large of the portfolio is a concentration`)
- Create: `risk_engine/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes everything above: `inputs.prepare`/`Inputs`, `simulate.*`, `stress.replay`, `benchmark.compare`, `metrics.*`, `optimize.*` (incl. `OptimizerError`), `backtest.backtest_var`, `factors.*`, `portfolio.sector_mix`.
- Produces:
  - `report.build_report(holdings, *, confidence=0.95, shrinkage=False, snapshot=None, live_fetch=None) -> dict`
  - `report.report_from_inputs(inp, *, confidence=0.95, shrinkage=False) -> dict`
  - `report.ASSUMPTIONS: list[str]`, `report.BACKTEST_WINDOW = 250`, `report.DIVERSIFIER_MARGIN = 0.02`
- The dict shape is fixed below. Task 7's pydantic models mirror it key for key.

- [ ] **Step 1: Write the failing tests** in `tests/test_report.py`:

```python
import dataclasses
import json

import numpy as np
import pytest

from risk_engine import data, optimize
from risk_engine.report import build_report

H = {"AAPL": 30000, "JPM": 20000, "TLT": 40000, "IEF": 10000}
SECTIONS = {"score", "portfolio", "risk", "profile", "benchmark", "distribution", "drawdown",
            "stress", "correlation", "backtest", "risk_contribution", "factors", "optimizer",
            "monte_carlo", "data"}


@pytest.fixture(scope="module")
def rep():
    return build_report(H)


def test_report_has_every_section_and_is_json_safe(rep):
    assert set(rep) == SECTIONS
    json.dumps(rep, allow_nan=False)        # raises on numpy types or NaN


def test_headline_score_is_read_off_the_one_year_monte_carlo(rep):
    one_year = next(h for h in rep["monte_carlo"]["horizons"] if h["days"] == 252)
    s, v = rep["score"]["portfolio"], rep["portfolio"]["value"]
    assert s["loss_dollars"] == pytest.approx(max(0.0, v - one_year["p5"]))
    assert s["gain_dollars"] == pytest.approx(one_year["p95"] - v)
    assert 1 <= s["score"] <= 99 and 1 <= rep["score"]["benchmark"]["score"] <= 99
    assert rep["score"]["horizon_days"] == 252


def test_bond_heavy_portfolio_scores_below_the_sp500(rep):
    assert rep["score"]["portfolio"]["score"] < rep["score"]["benchmark"]["score"]


def test_var_dollars_use_the_simple_return(rep):
    r, v = rep["risk"], rep["portfolio"]["value"]
    assert r["dollars"]["hist_var"] == pytest.approx(np.expm1(r["hist_var"]) * v)
    assert r["gap_dollars"] == pytest.approx(r["dollars"]["hist_var"] - r["dollars"]["gaussian_var"])


def test_every_crisis_is_present_with_a_status(rep):
    from risk_engine.config import CRISES
    assert [c["label"] for c in rep["stress"]] == [c[0] for c in CRISES]
    ok = [c for c in rep["stress"] if c["status"] == "ok"]
    assert ok and all(c["loss_dollars"] == pytest.approx(c["total"] * rep["portfolio"]["value"]) for c in ok)
    assert all(c["path"]["values"][0] == 0.0 for c in ok)     # return from start


def test_series_are_parallel_lists_of_iso_dates(rep):
    s = rep["drawdown"]
    assert len(s["dates"]) == len(s["values"]) > 500
    assert s["dates"][0] > rep["data"]["window_start"]       # first return is the day after the first price
    assert len(s["dates"][0]) == 10


def test_risk_shares_sum_to_one_and_name_a_driver(rep):
    rc = rep["risk_contribution"]
    assert sum(row["risk_share"] for row in rc["rows"]) == pytest.approx(1.0)
    assert rc["biggest_driver"] in H


def test_optimizer_section_and_sector_mix(rep):
    o = rep["optimizer"]
    assert o["available"] is True and o["reason"] == ""
    assert sum(o["max_sharpe"]["weights"].values()) == pytest.approx(1.0)
    assert o["frontier"] and all(0 < p["vol"] < 1 for p in o["frontier"])
    assert o["sector_mix"]["sectors"] == {"Technology": pytest.approx(0.3), "Financial Services": pytest.approx(0.2)}
    assert o["sector_mix"]["non_equity"] == {"Bond": pytest.approx(0.5)}
    assert o["sector_mix"]["concentrated"] is False


def test_unconverged_optimizer_is_a_partial_section_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise optimize.OptimizerError("the optimizer did not converge (test)")
    monkeypatch.setattr(optimize, "max_sharpe_weights", boom)
    rep = build_report(H)
    assert rep["optimizer"]["available"] is False
    assert "did not converge" in rep["optimizer"]["reason"]
    assert rep["optimizer"]["max_sharpe"] is None
    assert rep["optimizer"]["sector_mix"]["non_equity"] == {"Bond": pytest.approx(0.5)}


def test_thin_factor_overlap_is_a_partial_section():
    snap = data.load_snapshot(use_memo=False)
    bare = dataclasses.replace(snap, factors=snap.factors.iloc[:30])
    rep = build_report(H, snapshot=bare)
    assert rep["factors"]["available"] is False and "at least 60" in rep["factors"]["reason"]
    assert rep["factors"]["betas"] is None
    assert rep["score"]["portfolio"]["score"] >= 1        # the rest still computed


def test_shrinkage_changes_the_optimizer_only():
    a, b = build_report(H), build_report(H, shrinkage=True)
    assert a["score"] == b["score"] and a["risk"] == b["risk"]
    assert b["optimizer"]["shrinkage"] is True


def test_portfolio_holding_spy_scores_against_itself_cleanly():
    rep = build_report({"SPY": 50000, "AAPL": 50000})
    assert rep["score"]["benchmark"]["score"] >= 1
    assert rep["benchmark"]["beta"] > 0.5
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_report.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** `risk_engine/report.py`:

```python
"""Every dashboard section for one portfolio, as one JSON-safe dict. Pure.

The API serves this dict; the Streamlit page lays it out. Both read the same
numbers, so the two frontends can never disagree. A section that can't be
computed (thin factor data, an optimizer that won't converge) comes back with
`available: false` and a reason instead of failing the whole report."""
import numpy as np

from risk_engine import backtest, benchmark, data, metrics, optimize, simulate, stress
from risk_engine import factors as fac
from risk_engine import portfolio
from risk_engine.config import (BACKTEST_MIN_OBS, CONCENTRATION_LIMIT, FACTOR_MIN_OBS,
                                TRADING_DAYS)
from risk_engine.inputs import prepare

BACKTEST_WINDOW = 250
FRONTIER_POINTS = 40
DIVERSIFIER_MARGIN = 0.02     # risk share this far under capital share counts as a diversifier
REALLOC_THRESHOLD = 0.01      # weight change smaller than this is not worth suggesting

ASSUMPTIONS = [
    "Risk numbers come from this portfolio's own daily returns over the window shown; the "
    "future can differ from the past.",
    "The Monte Carlo replays real past days from 2018 onward, mostly a rising market, so the "
    "bad-year figure leans optimistic. The S&P 500 score replays the same days, so the "
    "comparison stays fair.",
    "Replaying days one at a time ignores that bad days come in clusters, so short-horizon "
    "losses are understated.",
    "Growth charts rebalance to today's weights every day; the stress test holds today's "
    "weights without rebalancing.",
    "Expected returns are average daily log returns times 252, slightly below the arithmetic "
    "average.",
    "Dollar figures convert log returns to real gains and losses with exp(r) - 1.",
    "Educational tool, not financial advice.",
]


def _sig(x, digits=6):
    return float(f"{float(x):.{digits}g}")


def _series(s):
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [_sig(v) for v in s.to_numpy()]}


def _score(sc):
    return {"score": sc.score, "bad_year_loss": sc.bad_year_loss,
            "loss_dollars": sc.loss_dollars, "gain_dollars": sc.gain_dollars}


def _holdings(inp):
    out = []
    for t, w in zip(inp.tickers, inp.weights):
        m = inp.res.meta[t]
        out.append({"ticker": t, "name": m["name"], "sector": m["sector"], "type": m["type"],
                    "asset_class": m["asset_class"], "amount": float(inp.res.holdings[t]),
                    "weight": float(w), "source": inp.res.source[t]})
    return out


def _risk(inp, conf):
    hv, hc = metrics.historical_var_cvar(inp.pr, conf)
    gv, gc = metrics.gaussian_var_cvar(inp.pr, conf)
    fr = {"hist_var": hv, "hist_cvar": hc, "gaussian_var": gv, "gaussian_cvar": gc}
    dollars = {k: metrics.to_dollars(v, inp.value) for k, v in fr.items()}
    return {"confidence": conf, **fr, "dollars": dollars,
            "gap_dollars": dollars["hist_var"] - dollars["gaussian_var"]}


def _stress(inp):
    out = []
    for r in stress.replay(inp.res.prices, inp.tickers, inp.weights):
        ok = r.status == "ok"
        out.append({"label": r.label, "context": r.context, "start": r.start, "end": r.end,
                    "status": r.status, "missing": r.missing, "total": r.total,
                    "worst_day": r.worst_day, "max_dd": r.max_dd,
                    "loss_dollars": r.total * inp.value if ok else None,
                    "path": _series(r.path - 1.0) if ok else None})
    return out


def _backtest(inp, conf):
    rows = backtest.backtest_var(inp.pr, conf, BACKTEST_WINDOW)
    methods = {r["method"]: {"observations": int(r["observations"]), "breaches": int(r["breaches"]),
                             "expected": float(r["expected"]), "kupiec_p": float(r["kupiec_p"]),
                             "christoffersen_p": float(r["christoffersen_p"]),
                             "passed": bool(r["passed"]) if r["observations"] >= BACKTEST_MIN_OBS else None}
               for r in rows}
    s = next(r["series"] for r in rows if r["method"] == "historical")
    series = {"dates": [d.strftime("%Y-%m-%d") for d in s["dates"]],
              "var": [_sig(v) for v in s["var"]], "realized": [_sig(v) for v in s["realized"]],
              "breach": [int(b) for b in s["breach"]]}
    return {"window": BACKTEST_WINDOW, "min_observations": BACKTEST_MIN_OBS,
            "methods": methods, "series": series}


def _risk_contribution(inp):
    w = inp.weights
    share = optimize.risk_contribution(optimize.sample_cov(inp.lr, inp.tickers), w)
    hot, div = int(np.argmax(share - w)), int(np.argmin(share))
    return {"rows": [{"ticker": t, "weight": float(a), "risk_share": float(b)}
                     for t, a, b in zip(inp.tickers, w, share)],
            "biggest_driver": inp.tickers[hot],
            "diversifier": inp.tickers[div] if share[div] < w[div] - DIVERSIFIER_MARGIN else None}


def _factors(inp):
    snap = inp.snap
    through = str(snap.factors.index.max().date()) if len(snap.factors) else None
    shared = len(inp.pr.index.intersection(snap.factors.index))
    if shared < FACTOR_MIN_OBS:
        return {"available": False, "reason": fac.factor_gap_reason(snap.factors, shared, inp.window.n_days),
                "data_through": through, "betas": None, "alpha_annual": None, "r2": None,
                "variance_split": None, "observations": shared}
    reg = fac.factor_regression(inp.pr, snap.factors)
    return {"available": True, "reason": "", "data_through": through, "betas": reg["betas"],
            "alpha_annual": reg["alpha_annual"], "r2": reg["r2"],
            "variance_split": fac.variance_attribution(reg), "observations": reg["n_obs"]}


def _sector_mix(inp):
    mix = portfolio.sector_mix(inp.tickers, inp.weights, inp.res.meta)
    top = max(mix.sectors, key=mix.sectors.get) if mix.sectors else None
    top_pct = mix.sectors[top] if top else 0.0
    return {"sectors": mix.sectors, "equity": mix.equity, "diversified": mix.diversified,
            "unplaced": mix.unplaced, "non_equity": mix.non_equity, "top_sector": top,
            "top_sector_pct": top_pct, "concentrated": top_pct >= CONCENTRATION_LIMIT}


def _optimizer(inp, shrinkage):
    t, w, v = inp.tickers, inp.weights, inp.value
    mu = optimize.annualized_mean(inp.lr, t)
    cov = optimize.ledoit_wolf_cov(inp.lr, t) if shrinkage else optimize.sample_cov(inp.lr, t)
    base = {"shrinkage": shrinkage, "sector_mix": _sector_mix(inp)}
    try:
        w_ms = optimize.max_sharpe_weights(mu, cov)
        w_mv = optimize.min_variance_weights(cov)
    except optimize.OptimizerError as e:
        return {**base, "available": False, "reason": str(e), "current": None, "max_sharpe": None,
                "min_variance": None, "frontier": [], "reallocation": [], "adds": [], "trims": []}

    def point(weights):
        r, vol, s = optimize.perf(weights, mu, cov)
        return {"ret": r, "vol": vol, "sharpe": s,
                "weights": {k: float(x) for k, x in zip(t, weights)}}

    fr_v, fr_r = optimize.efficient_frontier(mu, cov, n_points=FRONTIER_POINTS)
    delta = w_ms - w
    order_up, order_down = np.argsort(delta)[::-1], np.argsort(delta)
    return {**base, "available": True, "reason": "", "current": point(w),
            "max_sharpe": point(w_ms), "min_variance": point(w_mv),
            "frontier": [{"vol": a / 100, "ret": b / 100} for a, b in zip(fr_v, fr_r)],
            "reallocation": [{"ticker": k, "type": inp.res.meta[k]["asset_class"],
                              "sector": inp.res.meta[k]["sector"], "current": float(a * v),
                              "suggested": float(b * v), "delta_dollars": float((b - a) * v)}
                             for k, a, b in zip(t, w, w_ms)],
            "adds": [t[i] for i in order_up if delta[i] > REALLOC_THRESHOLD][:2],
            "trims": [t[i] for i in order_down if delta[i] < -REALLOC_THRESHOLD][:2]}


def _monte_carlo(sim):
    by_days = {d: label for label, d in simulate.HORIZONS.items()}
    return {"n_sims": simulate.N_SIMS, "seed": simulate.SEED, "value": sim.value,
            "band_days": sim.band_days,
            "bands": {f"p{p}": [_sig(x) for x in vals] for p, vals in sim.bands.items()},
            "horizons": [{"label": by_days[d], "days": d, **h} for d, h in sim.horizons.items()]}


def report_from_inputs(inp, *, confidence=0.95, shrinkage=False):
    draws = simulate.draw_indices(len(inp.pr))
    sim = simulate.simulate(inp.pr.to_numpy(), inp.value, draws)
    bench = simulate.simulate(inp.spy_r.to_numpy(), inp.value, draws)
    del draws
    cmp_ = benchmark.compare(inp.pr, inp.spy_r, inp.value)
    corr = metrics.correlation(inp.lr, inp.tickers)
    win = inp.window
    return {
        "score": {"portfolio": _score(simulate.risk_score(sim)),
                  "benchmark": _score(simulate.risk_score(bench)),
                  "horizon_days": simulate.SCORE_HORIZON},
        "portfolio": {"value": inp.value, "holdings": _holdings(inp)},
        "risk": _risk(inp, confidence),
        "profile": {"annual_return": float(inp.pr.mean() * TRADING_DAYS),
                    "annual_vol": metrics.annualized_vol(inp.pr),
                    "sharpe": metrics.sharpe_ratio(inp.pr),
                    "max_drawdown": metrics.max_drawdown(inp.pr),
                    "trading_days": int(len(inp.pr))},
        "benchmark": {"rows": cmp_.rows, "beta": cmp_.beta,
                      "growth": {"portfolio": _series(cmp_.growth_portfolio),
                                 "benchmark": _series(cmp_.growth_benchmark)}},
        "distribution": {**metrics.return_histogram(inp.pr), "mean": float(inp.pr.mean()),
                         "std": float(inp.pr.std())},
        "drawdown": _series(metrics.drawdown_series(inp.pr)),
        "stress": _stress(inp),
        "correlation": {"tickers": corr["tickers"],
                        "matrix": [[round(x, 3) for x in row] for row in corr["matrix"]]},
        "backtest": _backtest(inp, confidence),
        "risk_contribution": _risk_contribution(inp),
        "factors": _factors(inp),
        "optimizer": _optimizer(inp, shrinkage),
        "monte_carlo": _monte_carlo(sim),
        "data": {"as_of": inp.snap.as_of, "window_start": str(win.start.date()),
                 "window_end": str(win.end.date()), "window_days": int(win.n_days),
                 "window_status": win.status, "window_message": win.message,
                 "rejected": [{"ticker": r.ticker, "amount": r.amount, "reason": r.reason}
                              for r in inp.res.rejected],
                 "assumptions": ASSUMPTIONS},
    }


def build_report(holdings, *, confidence=0.95, shrinkage=False, snapshot=None, live_fetch=None):
    snap = snapshot or data.load_snapshot()
    return report_from_inputs(prepare(holdings, snap, live_fetch=live_fetch),
                              confidence=confidence, shrinkage=shrinkage)
```

Note for the implementer: `json.dumps(..., allow_nan=False)` in the test catches stray numpy scalars. `CrisisResult.missing` and `res.meta` values are plain Python already.

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_report.py -q`
Expected: 13 passed. Note the run time per report. With 10k sims it should be a few seconds; write the time in the commit body.

If `test_bond_heavy_portfolio_scores_below_the_sp500` fails, **stop and report both scores**. That is the calibration check from the spec, not something to tune.

- [ ] **Step 5: Commit.**

```bash
git add risk_engine/config.py risk_engine/report.py tests/test_report.py
git commit -m "Build every dashboard section into one report, with partial sections instead of failures"
```

---

### Task 7: `POST /report` endpoint, response models, gzip

**Files:**
- Modify: `api/schemas.py` (append the report models)
- Modify: `api/main.py` (route + `GZipMiddleware`)
- Create: `tests/test_api_report.py`

**Interfaces:**
- Consumes: `report.build_report`, `inputs.PortfolioError`, `schemas.validate_holdings`
- Produces:
  - `POST /report`, with body `ReportRequest{holdings, confidence=0.95 (0.90..0.99), shrinkage=False}` and response `ReportResponse`. Response errors:
    - 422 `ProblemResponse`
    - 503 `UnavailableResponse`
  - `main._report_or_http_error(holdings, **kw)`. Task 9 wraps it in the compute gate.

- [ ] **Step 1: Write the failing tests** in `tests/test_api_report.py`:

```python
import gzip

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
BODY = {"holdings": {"AAPL": 30000, "JPM": 20000, "TLT": 40000, "IEF": 10000}}


@pytest.fixture(scope="module")
def resp():
    return client.post("/report", json=BODY)


def test_report_returns_every_section(resp):
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {"score", "stress", "monte_carlo", "optimizer", "data"} <= set(body)
    assert body["score"]["horizon_days"] == 252


def test_same_request_twice_is_byte_identical(resp):
    again = client.post("/report", json=BODY)
    assert again.content == resp.content
    shrunk = [client.post("/report", json={**BODY, "shrinkage": True}).content for _ in range(2)]
    assert shrunk[0] == shrunk[1]


def test_report_is_gzipped_when_the_client_accepts_it():
    r = client.post("/report", json=BODY, headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


def test_unknown_ticker_is_a_422_with_problems():
    r = client.post("/report", json={"holdings": {"AAPL": 1000, "XYZQQ": 500}})
    assert r.status_code == 422
    assert r.json()["detail"]["problems"][0]["ticker"] == "XYZQQ"


def test_confidence_out_of_range_is_a_422():
    assert client.post("/report", json={**BODY, "confidence": 0.5}).status_code == 422
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_api_report.py -q`
Expected: FAIL with 404 / 405 on `/report`.

- [ ] **Step 3: Implement.** Append to `api/schemas.py`:

```python
class ReportRequest(BaseModel):
    holdings: dict[str, float]
    confidence: float = Field(default=0.95, ge=0.90, le=0.99)
    shrinkage: bool = False     # Ledoit-Wolf covariance for the optimizer

    @field_validator("holdings")
    @classmethod
    def _check(cls, v):
        return validate_holdings(v)


class Series(BaseModel):
    dates: list[str]
    values: list[float]


class ScoreOut(BaseModel):
    score: int
    bad_year_loss: float
    loss_dollars: float
    gain_dollars: float


class ScoreSection(BaseModel):
    portfolio: ScoreOut
    benchmark: ScoreOut
    horizon_days: int


class HoldingOut(BaseModel):
    ticker: str
    name: str
    sector: str
    type: str
    asset_class: str
    amount: float
    weight: float
    source: str


class PortfolioSection(BaseModel):
    value: float
    holdings: list[HoldingOut]


class RiskSection(BaseModel):
    confidence: float
    hist_var: float
    hist_cvar: float
    gaussian_var: float
    gaussian_cvar: float
    dollars: dict[str, float]
    gap_dollars: float


class ProfileSection(BaseModel):
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    trading_days: int


class CompareRow(BaseModel):
    label: str
    portfolio: float
    benchmark: float
    higher_is_better: bool
    kind: str


class BenchmarkSection(BaseModel):
    rows: list[CompareRow]
    beta: float
    growth: dict[str, Series]


class DistributionSection(BaseModel):
    edges: list[float]
    density: list[float]
    mean: float
    std: float


class CrisisOut(BaseModel):
    label: str
    context: str
    start: str
    end: str
    status: str
    missing: list[str]
    total: Optional[float]
    worst_day: Optional[float]
    max_dd: Optional[float]
    loss_dollars: Optional[float]
    path: Optional[Series]


class CorrelationSection(BaseModel):
    tickers: list[str]
    matrix: list[list[float]]


class BacktestSeries(BaseModel):
    dates: list[str]
    var: list[float]
    realized: list[float]
    breach: list[int]


class BacktestSection(BaseModel):
    window: int
    min_observations: int
    methods: dict[str, BacktestRow]
    series: BacktestSeries


class RiskShare(BaseModel):
    ticker: str
    weight: float
    risk_share: float


class RiskContributionSection(BaseModel):
    rows: list[RiskShare]
    biggest_driver: str
    diversifier: Optional[str]


class FactorsSection(BaseModel):
    available: bool
    reason: str
    data_through: Optional[str]
    betas: Optional[dict[str, float]]
    alpha_annual: Optional[float]
    r2: Optional[float]
    variance_split: Optional[dict[str, float]]
    observations: int


class SectorMixOut(BaseModel):
    sectors: dict[str, float]
    equity: float
    diversified: float
    unplaced: float
    non_equity: dict[str, float]
    top_sector: Optional[str]
    top_sector_pct: float
    concentrated: bool


class OptPoint(BaseModel):
    ret: float
    vol: float
    sharpe: float
    weights: dict[str, float]


class FrontierPoint(BaseModel):
    vol: float
    ret: float


class ReallocRow(BaseModel):
    ticker: str
    type: str
    sector: str
    current: float
    suggested: float
    delta_dollars: float


class OptimizerSection(BaseModel):
    available: bool
    reason: str
    shrinkage: bool
    sector_mix: SectorMixOut
    current: Optional[OptPoint]
    max_sharpe: Optional[OptPoint]
    min_variance: Optional[OptPoint]
    frontier: list[FrontierPoint]
    reallocation: list[ReallocRow]
    adds: list[str]
    trims: list[str]


class HorizonOut(BaseModel):
    label: str
    days: int
    p5: float
    p50: float
    p95: float
    prob_loss: float
    prob_up20: float


class MonteCarloSection(BaseModel):
    n_sims: int
    seed: int
    value: float
    band_days: list[int]
    bands: dict[str, list[float]]
    horizons: list[HorizonOut]


class RejectedOut(BaseModel):
    ticker: str
    amount: float
    reason: str


class ReportData(BaseModel):
    as_of: str
    window_start: str
    window_end: str
    window_days: int
    window_status: str
    window_message: str
    rejected: list[RejectedOut]
    assumptions: list[str]


class ReportResponse(BaseModel):
    score: ScoreSection
    portfolio: PortfolioSection
    risk: RiskSection
    profile: ProfileSection
    benchmark: BenchmarkSection
    distribution: DistributionSection
    drawdown: Series
    stress: list[CrisisOut]
    correlation: CorrelationSection
    backtest: BacktestSection
    risk_contribution: RiskContributionSection
    factors: FactorsSection
    optimizer: OptimizerSection
    monte_carlo: MonteCarloSection
    data: ReportData
```

Then make these edits to `api/main.py`:

- Add these imports:
  - `from fastapi.middleware.gzip import GZipMiddleware`
  - `from risk_engine.report import build_report`
  - `ReportRequest, ReportResponse` in the schemas import
- Import `PortfolioError` from `risk_engine.inputs`. The analyze re-export also works, but import from the source.
- Right after `app = FastAPI(...)`, add:

```python
app.add_middleware(GZipMiddleware, minimum_size=1000)
```

- Add the route:

```python
def _report_or_http_error(holdings, **kwargs):
    try:
        return build_report(holdings, **kwargs)
    except PortfolioError as e:
        raise HTTPException(status_code=422, detail={"problems": e.problems})
    except data.SnapshotUnavailable as e:
        raise HTTPException(status_code=503, detail=f"market data is unavailable: {e}")


@app.post("/report", response_model=ReportResponse, responses={
    422: {"model": ProblemResponse,
          "description": "The holdings were refused: `detail.problems` gives one "
                         "`{ticker, reason}` per ticker."},
    503: {"model": UnavailableResponse, "description": "No market-data snapshot could be read."}})
def report(req: ReportRequest):
    """Everything the dashboard shows for one portfolio, in one call: the 1-99 risk score
    beside the S&P 500's, the bad-year dollar range, and every detail section."""
    return _report_or_http_error(req.holdings, confidence=req.confidence, shrinkage=req.shrinkage)
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_api_report.py tests/test_api.py -q`
Expected: all pass. Record the gzipped size of the fixture response, measured with `len(gzip.compress(resp.content))` in a scratch session, in the commit body.

- [ ] **Step 5: Commit.**

```bash
git add api/schemas.py api/main.py tests/test_api_report.py
git commit -m "Serve the full report from POST /report with gzip"
```

---

### Task 8: `GET /universe` and `POST /import`

**Files:**
- Modify: `api/schemas.py`, `api/main.py`
- Modify: `tests/test_api_report.py` (append)

**Interfaces:**
- Consumes:
  - `snap.universe` (ticker → `{name, sector, asset_class, in_sp500, ...}`)
  - `snap.quarantined`
  - `importer.parse_csv(text, *, last_price)`, `importer.parse_paste(text)`, both returning `ParseResult(holdings, notes)`
- Produces:
  - `GET /universe -> list[UniverseRow{ticker, name, sector, asset_class, in_sp500}]`, sorted by ticker
  - `POST /import` with body `ImportRequest{format: "csv"|"paste", text: str (max 200_000)}`, returning `ImportResponse{holdings, notes}`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_api_report.py`):

```python
def test_universe_lists_tickers_with_names_sorted():
    rows = client.get("/universe").json()
    tickers = [r["ticker"] for r in rows]
    assert tickers == sorted(tickers) and "AAPL" in tickers and "TLT" in tickers
    tlt = next(r for r in rows if r["ticker"] == "TLT")
    assert tlt == {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF",
                   "sector": "Govt Bonds", "asset_class": "Bond", "in_sp500": False}


def test_import_paste_returns_holdings_and_notes():
    r = client.post("/import", json={"format": "paste", "text": "AAPL 10000\nMSFT 5,000\ngarbage"})
    assert r.status_code == 200
    body = r.json()
    assert body["holdings"] == {"AAPL": 10000.0, "MSFT": 5000.0}
    assert any("line 3" in n for n in body["notes"])


def test_import_csv_values_share_counts_at_the_snapshot_close():
    from risk_engine import data
    close = float(data.load_snapshot().prices["AAPL"].dropna().iloc[-1])
    r = client.post("/import", json={"format": "csv", "text": "Symbol,Quantity\nAAPL,10\n"})
    assert r.json()["holdings"]["AAPL"] == pytest.approx(10 * close)


def test_import_rejects_oversized_text_and_unknown_format():
    assert client.post("/import", json={"format": "csv", "text": "x" * 200_001}).status_code == 422
    assert client.post("/import", json={"format": "xlsx", "text": "a"}).status_code == 422
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_api_report.py -q -k "universe or import"`
Expected: FAIL with 404 / 405.

- [ ] **Step 3: Implement.**

Append to `api/schemas.py`. Add `Literal` to the typing import: `from typing import Literal, Optional`.

```python
class UniverseRow(BaseModel):
    ticker: str
    name: str
    sector: str
    asset_class: str
    in_sp500: bool


class ImportRequest(BaseModel):
    format: Literal["csv", "paste"]
    text: str = Field(max_length=200_000)


class ImportResponse(BaseModel):
    holdings: dict[str, float]
    notes: list[str]
```

Make these edits to `api/main.py`:
- Import `importer` from `risk_engine`.
- Import the three models.
- Add a snapshot helper and use it in both routes:

```python
def _snapshot_or_503():
    try:
        return data.load_snapshot()
    except data.SnapshotUnavailable as e:
        raise HTTPException(status_code=503, detail=f"market data is unavailable: {e}")


@app.get("/universe", response_model=list[UniverseRow])
def universe():
    """Every ticker in today's dataset, for search-as-you-type. Tickers outside it still
    work in /report through a live lookup."""
    snap = _snapshot_or_503()
    return [{"ticker": t, "name": m["name"], "sector": m["sector"],
             "asset_class": m["asset_class"], "in_sp500": bool(m.get("in_sp500", False))}
            for t, m in sorted(snap.universe.items())
            if t in snap.prices.columns and t not in snap.quarantined]


@app.post("/import", response_model=ImportResponse)
def import_holdings(req: ImportRequest):
    """Parse a broker CSV export or pasted 'TICKER AMOUNT' lines into holdings. Share
    counts are valued at the dataset's latest close; nothing is analysed here."""
    if req.format == "paste":
        parsed = importer.parse_paste(req.text)
    else:
        snap = _snapshot_or_503()

        def last_price(sym):
            if sym not in snap.prices.columns:
                return None
            series = snap.prices[sym].dropna()
            return float(series.iloc[-1]) if len(series) else None

        parsed = importer.parse_csv(req.text, last_price=last_price)
    return {"holdings": parsed.holdings, "notes": parsed.notes}
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_api_report.py -q`
Expected: all pass. If the paste-note assertion fails because the note wording differs, read `importer.parse_paste` and assert on its actual `line 3` note. Do not change the importer.

- [ ] **Step 5: Commit.**

```bash
git add api/schemas.py api/main.py tests/test_api_report.py
git commit -m "Add the ticker universe and holdings import endpoints"
```

---

### Task 9: Rate limits, compute gate, CORS

**Files:**
- Create: `api/limits.py`
- Create: `tests/test_limits.py`
- Modify: `api/main.py` (middleware order matters, see below; wrap `/report` and `/analyze` in the gate)
- Modify: `tests/conftest.py` (reset the limiter between tests)

**Interfaces:**
- Produces:
  - `limits.RateLimiter(limits: dict[str, int], default: int, window_s=60.0, clock=time.monotonic, max_keys=10_000)`
    - `.check(ip, path) -> float | None`: seconds until retry, or None if allowed
    - `.reset()`
  - `limits.ComputeGate(slots=2, wait_s=15.0)`
    - `.slot()` is a context manager that raises `limits.Busy` when no slot frees in time
  - `limits.client_ip(request) -> str`
  - Module globals in `api.main`: `limiter`, `gate`, `ALLOWED_ORIGINS`

- [ ] **Step 1: Write the failing tests** in `tests/test_limits.py`:

```python
import threading

import pytest
from fastapi.testclient import TestClient

from api import limits, main


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_fixed_window_allows_the_limit_then_says_when_to_retry():
    clock = Clock()
    rl = limits.RateLimiter({"/report": 2}, default=5, clock=clock)
    assert rl.check("1.1.1.1", "/report") is None
    assert rl.check("1.1.1.1", "/report") is None
    clock.t = 20.0
    assert rl.check("1.1.1.1", "/report") == pytest.approx(40.0)
    assert rl.check("2.2.2.2", "/report") is None          # per IP
    assert rl.check("1.1.1.1", "/health") is None          # per bucket
    clock.t = 60.0
    assert rl.check("1.1.1.1", "/report") is None          # new window


def test_limiter_memory_is_bounded_under_many_ips():
    clock = Clock()
    rl = limits.RateLimiter({}, default=5, clock=clock, max_keys=100)
    for i in range(1000):
        rl.check(f"10.0.{i // 256}.{i % 256}", "/x")
        clock.t += 0.1
    assert len(rl._hits) <= 101


def test_client_ip_uses_the_proxy_added_last_entry():
    class Req:
        headers = {"x-forwarded-for": "6.6.6.6, 9.9.9.9"}
        client = None
    assert limits.client_ip(Req()) == "9.9.9.9"

    class Direct:
        headers = {}
        client = type("C", (), {"host": "127.0.0.1"})()
    assert limits.client_ip(Direct()) == "127.0.0.1"


def test_compute_gate_raises_busy_when_full():
    gate = limits.ComputeGate(slots=1, wait_s=0.01)
    with gate.slot():
        with pytest.raises(limits.Busy):
            with gate.slot():
                pass
    with gate.slot():
        pass                                                  # released


ORIGIN = {"Origin": "http://localhost:5173"}


def test_over_the_limit_is_a_429_that_still_carries_cors(monkeypatch):
    monkeypatch.setattr(main, "limiter", limits.RateLimiter({"/health": 1}, default=100))
    client = TestClient(main.app)
    assert client.get("/health", headers=ORIGIN).status_code == 200
    r = client.get("/health", headers=ORIGIN)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_preflight_allows_the_dev_origin_and_not_others():
    client = TestClient(main.app)
    ok = client.options("/report", headers={**ORIGIN, "Access-Control-Request-Method": "POST",
                                            "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_a_full_compute_gate_returns_503_busy(monkeypatch):
    gate = limits.ComputeGate(slots=1, wait_s=0.01)
    monkeypatch.setattr(main, "gate", gate)
    client = TestClient(main.app)
    with gate.slot():
        r = client.post("/report", json={"holdings": {"AAPL": 1000, "MSFT": 1000}}, headers=ORIGIN)
    assert r.status_code == 503 and r.json() == {"detail": "busy"}
    assert r.headers["retry-after"] == "10"
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
```

Append to `tests/conftest.py`. Add `import sys` to its imports.

```python
@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """API tests share one in-process limiter; start every test with a clean window."""
    main = sys.modules.get("api.main")
    if main is not None:
        main.limiter.reset()
    yield
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_limits.py -q`
Expected: FAIL with `ImportError: api.limits`.

- [ ] **Step 3: Implement** `api/limits.py`:

```python
"""Per-IP fixed-window rate limits and a cap on concurrent heavy computations.

In-memory, so they suit one free instance and reset on restart. They stop one
client hogging the service; they are not a defence against a determined attacker."""
import threading
import time
from contextlib import contextmanager


class Busy(Exception):
    """Every compute slot stayed taken for the whole wait."""


class RateLimiter:
    def __init__(self, limits, default, window_s=60.0, clock=time.monotonic, max_keys=10_000):
        self._limits, self._default, self._window = dict(limits), default, window_s
        self._clock, self._max_keys = clock, max_keys
        self._lock = threading.Lock()
        self._hits = {}       # (ip, bucket) -> (window start, count)

    def reset(self):
        with self._lock:
            self._hits.clear()

    def _prune(self, now):
        """Drop expired windows; if that is not enough, drop the oldest half. Dropping an
        IP early only hands it a fresh window (fails open); bounded memory matters more
        on a 512 MB instance."""
        self._hits = {k: v for k, v in self._hits.items() if now - v[0] < self._window}
        if len(self._hits) >= self._max_keys:
            keep = sorted(self._hits.items(), key=lambda kv: kv[1][0])[len(self._hits) // 2:]
            self._hits = dict(keep)

    def check(self, ip, path):
        bucket = path if path in self._limits else "*"
        limit = self._limits.get(path, self._default)
        now = self._clock()
        with self._lock:
            if len(self._hits) >= self._max_keys:
                self._prune(now)
            start, count = self._hits.get((ip, bucket), (now, 0))
            if now - start >= self._window:
                start, count = now, 0
            if count >= limit:
                return self._window - (now - start)
            self._hits[(ip, bucket)] = (start, count + 1)
            return None


class ComputeGate:
    def __init__(self, slots=2, wait_s=15.0):
        self._sem, self._wait = threading.BoundedSemaphore(slots), wait_s

    @contextmanager
    def slot(self):
        if not self._sem.acquire(timeout=self._wait):
            raise Busy()
        try:
            yield
        finally:
            self._sem.release()


def client_ip(request):
    """Render's proxy appends the address it saw; earlier entries are client-supplied."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"
```

In `api/main.py`:
- Add `import os`.
- Add `from fastapi.middleware.cors import CORSMiddleware`.
- Add `from api import limits`.
- **Order is load-bearing.** Starlette makes the **last-added** middleware the outermost. CORS must be outermost, so 429 and 503 responses still carry `Access-Control-Allow-Origin`. Replace the single `GZipMiddleware` line from Task 7 with this block, right after `app = FastAPI(...)`:

```python
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
                   if o.strip()]
limiter = limits.RateLimiter({"/report": 10, "/analyze": 10, "/import": 20}, default=60)
gate = limits.ComputeGate(slots=2, wait_s=15.0)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if request.method != "OPTIONS":
        retry = limiter.check(limits.client_ip(request), request.url.path)
        if retry is not None:
            return JSONResponse(status_code=429, content={"detail": "too many requests"},
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
    return await call_next(request)


app.add_middleware(GZipMiddleware, minimum_size=1000)
# Added last, so it runs outermost: even a 429 or 503 carries CORS headers and the browser
# shows the real error instead of a CORS failure.
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], allow_credentials=False)
```

The middleware reads the module-global `limiter` at call time. That's why `monkeypatch.setattr(main, "limiter", ...)` works.

Gate the two heavy routes. Replace the bodies of `analyze` and `report`:

```python
def _gated(fn, *args, **kwargs):
    try:
        with gate.slot():
            return fn(*args, **kwargs)
    except limits.Busy:
        raise HTTPException(status_code=503, detail="busy", headers={"Retry-After": "10"})
```

- `analyze` body becomes `return _gated(_analyze_or_http_error, req.holdings, confidence=..., include_backtest=...)`.
- `report` body becomes `return _gated(_report_or_http_error, req.holdings, confidence=..., shrinkage=...)`.
- `gate` must be looked up at call time (module global). Do not bind it as a default argument.
- Add `503: {"model": UnavailableResponse, ...}` to `/report`'s and `/analyze`'s docs. The description becomes "No market-data snapshot could be read, or the server is busy (`detail: busy`, retry after the `Retry-After` seconds)."

- [ ] **Step 4: Run the limit tests, then the full suite.**

Run: `python -m pytest tests/test_limits.py -q`, then `python -m pytest -q`.
Expected: all pass. If an existing API test now returns 429, the conftest reset is not running: confirm `api.main` is imported under that exact name. Do not raise the limits.

- [ ] **Step 5: Commit.**

```bash
git add api/limits.py api/main.py tests/test_limits.py tests/conftest.py
git commit -m "Add per-IP rate limits, a two-slot compute gate and CORS for the React dev server"
```

---

### Task 10: Close the `/portfolios` privacy hole

**Files:**
- Modify: `api/store.py` (random ids, new table name, drop `list_portfolios`)
- Modify: `api/schemas.py` (`PortfolioOut.id: str`)
- Modify: `api/main.py` (remove `GET /portfolios`; `pid: str` with a pattern)
- Modify: `tests/test_store.py`, `tests/test_api.py` (portfolio tests)
- Modify: `app.py:223-224` (API showcase text)

**Interfaces:**
- Produces:
  - `store.save_portfolio(name, holdings, path=None) -> str`, a 22-character url-safe token
  - `store.get_portfolio(pid: str, path=None) -> dict | None`
  - `GET /portfolios` no longer exists (405, because the POST route shares the path)

- [ ] **Step 1: Write the failing tests.**

In `tests/test_store.py`:
- Replace `test_list_returns_all_in_order` with the test below.
- Update the other tests where they assume integer ids.

```python
def test_ids_are_random_tokens_not_a_sequence(tmp_path):
    db = str(tmp_path / "t.db")
    a = store.save_portfolio("a", {"AAPL": 1.0}, path=db)
    b = store.save_portfolio("b", {"AAPL": 1.0}, path=db)
    assert isinstance(a, str) and len(a) == 22 and a != b
    assert not hasattr(store, "list_portfolios")
```

In `tests/test_api.py`:
- In `test_portfolio_crud`, replace the line `assert any(r["id"] == pid for r in client.get("/portfolios").json())` with:

```python
    assert isinstance(pid, str) and len(pid) == 22
    assert client.get("/portfolios").status_code == 405       # nobody can list everyone's portfolios
```

- In `test_get_missing_portfolio_404`, use `client.get("/portfolios/AAAAAAAAAAAAAAAAAAAAAA")` (22 A's).
- Add:

```python
def test_portfolio_id_with_bad_characters_is_a_422():
    assert client.get("/portfolios/not a token!").status_code == 422
```

- [ ] **Step 2: Run the tests and confirm they fail.**

Run: `python -m pytest tests/test_store.py tests/test_api.py -q -k "portfolio or store or ids"`
Expected: FAIL (integer ids, the list still exists).

- [ ] **Step 3: Implement.** In `api/store.py`:
- Add `import secrets`.
- Change the module docstring's first line to: `SQLite persistence for saved portfolios. Ids are random tokens, so one person's link can't be guessed from another's, and there is no list-everything query.`

```python
_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS saved_portfolios ("
    "id TEXT PRIMARY KEY, name TEXT NOT NULL, holdings TEXT NOT NULL, created_at TEXT NOT NULL)"
)


def save_portfolio(name, holdings, path=None):
    path = _resolve(path)
    pid = secrets.token_urlsafe(16)            # 22 characters, 128 bits
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(
            "INSERT INTO saved_portfolios (id, name, holdings, created_at) VALUES (?, ?, ?, ?)",
            (pid, name, json.dumps(holdings), datetime.now(timezone.utc).isoformat()),
        )
    return pid
```

- `get_portfolio` changes its table name to `saved_portfolios`.
- Delete `list_portfolios`.

The table is renamed so an old `portfolios` table with integer ids is never read. The Render filesystem is ephemeral, so nothing needs migrating.

In `api/schemas.py`: `PortfolioOut.id: str`.

In `api/main.py`:
- Delete the `get_portfolios` route.
- Change the single-portfolio route signature to `def get_portfolio(pid: str = Path(pattern=r"^[A-Za-z0-9_-]{1,64}$"), analyze: bool = Query(False)):`, importing `Path` from fastapi.
- Wrap `_analyze_or_http_error` in `_gated` there too, like `/analyze`.

In `app.py` around line 223, change the endpoint card:

```python
        ("💾", "POST /portfolios · GET /portfolios/{id}",
         "Save a named portfolio and get back a private id. <code>GET /portfolios/{id}</code> "
         "fetches it; ids are random, and there is no way to list other people's portfolios."),
```

- [ ] **Step 4: Run the tests and confirm they pass.**

Run: `python -m pytest tests/test_store.py tests/test_api.py -q`
Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add api/store.py api/schemas.py api/main.py app.py tests/test_store.py tests/test_api.py
git commit -m "Close the saved-portfolio privacy hole: random ids and no list endpoint"
```

---

### Task 11: Streamlit reads the report, part 1 (pipeline, headline, profile, benchmark, distribution, drawdown, stress, correlation)

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_smoke.py` (only where noted)

**Interfaces:**
- Consumes:
  - `inputs.prepare(..., allow_rejected=True)`, `inputs.PortfolioError`
  - `report.report_from_inputs(inp, confidence=, shrinkage=)`
  - the report dict shape from Task 6

**Rule for this task and the next:**
- Keep every heading, caption, HTML class, colour and chart type exactly as it is.
- Only replace where the numbers come from.
- Each smoke test is the regression guard for its section.

- [ ] **Step 1: Move the shrinkage checkbox up.** The checkbox at app.py:1043-1047 must exist before the report is built. Move that `with st.sidebar: use_shrinkage = st.checkbox(...)` block, unchanged, to directly after the `conf = CONFIDENCE_LEVELS[...]` line, inside the same `with st.sidebar:` block.

- [ ] **Step 2: Replace resolution and the engine block with one cached report.** Replace app.py:419-474. That runs from `with st.spinner("Looking up tickers outside the dataset..."):` through `max_dd = metrics.max_drawdown(pr)`. **Keep** the rejected/notes sidebar box, the snapshot warnings, the stale banner and the share link, reading their data as shown:

```python
@st.cache_data(ttl=3600, show_spinner=False)
def _build_report(holdings_items, conf, shrinkage, as_of, source, _snap, _live_fetch):
    """One report per (holdings, settings, data version). The underscore arguments are
    not hashed: the snapshot is identified by as_of + source."""
    inp = inputs.prepare(dict(holdings_items), _snap, live_fetch=_live_fetch,
                         max_live=LIVE_MAX_UI, budget_s=LIVE_BUDGET_UI_S, allow_rejected=True)
    return report.report_from_inputs(inp, confidence=conf, shrinkage=shrinkage)


with st.spinner("Looking up tickers and running the numbers..."):
    try:
        rep = _build_report(tuple(_parsed.holdings.items()), conf, use_shrinkage,
                            snap.as_of, snap.source, snap, _live)
    except inputs.PortfolioError as e:
        rep, _error = None, e
_rejected = rep["data"]["rejected"] if rep is not None else _error.rejected
```

Then:
- The sidebar "Not included" list is built from `_rejected` (dicts with `ticker`, `amount`, `reason`) instead of `res.rejected`. It is shown **before** the refusal below, so a bad ticker is listed even when it leaves too few holdings.
- After the sidebar box and warnings:

```python
if rep is None:
    if _error.problems[0]["ticker"] == inputs.HOLDINGS_PROBLEM:
        st.warning("Add at least 2 holdings with an amount above $0 to build a portfolio.")
    else:
        st.error(" ".join(p["reason"] for p in _error.problems))
    st.stop()
if rep["data"]["window_status"] == "short":
    st.warning(rep["data"]["window_message"])

P = rep["portfolio"]
selected = [h["ticker"] for h in P["holdings"]]
weights = np.array([h["weight"] for h in P["holdings"]])
port_val = P["value"]
SECTOR = {h["ticker"]: h["sector"] for h in P["holdings"]}
_link = importer.encode_share({h["ticker"]: h["amount"] for h in P["holdings"]})
if st.query_params.get("p") != _link:
    st.query_params["p"] = _link
R, PROF = rep["risk"], rep["profile"]
```

- Import `inputs, report` from `risk_engine`.
- Remove imports that become unused at the end of Task 12, not now.

- [ ] **Step 3: Rewire the sections.** Every `$` figure that came from `x * port_val` on a log return now comes from `R["dollars"]`.

| Section | Old source | New source |
|---|---|---|
| hero "trading days analysed" | `len(ret_sel)` | `PROF["trading_days"]` |
| VaR cards | `abs(h_cvar*port_val)`, `abs(h_var*port_val)`, g_var | `abs(R["dollars"]["hist_cvar"])`, `abs(R["dollars"]["hist_var"])`, `abs(R["dollars"]["gaussian_var"])`; percentages from `R["hist_var"]` etc.; `gap` → `R["gap_dollars"]` |
| profile | `ann_vol, sharpe, max_dd` | `PROF["annual_vol"], PROF["sharpe"], PROF["max_drawdown"]` |
| benchmark | lines 568-584 + `pg/sg` | delete the `if "SPY" in lr.columns:` guard (always true) and dedent; `cmp_rows` from `rep["benchmark"]["rows"]` (label, portfolio, benchmark, higher_is_better, `pct = row["kind"] == "pct"`, `dec = 1 if pct else 2`); `beta = rep["benchmark"]["beta"]`; `pg`, `sg` = `pd.Series(g["values"], index=pd.to_datetime(g["dates"]))` from `rep["benchmark"]["growth"]["portfolio"]` / `["benchmark"]` |
| distribution | `go.Histogram(x=pr*100, nbinsx=80, ...)` | `D = rep["distribution"]`; `go.Bar(x=[(a+b)/2*100 for a,b in zip(D["edges"],D["edges"][1:])], y=[d/100 for d in D["density"]], width=[(b-a)*100 for a,b in zip(D["edges"],D["edges"][1:])], marker_color="#4d9fff", opacity=0.6, name="Actual daily returns")`. Density is divided by 100 because the x axis is in percent. Normal curve: `x_r = np.linspace(D["edges"][0]*100, D["edges"][-1]*100, 400)`, `norm.pdf(x_r, D["mean"]*100, D["std"]*100)`. VaR lines from `R["hist_var"]`, `R["gaussian_var"]`. |
| drawdown | `drawdown` series | `dd = rep["drawdown"]`; `x=pd.to_datetime(dd["dates"])`, `y=np.array(dd["values"])*100` |
| stress | `CRISIS_COLOR`, `stress()`, `_missed`, `results` | delete `stress()` and `CRISIS_COLOR`. Add `CRISIS_PALETTE = ["#ffb347", "#ff3d5a", "#4d9fff", "#05d69e", "#c792ea", "#7fdbca"]`; colour = `CRISIS_PALETTE[i % len(CRISIS_PALETTE)]` by crisis position in `rep["stress"]`. `results = [c for c in rep["stress"] if c["status"] == "ok"]`; card numbers from `c["total"], c["worst_day"], c["max_dd"]`, dollars `abs(c["loss_dollars"])`; chart y = `np.array(c["path"]["values"])*100`, x = `range(len(...))`. `_missed = {c["label"]: c["missing"] for c in rep["stress"] if c["status"] == "not_covered"}`. **New:** crises with status `too_few_days` get the caption `"Not shown, because the data has fewer than 5 clean trading days in the window: " + "; ".join(labels)`. |
| correlation | `ret_sel.corr()` | `C = rep["correlation"]`; `corr = pd.DataFrame(C["matrix"], index=C["tickers"], columns=C["tickers"])`; the rest unchanged |

- [ ] **Step 4: Run the smoke tests.**

Run: `python -m pytest tests/test_app_smoke.py -q`
Expected: all pass, with these exceptions:
- `test_every_crisis_gets_its_own_line_and_colour` must still pass via the palette.
- If a smoke test asserts an exact VaR dollar figure that moved because of `expm1`, update that number. Add a comment: `# expm1 dollars since project A: a log return is not a dollar fraction`.
- Any other failure is a regression: fix `app.py`, not the test.

- [ ] **Step 5: Commit.**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Lay out the top dashboard sections from the shared report"
```

---

### Task 12: Streamlit reads the report, part 2 (backtest, risk contribution, factors, optimizer, Monte Carlo), then delete the dead math

**Files:**
- Modify: `app.py`
- Modify: `tests/test_app_smoke.py` (only where noted)

- [ ] **Step 1: Rewire the remaining sections.**

| Section | Old source | New source |
|---|---|---|
| backtest | `_run_backtest` (app.py:804-812) | delete `_run_backtest`. `B = rep["backtest"]`; `bt_rows = [{"method": m, **row} for m, row in B["methods"].items()]`; `bt_graded = all(r["passed"] is not None for r in bt_rows)`; chart from `B["series"]` (`dates` via `pd.to_datetime`, `var`, `realized`, `breach`). Keep the verdict text logic as is. |
| risk contribution | `optimize.sample_cov` / `risk_contribution`, `_gap/_hot/_div` | `RC = rep["risk_contribution"]`; `risk_pct = np.array([r["risk_share"] for r in RC["rows"]])`; `_hot = selected.index(RC["biggest_driver"])`; `_div_txt` only if `RC["diversifier"]` is not None, with `_div = selected.index(RC["diversifier"])` |
| factors | `snap.factors`, `fac.factor_regression`, `variance_attribution`, `_factor_days` gate | `F = rep["factors"]`. If `not F["available"]`, show `st.info(F["reason"])`, the existing branch's message (smoke test expects "at least 60"). Otherwise betas, alpha, r², variance split come from `F`. The caption "Factor data through …" uses `F["data_through"]`. Keep the tilt-text thresholds exactly. |
| optimizer | lines 1049-1172 | `O = rep["optimizer"]`. `opt_ok = O["available"]`; when false, `st.info("Optimizer couldn't converge for this selection — try different holdings. (" + O["reason"] + ")")`. Card numbers from `O["current"]`, `O["max_sharpe"]`, `O["min_variance"]` (`ret`, `vol`, `sharpe`). Frontier `fr_v = [p["vol"]*100 for p in O["frontier"]]`, `fr_r = [p["ret"]*100 ...]`. Reallocation table from `O["reallocation"]` (Holding, Type, Sector, Now, Suggested, Change, formatted as today). Sector text from `O["sector_mix"]`: `top_sec = SM["top_sector"]`, `top_sec_pct = SM["top_sector_pct"]*100`, `mix.equity` → `SM["equity"]` etc. `add_idx`/`trim_idx` → `O["adds"]` / `O["trims"]` (add text keeps `SECTOR[t]`). `conc = "danger" if SM["concentrated"] else "warn"`, and the same for the icon. **The diversification box is shown even when the optimizer failed** (sector mix is always available). Move it out of the `else:` branch. When the optimizer failed, omit the "add/trim/lift your Sharpe" sentence. |
| Monte Carlo | lines 1181-1192 | `MC = rep["monte_carlo"]`; `mc_h = HORIZON_DAYS[...]`, with the slider unchanged (`HORIZON_DAYS = {h["label"]: h["days"] for h in MC["horizons"]}`). `H = next(h for h in MC["horizons"] if h["days"] == mc_h)`; `t5, t50, t95, prob_loss, prob_up20` come from `H`. Fan chart: `k = MC["band_days"].index(mc_h)`; `_x = MC["band_days"][:k + 1]`; `bands = [MC["bands"][f"p{p}"][:k + 1] for p in (5, 25, 50, 75, 95)]`. The insight text uses `f"{MC['n_sims']:,}"` instead of "10,000", and `H["label"]` instead of "1 year": "but 1 in 20 times, after {H['label']}, it falls to …". Replace the `{{}}` + `.replace("{}", "50–90%")` hack with the literal "50–90%" in the f-string. |

- [ ] **Step 2: Delete the dead code and imports.**
- Delete every leftover inline computation: `ret_sel`, `pr`, `mu`, `std`, `lr`, `prices`, `_cov_rc`, `mu_v`, `cov_m`, `_rng`, `N_SIMS`, `hist_r`, `_draws`, `sim_paths`, `terminal`.
- Delete the `try/except Exception` around the optimizer (spec bug #3; the report now raises nothing there).
- Remove the imports nothing uses: likely `optimize`, `backtest`, `metrics`, `fac`, `portfolio` and some config names. Check each with grep.
- Then confirm no math is left:

```bash
grep -nE "np\.(percentile|cov|var|cumsum|random)|\.corr\(|optimize\.|backtest\.|metrics\.|factor_regression" app.py
```

Expected: no matches, except `norm.pdf` for drawing the Gaussian curve and `np.array`/`np.linspace` for drawing.

- [ ] **Step 3: Run the smoke tests, then the full suite.**

Run: `python -m pytest tests/test_app_smoke.py -q`, then `python -m pytest -q`.
Expected: everything passes. Same rule as Task 11: only exact-number assertions moved by the documented fixes may change, and each gets a comment.

- [ ] **Step 4: Manual look.**

Run: `streamlit run app.py --server.headless true` in the background. Open `http://localhost:8501`, scroll the whole page and check that every chart renders. Then stop the server. If the browser can't be driven, say so in the report; don't claim it rendered.

- [ ] **Step 5: Commit.**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Lay out the remaining sections from the report and delete the page's inline math"
```

---

### Task 13: Final verification (no push)

- [ ] **Step 1:** Full suite: `python -m pytest -q`. Record the pass count. It should be 370 plus the new tests, all green.
- [ ] **Step 2: Clean-env parity**, the same way CI runs: `python -m pytest -q -p no:cacheprovider` with `PYTHONHASHSEED=0`. Expected: identical results.
- [ ] **Step 3: Time one fixture report** with a 10-ticker and a 25-ticker portfolio:

```bash
python - <<'EOF'
import time, gzip, json, os
os.environ["MARKETPLUG_DATA_DIR"] = "tests/fixtures/snapshot"; os.environ["MARKETPLUG_NO_LIVE"] = "1"
from risk_engine.report import build_report
from risk_engine import data
snap = data.load_snapshot()
tick = [t for t in snap.prices.columns]
for n in (10, 25):
    h = {t: 10000 for t in tick[:n]}
    t0 = time.perf_counter(); r = build_report(h); dt = time.perf_counter() - t0
    print(n, f"{dt:.2f}s", f"{len(gzip.compress(json.dumps(r).encode()))/1024:.0f} KB gz")
EOF
```

  Record the output. Local time is not the gate (Render's CPU is slower), but anything over 8 s locally means stop and report.
- [ ] **Step 4:** Update the spec:
  - Section 4: bands every **6** trading days, 127 points.
  - Section 5: `score.*.bad_year_loss` is a fraction.
  - Section 7 row 8: the showcase was already labelled "computed 2026-08-23", so no change was needed.
  - Section 5: `/analyze` reports a non-converging optimizer as a 422 `(optimizer)` problem.
  - Commit it as `Bring the API growth spec in line with the build`.
- [ ] **Step 5:** Hand back to the controller. **Do not push.** Tam gives the word after ship-check.

---

## After the build (controller, not a task agent)

1. Run **ship-check** on the branch, fix what it finds, then ask Tam for the word to merge and push.
2. Once Render redeploys, measure section 9 of the spec on the live service and write the numbers into the spec:
   - warm p50 and p95 at 10 and 50 tickers
   - peak RSS
   - gzipped size
   - SPY's score
   - the `X-Forwarded-For` layout
3. Apply the 8-second gate. If it trips, split into `/report/summary` and `/report` before project B starts.
