# Portfolio Risk Dashboard — Scale-Up Design

**Date:** 2026-08-20
**Status:** Approved design, pending pre-implementation verification
**Goal:** Take the shipped Streamlit dashboard from "smart student project" to "serious portfolio flagship" — depth of risk methods + real engineering structure, in three sequenced phases.

---

## The one principle everything obeys

> The `risk_engine` is pure. It takes numbers in, returns numbers out, and never
> imports Streamlit, FastAPI, or anything with a UI.

Streamlit imports the engine. FastAPI imports the *same* engine. Tests import the
*same* engine. One brain, three faces. The moment engine code knows what a web
page is, the whole thing rots back into a monolith.

---

## Target structure

```
portfolio-risk-dashboard/
├── risk_engine/
│   ├── __init__.py
│   ├── config.py         # universe, sector/asset-class maps, constants
│   ├── data.py           # price loading (urllib live + CSV fallback) → returns
│   ├── metrics.py        # VaR/CVaR: historical, gaussian, student-t, (EVT); sharpe, drawdown
│   ├── backtest.py       # rolling VaR backtest + Kupiec POF + Christoffersen CC   ← headline
│   ├── optimize.py       # markowitz + Ledoit-Wolf shrinkage, frontier, min-var, max-sharpe
│   ├── factors.py        # (P2) Fama-French load + OLS regression + variance attribution
│   └── (stress/simulate/contribution stay in app.py until a phase needs them)
├── api/                  # (P3)
│   ├── main.py           # FastAPI app — thin transport over risk_engine
│   ├── schemas.py        # pydantic request/response models
│   └── store.py          # SQLite save/load (optional garnish)
├── tests/
│   ├── conftest.py       # fixtures: synthetic series with KNOWN statistical properties
│   ├── test_golden.py    # refactor guard: outputs identical to pre-refactor
│   ├── test_metrics.py
│   ├── test_backtest.py
│   ├── test_optimize.py
│   └── test_factors.py
├── app.py                # thin Streamlit skin over risk_engine
├── prices.csv            # existing snapshot
├── factors.csv           # (P2) cached Fama-French, committed like prices.csv
└── .github/workflows/ci.yml  # pytest on push → green badge
```

---

## Phase 1 — Credible risk engine (foundation + biggest single win)

### Step 1 — Refactor, behavior-preserving (TWO steps, not one)
**Reality check (verified against actual `app.py`, 2026-08-20):** the file is a
single 956-line *linear script*. Almost all computation is inline top-level code
wired directly to Streamlit widget values (`selected`, `weights`, `conf`). Only
four functions exist: `load()`, `stress()`, `_perf()`, `_vs()`. There is nothing
to "just extract" — the math must first be lifted out of the inline flow.

So the refactor is two moves:

1. **Lift-in-place** — convert the inline math (VaR/CVaR at lines ~331–346,
   optimizer/frontier ~711–803, risk contribution ~670–677, beta ~446–456,
   Monte Carlo ~871–881) into *pure functions inside `app.py`*, behavior
   unchanged.
2. **Move out** — relocate those now-clean functions into `risk_engine/`.

Extract ONLY what the new work touches: `data`, `metrics`, `optimize`, plus new
`backtest`. Leave stress/simulate/contribution inline until a phase needs them.

Guard the whole thing with a **golden-master test** (mandatory, not optional):
capture current outputs for a fixed portfolio *before* step 1, assert identical
*after* each step. This is the seatbelt that keeps the live URL working while the
guts are replaced.

**Free cleanup:** the covariance matrix is currently computed three times
(lines 670, 719, and inside the beta calc). Extraction de-duplicates it.

### Step 2 — The methods that change how it reads

| Feature | What it is | Why it's the tell |
|---|---|---|
| **VaR backtesting** | Rolling 250-day window; each day, did realized loss breach the VaR? Count breaches vs expected. **Kupiec POF** (right breach *rate*) + **Christoffersen** (breaches don't *cluster*). | The #1 "you actually understand risk" signal. Almost no student project has it. |
| **Student-t VaR** | Fit fat-tailed t-distribution alongside historical + gaussian; compare at 95/99%. | Shows you know gaussian underestimates tails. |
| **Ledoit-Wolf shrinkage** | Replace sample covariance in the optimizer with the shrinkage estimator. | Kills the "Markowitz spits garbage weights" objection before it's raised. |

**UI added:** a "Model Validation" section — breach-timeline chart with markers,
and a table: *method → breaches → expected → Kupiec p → Christoffersen p →
PASS/FAIL*. That table is the artifact to screenshot for the README.

**Tests:** VaR of a standard-normal sample ≈ analytical z-quantile; a correctly
specified VaR passes Kupiec ~95% of the time on simulated data; shrinkage
covariance is PSD and min-var vol ≤ any random portfolio's.

**Done when:** engine extracted + tests green + validation section live + pushed +
README gets a "Model Validation" subsection.

---

## Phase 2 — Factor models (deepens the finance story)

- **Data:** Fama-French 3-factor (Mkt-RF, SMB, HML) + RF. Download once,
  **commit `factors.csv`** — same snapshot pattern as prices. App never fetches live.
- **Engine (`factors.py`):** OLS regress `r_p − rf` on the three factors → betas,
  annualized alpha, R². Then **variance attribution**: split portfolio variance
  into market / size / value / idiosyncratic.
- **UI:** exposure table + variance-decomposition bar + the money sentence:
  *"87% of your risk is plain market beta; you carry a small-cap tilt and a
  negative value tilt — you're betting on growth."*
- **Guardrail:** 3 factors, one section, **zero ML, no rolling-beta animations.**
- **Tests:** build returns *as* a known linear combo of factors + noise → regression
  must recover the betas.

**Done when:** factor section live + tests + pushed.

---

## Phase 3 — Make it a system (the SWE layer)

- **`api/main.py` (FastAPI):** `POST /analyze {holdings}` → full risk JSON, importing
  the *same* engine. `POST/GET /portfolios` for save/load.
- **`schemas.py`:** pydantic models — input validation is free credibility; bad
  input → clean 422.
- **`store.py`:** SQLite via stdlib. Optional garnish, not the point.
- **The actual signal:** the auto-generated **Swagger `/docs` page** backed by the
  tested engine. Screenshot for the resume — "documented REST service over a
  validated risk engine."
- **Deploy:** minimal. Free-tier host or documented-runnable. Not a deploy-config fight.
- **Tests:** FastAPI TestClient — 200 + schema-valid on good input, 422 on bad,
  save/load round-trips.

**Done when:** API + Swagger + tests + pushed + minimal deploy or documented run.

**Honest note:** Phase 3 is the lowest signal-per-hour of the three. It survives
because (a) all three were chosen deliberately and (b) a Swagger-documented service
is a legit line for SWE intern roles. Planned lean, last, and trimmable without shame.

---

## Cross-cutting decisions

| Decision | Choice | Reason |
|---|---|---|
| Test framework | pytest (built from zero — no tests exist today) | pytest for engine; optional AppTest UI smoke later |
| CI | GitHub Actions → green badge | cheap, reads as "real engineering" |
| New deps | **avoid** sklearn & statsmodels | hand-roll Ledoit-Wolf (~15 lines) + `numpy.linalg.lstsq` for OLS; `scipy.stats` covers t / chi2 |
| P3 deps | FastAPI + uvicorn + pydantic | only introduced in Phase 3 |
| Live URL | stays working every phase | work on branches, merge only when green + shipped |

---

## Decisions put on trial (self-review outcomes)

1. **Whole-app refactor → WRONG.** Scoped down to only `data`/`metrics`/`optimize`/`backtest`. Stress/simulate/contribution stay in `app.py` until needed.
2. **EVT / GPD tail fitting → DEMOTED to stretch.** Statistically shaky on ~250–1000 daily returns; threshold selection is a rabbit hole. Student-t is the committed fat-tail method; EVT optional with a fixed threshold + honest caveat, or cut.
3. **Backtesting headline → RIGHT, but depends on data span.** Needs years of daily history. → verification item.
4. **Fetch Fama-French live → WRONG.** SSL interception on Tam's network already broke yfinance. Snapshot `factors.csv` instead.
5. **Phase 3 → kept but demoted.** Lowest signal; lean, last, trimmable.
6. **Order 1 → 2 → 3 → holds.** Finance depth before plumbing. Swappable if desired.
7. **Whole design rests on a 16-day-old memory of `app.py`.** Architecture is sound; exact extraction map is not real until the file is read.

---

## Pre-implementation verifications — DONE (2026-08-20)

1. **Read `app.py` for real** — ✅ done. Finding: it's a linear script, not a set of
   functions. Refactor is two-step (lift-in-place → move out), harder than first
   scoped. Cov matrix computed 3×. Details folded into Phase 1 above.
2. **Check `prices.csv` date span** — ✅ PASS. 2018-01-02 → 2026-06-18, 2,127 daily
   rows (~8.5 yrs), 25 tickers + SPY. More than enough for rolling VaR backtesting.
3. **Confirm existing AppTest setup** — ✅ done. Finding: no tests exist at all. The
   suite is built from zero (corrected in Cross-cutting table).

---

## Sequencing summary

**P1 is the project** — foundation + biggest credibility win, ships on its own.
**P2** deepens the finance story. **P3** is SWE garnish — lean, cut without guilt if
time runs short. EVT demoted, factor data snapshotted, refactor scoped tight, live
URL protected throughout. Each phase ships and commits before the next begins.
