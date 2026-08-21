# Phase 3 — System Layer: Deployed Risk API

**Date:** 2026-08-21
**Status:** Approved (design), plan pending
**Builds on:** Phase 1 (risk engine) + Phase 2 (factor models), both merged to `main` and live.

**Revision (2026-08-21, post-review against real engine code):** folded in 3 fixes
+ 3 upgrades found by reading the actual modules — (1) `analyze_portfolio` forces
`prefer_live=False` (snapshot only) so no request hits Yahoo; (2) optimizer returns
structured numbers, not the UI-coupled prose that lives in `app.py`; (3) validation
requires 2–12 holdings; (4) SQLite fresh-connection-per-call for thread safety;
(5) `confidence` is an explicit request field (0.90–0.99); (6) self-contained
`requirements-api.txt` so the Streamlit deploy stays lean.

## Goal

Expose the tested `risk_engine` as a documented, deployed REST service. The
resume signal is a Swagger-documented API over a validated risk engine —
"I build the products finance runs on," made literal. Full build (deliberately
chosen over the leaner options): full analysis response, portfolio persistence,
and a live deployed URL.

## The one principle everything obeys

> `risk_engine` is pure. It takes numbers in, returns numbers out, and never
> imports Streamlit, FastAPI, pydantic, or anything with a UI or transport.

Streamlit imports the engine. FastAPI imports the *same* engine. Tests import
the *same* engine. One brain, three faces.

## Architecture

Today `app.py` holds the orchestration glue (load prices → build returns → call
metrics/factors/optimize/backtest → format for UI). The API needs that same
orchestration minus UI formatting. Rather than duplicate it, extract the
orchestration into a new **pure** engine module. The API becomes thin transport.

```
risk_engine/
  analyze.py     NEW — analyze_portfolio(holdings, *, include_backtest) -> plain dict
                 (pure: loads committed data, computes returns, runs
                  metrics + factors + optimizer + backtest, returns JSON-ready dict)
api/
  __init__.py
  main.py        FastAPI app: endpoints, wiring, validation → calls the engine
  schemas.py     pydantic request/response models
  store.py       SQLite persistence via stdlib sqlite3
tests/
  test_analyze.py   pure orchestration unit tests (no FastAPI)
  test_api.py       FastAPI TestClient: 200 + schema-valid, 422 on bad input
  test_store.py     save/load/list round-trip
```

**`app.py` is NOT rewired this phase.** It stays working exactly as-is; the new
`analyze.py` is additive. Deduping the Streamlit UI onto `analyze_portfolio()`
is a clean follow-up, explicitly out of scope here.

## `risk_engine/analyze.py` (the shared brain)

```
analyze_portfolio(holdings: dict[str, float], *, confidence: float = 0.95,
                  include_backtest: bool = True) -> dict
```

- `holdings`: ticker → dollar amount (e.g. `{"AAPL": 20000, "TLT": 10000}`).
- `confidence`: VaR/backtest confidence level (0.90–0.99), passed through to
  `metrics` and `backtest`.
- **Loads the SNAPSHOT ONLY** via `data.load_prices(prefer_live=False)`. This is
  a hard requirement, not a default: `load_prices()` defaults to `prefer_live=True`,
  which loops every ticker against Yahoo with a 15s timeout each — unacceptable
  in a request path (would stall or behave nondeterministically per call, and
  breaks the deployed service's reliability). The engine's committed `prices.csv`
  is the single runtime source. Then restrict to the requested tickers, compute
  dollar weights (amount ÷ total).
- Calls the existing pure functions: `metrics` (historical + Gaussian VaR/CVaR,
  Sharpe, vol, max-drawdown), `factors` (regression + variance attribution),
  `optimize` (max-Sharpe / min-variance weights + Sharpe comparison + sector
  concentration — **numbers, not prose**), and — when `include_backtest` —
  `backtest_var` (Kupiec + Christoffersen on historical & Gaussian VaR).
- Returns a plain, JSON-serializable dict (floats/lists/dicts only). **`analyze.py`
  is the SINGLE conversion point**: the only non-native values from the engine are
  the optimizer weight `np.ndarray`s, which it converts to `{ticker: float}` maps;
  `metrics`/`backtest_var`/`factors.betas` already return native Python types
  (verified). A test asserts the entire returned dict is `json.dumps`-able.
  Structure (illustrative — exact metric keys pinned to real signatures):

```
{
  "metrics":   {"hist_var": ..., "hist_cvar": ..., "gaussian_var": ...,
                "sharpe": ..., "annual_vol": ..., "max_drawdown": ...},
  "factors":   {"betas": {...}, "alpha_annual": ..., "r2": ...,
                "variance_split": {"Mkt-RF": ..., "SMB": ..., "HML": ..., "Idiosyncratic": ...}},
  "optimizer": {"current_sharpe": ..., "max_sharpe_weights": {...}, "max_sharpe_value": ...,
                "min_variance_weights": {...}, "top_sector": "...", "top_sector_pct": ...},
  "backtest":  {"historical": {"breaches": ..., "expected": ..., "kupiec_p": ...,
                               "christoffersen_p": ..., "passed": ...},
                "gaussian":   {...}}          # key omitted entirely when include_backtest is False
}
```

**No prose in the response.** The plain-English diversification "suggestion" lives
in `app.py` (UI-coupled, HTML-flavored, ~lines 917–954) and is NOT extracted. A
JSON API returns data; turning numbers into sentences is the UI's job.

- Pure and independently testable: `test_analyze.py` asserts a known portfolio
  produces sane, finite values and that `include_backtest=False` omits the
  backtest key.
- **Exact keys are pinned to the real engine, not invented.** The metric keys
  (which VaR — historical vs Gaussian — CVaR, Sharpe, vol, drawdown) and the
  known-ticker universe used for validation come from the actual signatures of
  `metrics`, `optimize`, `backtest`, and the config/data universe constant. The
  implementation plan reads those modules and pins the response shape to what
  they truly return. The dict above is illustrative of structure, not a
  contract on names.

## API — `api/main.py`

| Method | Path | Behavior |
|---|---|---|
| POST | `/analyze` | validate body → `analyze_portfolio` → `AnalyzeResponse` |
| POST | `/portfolios` | save a named portfolio → `{id}` |
| GET | `/portfolios` | list saved (id, name, created_at) |
| GET | `/portfolios/{id}` | fetch one; `?analyze=true` also runs analysis |
| GET | `/health` | `{"status": "ok"}` (liveness for the host) |
| — | `/docs` | auto-generated Swagger UI — the resume artifact |

Handlers are thin: parse → validate → call engine/store → return a pydantic
model. No business logic in the transport layer.

## `api/schemas.py` (pydantic)

- `AnalyzeRequest {holdings: dict[str, float], confidence: float = 0.95,
  include_backtest: bool = True}`.
- `AnalyzeResponse` mirrors the `analyze_portfolio` dict (typed sub-models:
  `Metrics`, `Factors`, `Optimizer`, `Backtest` optional).
- `PortfolioIn {name, holdings}`, `PortfolioOut {id, name, holdings, created_at}`.
- **Validation → clean 422** (input validation is free credibility):
  - unknown ticker (outside `config.TICKERS`, the engine's known universe) → 422
  - any amount ≤ 0 → 422
  - fewer than 2 holdings (empty or single) → 422 — the optimizer and
    correlations are meaningless below 2, and the app enforces 2–12
  - more than 12 holdings → 422
  - `confidence` outside 0.90–0.99 → 422

## `api/store.py` (SQLite, stdlib only)

- `sqlite3`, single table `portfolios(id INTEGER PK, name TEXT, holdings TEXT
  JSON, created_at TEXT)`.
- `save_portfolio(name, holdings) -> id`, `get_portfolio(id) -> row | None`,
  `list_portfolios() -> list`.
- **Fresh connection per call** (`with sqlite3.connect(path) as conn: ...`), not a
  shared module-level connection. FastAPI runs sync handlers in a threadpool and
  a sqlite3 connection cannot cross threads — a shared connection throws
  intermittently under concurrency. Per-call connect is the simple safe pattern
  at this scale.
- DB path configurable (default `portfolios.db` at repo root, git-ignored);
  tests use a temp DB via a fixture. No ORM, no migrations.

## Tests

- `test_analyze.py` — pure: known portfolio → finite metrics; backtest toggle.
- `test_api.py` — FastAPI TestClient: `/analyze` 200 + schema-valid on good
  input; 422 on each bad-input case; `include_backtest=false` omits backtest;
  `/health` 200.
- `test_store.py` — save → get round-trips; list returns saved rows; temp DB.
- All added to the existing suite; CI (`.github/workflows/ci.yml`) runs them.

## Dependencies

- **New `requirements-api.txt`, self-contained for the API deploy**: the engine's
  numeric deps (`numpy`, `pandas`, `scipy`) PLUS `fastapi`, `uvicorn[standard]`,
  `pydantic`. Deliberately excludes `streamlit`/`plotly` — the API never renders a
  UI. Render installs ONLY this file and gets exactly what the API needs.
- `requirements.txt` (Streamlit UI) is left as-is — it keeps `streamlit`/`plotly`
  and is NOT polluted with the web framework.
- **`ci.yml` updated** to install `requirements.txt` + `requirements-api.txt` +
  `requirements-dev.txt` (CI runs the full suite: engine + Streamlit smoke + API).
- Dev (add to `requirements-dev.txt`): `httpx` (TestClient transport).
- No change to the pure engine's dependency profile — the engine still imports
  only numpy/pandas/scipy.

## Deploy — Render free tier

- Build: `pip install -r requirements-api.txt` (self-contained — includes the
  engine's numpy/pandas/scipy, excludes streamlit/plotly).
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- Config committed (`render.yaml` or documented README steps).
- Live `/docs` URL for the resume; `/health` for the host's checks.

### Honest caveats (documented, not hidden)

1. **Cold start** — free tier sleeps; first request after idle takes ~30–50s.
2. **Ephemeral DB** — free-tier filesystem resets on redeploy/sleep, so saved
   portfolios do not persist long-term. Demonstrates the capability; a
   production deploy would use a managed database. Stated plainly in the README.
3. **Backtest cost** — per-request backtesting rolls a 250-day window across
   years of history (seconds). The `include_backtest` flag (default true) lets
   callers opt out for a fast response.

## Done when

API (`/analyze` + portfolios CRUD + `/health`) + Swagger `/docs` + pydantic
validation + full test suite green in CI + live deployed URL + README section
documenting run/deploy and the three caveats. Live Streamlit app unaffected
throughout (separate deploy).

## Explicitly out of scope (YAGNI)

- Rewiring `app.py` to use `analyze_portfolio` (follow-up).
- Auth / API keys / rate limiting.
- Managed database / durable persistence.
- Async engine calls, caching layers, background workers.
- Any new modeling — the engine is frozen for this phase.
