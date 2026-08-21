# Phase 3 — System Layer: Deployed Risk API

**Date:** 2026-08-21
**Status:** Approved (design), plan pending
**Builds on:** Phase 1 (risk engine) + Phase 2 (factor models), both merged to `main` and live.

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
analyze_portfolio(holdings: dict[str, float], *, include_backtest: bool = True) -> dict
```

- `holdings`: ticker → dollar amount (e.g. `{"AAPL": 20000, "TLT": 10000}`).
- Loads the committed price snapshot (via existing `data` module), restricts to
  the requested tickers, computes log returns and dollar weights.
- Calls the existing pure functions: `metrics` (VaR/CVaR/Sharpe/vol/max-drawdown),
  `factors` (regression + variance attribution), `optimize` (max-Sharpe /
  min-variance / suggestion), and — when `include_backtest` — `backtest`
  (Kupiec + Christoffersen on historical & Gaussian VaR).
- Returns a plain, JSON-serializable dict (floats/lists/dicts only — no numpy
  scalars, no pandas objects). Structure:

```
{
  "metrics":   {"var": ..., "cvar": ..., "sharpe": ..., "vol": ..., "max_drawdown": ...},
  "factors":   {"betas": {...}, "alpha_annual": ..., "r2": ...,
                "variance_split": {"Mkt-RF": ..., "SMB": ..., "HML": ..., "Idiosyncratic": ...}},
  "optimizer": {"max_sharpe_weights": {...}, "min_variance_weights": {...}, "suggestion": "..."},
  "backtest":  {"historical": {"kupiec_p": ..., "christoffersen_p": ..., "verdict": "..."},
                "gaussian":   {...}}          # key omitted entirely when include_backtest is False
}
```

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

- `Holding` or a `holdings: dict[str, float]` map; `AnalyzeRequest {holdings,
  include_backtest: bool = True}`.
- `AnalyzeResponse` mirrors the `analyze_portfolio` dict (typed sub-models:
  `Metrics`, `Factors`, `Optimizer`, `Backtest` optional).
- `PortfolioIn {name, holdings}`, `PortfolioOut {id, name, holdings, created_at}`.
- **Validation → clean 422** (input validation is free credibility):
  - unknown ticker (outside the engine's known universe) → 422
  - any amount ≤ 0 → 422
  - empty holdings → 422
  - more than 12 holdings → 422

## `api/store.py` (SQLite, stdlib only)

- `sqlite3`, single table `portfolios(id INTEGER PK, name TEXT, holdings TEXT
  JSON, created_at TEXT)`.
- `save_portfolio(name, holdings) -> id`, `get_portfolio(id) -> row | None`,
  `list_portfolios() -> list`.
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

- Runtime (add to `requirements.txt`): `fastapi`, `uvicorn[standard]`, `pydantic`.
- Dev (add to `requirements-dev.txt`): `httpx` (TestClient transport).
- No change to the pure engine's dependency profile — the engine still imports
  only numpy/pandas/scipy.

## Deploy — Render free tier

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
