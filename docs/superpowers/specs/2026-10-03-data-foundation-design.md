# Sub-project 1 — Data Foundation: S&P 500 Universe, Daily Refresh, Real Import

**Date:** 2026-10-03
**Status:** Approved (design), plan pending
**Builds on:** the 3-phase scale-up (risk engine, factor models, deployed API), all on `main` and live.

## Where this sits

MarketPlug is being turned from a portfolio project into a tool someone would
actually use. Product thesis, agreed 2026-10-03:

> MarketPlug answers one question honestly: what actually happens to *my* money
> when it goes wrong, and what would change that.

The work is split into six sub-projects, each with its own spec → plan → build,
each shipped live before the next starts:

| # | Sub-project | Depends on |
|---|---|---|
| **1** | **Data foundation (this spec)** | — |
| 2 | What you really own — ETF look-through, effective holdings, single-stock check + "I work here" | 1 |
| 3 | Learned models — GARCH/regime volatility graded by Kupiec/Christoffersen; return-based clustering of the universe | 1 |
| 4 | Diversification page — where to diversify, cluster-based AI lens, stock-bond regime, cushion vs cost, "stop diversifying" | 1, 3 |
| 5 | Panic replay — day-by-day crash replay, the "sell?" moment, losses in months of expenses | 1 |
| 6 | LLM explanation layer — "ask your portfolio," grounded in engine numbers; key + abuse limits | all |

## Goal

Replace the hand-picked 20-stock universe with the full S&P 500 plus a curated
ETF set, keep the data fresh daily without bloating the repo, let users bring
their real portfolio in (paste / CSV / search) including tickers outside the
universe, and make every data failure visible instead of silent.

## Decisions (with the reason each was made)

| Decision | Choice | Why |
|---|---|---|
| Target user | Investor holding US-listed securities | Engine is built for US data (FF factors, SPY, US bond ETFs); fits the global audience |
| Universe | **Open**: S&P 500 + curated ETFs in the snapshot, any other USD ticker fetched live | Real retail portfolios are mostly ETFs and include non-S&P names; a tool that rejects them is unusable |
| Curated ETF list | Kept, as the set we can see *inside* and the set the diversification page may recommend | Arbitrary ETFs have no retrievable holdings (see probe) |
| Currency | USD-listed only; others rejected with a reason | Mixing currencies without FX conversion produces wrong risk numbers |
| Holdings cap | 2–50 (was 2–12) | Imported portfolios are larger than 12 |
| Sector taxonomy | Yahoo's, everywhere; the 11 sector SPDRs mapped onto it | Mixing GICS (Wikipedia) and Yahoo names splits one sector into two |
| Snapshot hosting | GitHub Release asset (option B), not committed to git, not external storage | Daily freshness, no git-history bloat, no new account or secret |

### Yahoo probe (run 2026-10-03 from the dev machine)

| Request | Result |
|---|---|
| `v1/finance/search?q=PLTR` | `EQUITY`, sector Technology, industry Software—Infrastructure — no crumb needed |
| `search?q=VOO`, `SCHD` | `ETF`, no sector, no holdings |
| `v10/finance/quoteSummary/VOO?modules=topHoldings` | **401 Unauthorized** without a crumb |
| `search?q=XYZQQ` | empty `quotes` → clean "not found" |
| `v8/finance/chart/RELIANCE.NS` | `meta.currency = INR` → non-USD detectable |

### External facts the design depends on

- Yahoo rate-limits / blocks plain HTTP clients from cloud IPs; `yfinance`
  handles the bot checks (yfinance issue #2297). GitHub Actions, Streamlit Cloud
  and Render are all cloud IPs. **Unverified for this app — Step 0 tests it.**
- GitHub disables scheduled workflows in a public repo after 60 days with no
  commit, silently. Option B never commits, so a keepalive is required.

## Changes made during planning (2026-10-03)

| Spec said | Plan does | Why |
|---|---|---|
| `prices.parquet` | `prices.csv.gz` | Parquet needs pyarrow (~40 MB) in both runtimes; pandas reads gzip CSV with no new dependency. Measured: gzip shrinks the current price CSV to 37.8%, so ~530 tickers ≈ 8 MB. |
| Separate monthly keepalive workflow | The refresh workflow re-enables itself as its last step | The 60-day rule disables *every* scheduled workflow in the repo, so a separate keepalive would be disabled too. |
| Sector via Yahoo for every ticker | S&P 500 sectors via a fixed GICS → Yahoo name map; Yahoo lookup only for live tickers | Verified 2026-10-03: the Wikipedia table has 503 rows and exactly the 11 GICS sectors; the map is deterministic and costs 0 requests. |
| Response adds `data_as_of`, `window_start`, `crisis_coverage` | Grouped under one `data` object: `as_of`, `window_start`, `window_days`, `window_status`, `crisis_coverage` | One place for provenance; keeps the top-level response shape stable. |
| (silent) | UI live cap 20 / 40 s | Imported portfolios can hold several non-S&P names; results are cached for 6 h per ticker. |
| Step 0 tests Render | Render is checked in the post-deploy matrix (Task 13) | No zero-cost way to run a probe on Render's free tier; the live path degrades to a clean 422 if blocked. |
| Editable table adds Sector and Source columns | Not added | The table renders before tickers resolve, so those columns would show the previous run's data; the reallocation table shows each holding's type and sector; whether a ticker came from the snapshot or a live lookup is not shown. |

## Data flow

```
STEP 0 — FEASIBILITY SPIKE (gates everything)
   From GitHub Actions, Streamlit Cloud and Render: can each reach Yahoo
   chart + search? Test urllib and yfinance on each. Record results here.

DAILY, Mon–Fri ~23:00 UTC  (.github/workflows/refresh-data.yml)
  tools/build_snapshot.py
   1. S&P 500 constituents ← Wikipedia (BRK.B → BRK-B)
        sanity: 490–510 names, else reuse the previous universe
   2. + curated ETFs, SPY, bond ETFs  (list lives in risk_engine/config.py)
   3. INCREMENTAL fetch: last 10 trading days per ticker
        overlap days disagree with stored adjclose by > 0.01% (relative)
        → history was re-scaled by a dividend/split → re-pull that ticker in full
   4. QUALITY GATE per ticker
        bad tick: |daily return| > 40% AND the next day reverses at least
                  half of it in the opposite direction (a real crash doesn't
                  bounce back overnight; a bad print does). Thresholds are
                  starting values, re-set from the first runs' reports.
        stale:    no new price in 5 trading days
        → quarantined: excluded, listed in the report, never auto-"fixed"
   5. failed + quarantined > 2% of tickers → job fails, publishes nothing
   6. publish release data-YYYY-MM-DD, move data-latest to it, prune > 14 days
SUNDAY: full 8-year rebuild + Fama-French factor refresh
MONTHLY: keepalive job (re-enables the schedule via the API)

Release assets
   prices.parquet        wide, one column per ticker, gaps left as NaN
   universe.json         ticker → name, sector, type, in_sp500, curated
   refresh_report.json   as_of, fetched, failed, quarantined (with reasons)

APP + API at boot
   fetch refresh_report.json → as_of changed? download prices.parquet : reuse cache
   download fails → bundled fallback in data/fallback/, labeled with its date
   freshness line: "Prices as of <date> close · N/M tickers · K quarantined: …"
   stale alarm: > 3 trading days old → banner; /health → stale: true

PER PORTFOLIO
   known ticker   → snapshot
   unknown ticker → live fetch: format-validated, USD-only, timeout, capped
   window = dates where ALL of the user's holdings have prices
   crisis a holding wasn't listed for → labeled "not listed yet", never dropped
```

**Fallback if Step 0 shows live fetch is blocked on Streamlit Cloud or Render:**
unknown tickers return "not available yet"; the request is logged so the ticker
joins the next nightly build. Recorded here only as the contingency — Step 0 decides.

Factor data lags about a month (Ken French publishes monthly); the factor
regression runs on the overlap and the UI labels the factor data's end date.

Size: ~525 tickers × ~2,200 days × 8 bytes ≈ 9 MB in memory. Parquet size and
app boot time are measured during the build, not assumed.

## Components

| File | Change |
|---|---|
| `tools/build_snapshot.py` | NEW. Builds the three release assets. Uses `yfinance` (tool dependency only, `requirements-refresh.txt`), urllib as fallback. Fetcher is injectable so tests run without network. |
| `.github/workflows/refresh-data.yml` | NEW. Daily incremental, Sunday full rebuild + factors, publish to releases. |
| `.github/workflows/keepalive.yml` | NEW. Monthly. |
| `risk_engine/data.py` | REWRITTEN. `load_snapshot()` (release download → cache → bundled fallback, returns as_of + report); `fetch_live(sym)` returning a typed result: `ok` / `not_found` / `non_usd` / `invalid` / `unavailable`. No bare `except: pass` — every failure carries a reason. |
| `risk_engine/portfolio.py` | NEW. `resolve_holdings(raw)` → snapshot / live / rejected (each with reason; duplicates merged and reported). `portfolio_window(prices, tickers)` → aligned returns, window start, per-ticker first date, crisis coverage. Window rules: ≥ 2 yrs normal; 1–2 yrs runs with a warning; < 1 yr refuses and names the young ticker. |
| `risk_engine/importer.py` | NEW. Pure parsers. Paste: one holding per line, case-insensitive, `$` / commas / tabs tolerated. CSV: header match for ticker (`Symbol`, `Ticker`) and value (`Market Value`, `Current Value`, `Value`, `Amount`); quantity-only files valued at latest close and labeled. Unmatched headers → error naming the expected headers. Broker formats are unverified; the parser does not claim support for any named broker. |
| `risk_engine/config.py` | Curated ETF list, crisis windows (adds the April 2025 tariff crash — exact peak/trough dates read from SPY in the data, not from memory), thresholds. The ticker universe now comes from `universe.json`. |
| `risk_engine/analyze.py` | Uses `portfolio.py`; response adds `data_as_of`, `window_start`, `crisis_coverage`; docstring rewritten to describe the bounded live path. |
| `api/schemas.py` | Ticker format regex `^[A-Z0-9.\-^=]{1,10}$` (after upper-casing) replaces list membership; 2–50 holdings. |
| `api/main.py` | Live fetches capped at 5 per request, ~10 s total; any unresolvable ticker → 422 with a reason per ticker (the API never returns a partial portfolio). `/health` adds `data_as_of`, `stale`. |
| `app.py` | Delete the duplicated universe (`app.py:274–294`). Sidebar: paste / CSV / search feeding the existing editable table (adds Sector and Source columns). "Not included" box. Preloaded example portfolio, labeled. Shareable URL via `st.query_params`, with a note that the link carries holdings. Hero "refreshed hourly" replaced with the real freshness line. Stress section uses `portfolio_window` crisis coverage. |

### Existing bugs fixed in passing

- `app.py:357` — all-zero amounts silently become equal weights. Replaced with an explicit "enter at least one amount" message.
- Hero copy claims "refreshed hourly" — becomes false under this design; replaced.
- `stress()` drops a whole crisis when any holding is missing and the caller filters `None` out silently — replaced by per-holding coverage labels.

## Security

User-typed tickers now reach a URL. The 2026-08-23 review cleared SSRF only
because tickers were hardcoded; that clearance no longer holds. Controls: strict
format regex before any request, fixed Yahoo hosts, per-request fetch cap,
timeouts. The stored-XSS note on the saved-portfolio `name` field still stands.

## Error handling

| Failure | User sees | Never |
|---|---|---|
| Nightly refresh fails | Yesterday's data with its date; Action failure email to owner | Half-empty data published |
| Data > 3 trading days old | Banner; `/health` `stale: true` | Old data shown as fresh |
| Release download fails | Bundled fallback, "data as of <date>" | Crash or wrong date |
| Yahoo down / blocked for a live ticker | That ticker in "Not included: data source unavailable" | Whole app failing over one ticker |
| Ticker quarantined | Named in freshness line and "Not included" | Poisoned numbers |
| Window < 1 year | Refusal naming the young ticker | VaR on 4 months of data |
| All amounts 0 | "Enter at least one amount" | Silent equal weights |
| Bad API input | 422, reason per ticker | 500, partial result, or silent drop |

## Testing

| Layer | Method |
|---|---|
| Golden master | Current `prices.csv` frozen under `tests/fixtures/`; existing metrics must match to 1e-9 |
| Refresh job | Fake fetcher, no network. Cases: bad tick → quarantined; stale → quarantined; adjclose mismatch → full re-pull; > 2% failed → no publish; universe of 400 names → previous list kept. Each test must fail if its rule is removed. |
| Importer | Table-driven messy inputs: `$`, commas, tabs, lowercase, duplicates, junk lines, quantity-only CSV, unknown headers |
| Window | Synthetic panel with a 2024 listing: COVID shows "not listed yet" for that holding and the window does not collapse for the others |
| API | `TestClient` with raw JSON `content=` for: valid, unknown-real, fake, non-USD, 51 holdings, malformed ticker, inf/NaN |
| App | Streamlit `AppTest` smoke run on the new data layer |
| CI | All network stubbed; no live Yahoo calls in CI |

### Post-deploy verification (live, not local)

1. Watch the CI run go green.
2. Streamlit: freshness line real; example loads; a paste with one bad ticker shows "Not included"; a 2024 listing shows "not listed yet" under COVID.
3. Render API matrix: known → 200, unknown-real → 200, fake → 422, INR → 422, malformed → 422, 51 holdings → 422; `/health` shows `data_as_of`.
4. First real Action run: read the report, record the real failure rate here (turns the 2% threshold from a guess into a measurement).
5. Measure and record app boot time with the full universe.

## Out of scope (belongs to later sub-projects)

ETF look-through, effective holdings, single-stock check (2) · volatility/regime
model, clustering (3) · diversification page (4) · panic replay (5) · LLM layer (6).
| Bad-tick rule scans each ticker's full history | The bad-print rule judges only the latest 20 rows (`QUALITY_WINDOW_ROWS`); the stale rule is unchanged | A full-history rescan turned real historical moves into permanent quarantines (HOOD, 2021-08-04: +50.4% then −27.6%). Accepted cost: a bad print Yahoo never corrects publishes once it is older than 20 rows. |

## Step 0 results (2026-10-03)

| Probe | Result |
|---|---|
| GitHub Actions, urllib chart + search (AAPL, MSFT, VOO, PLTR, BRK-B) | HTTP 200 on all 10 requests |
| GitHub Actions, yfinance 1.7.0 batch download of all 503 S&P 500 constituents since 2018 | 0 of 503 empty · 30.8 s · 2,200 rows |
| Size of that price table as gzip CSV (6 decimals) | 5,200,949 bytes (≈5.0 MiB) |
| Live Streamlit app (cold start), hero label | "Yahoo Finance · live" → urllib reaches Yahoo from Streamlit Cloud |

**Decisions:** the refresh job uses yfinance (pinned 1.7.0), with urllib as its per-ticker retry; the app's live lookup keeps urllib (no new runtime dependency). Render is still unprobed and is checked in the post-deploy matrix.

**Caveat:** one run on one day. It shows the path works today, not that Yahoo will never throttle cloud hosts; the stale-data banner and bundled fallback are the guard for that.
