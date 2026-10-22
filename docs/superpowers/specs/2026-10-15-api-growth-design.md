# Project A — Grow the API (design)

**Date:** 2026-10-15
**Status:** draft, awaiting Tam's review
**Program:** UI redesign. This is the first of three projects:
- **A:** grow the API (this document)
- **B:** React frontend
- **C:** hosting and switchover

Each project ships live before the next one starts. The old sub-project 2 (ETF look-through) comes after C.

## 1. Goal

The React frontend (project B) must be able to draw every section of today's dashboard from **one API call**. On top of that, the API returns the new **risk score**.

Today most of the numbers are computed inline in `app.py`, the Streamlit page. Project A does three things:
- Moves that math into `risk_engine` as pure, tested functions.
- Exposes the math through a new `POST /report` endpoint.
- Points the Streamlit app at the same functions, so the two frontends can never disagree.

### Success criteria

1. `POST /report` returns everything the 13 dashboard sections need, plus the risk score for the portfolio and for the S&P 500.
2. No dashboard number is computed in `app.py` any more. `app.py` only lays the numbers out.
3. Same request in, byte-identical response out. Every random draw is seeded.
4. The React dev server (`http://localhost:5173`) can call the API from the browser. Other origins are blocked unless they are configured.
5. One client cannot hog the free Render instance:
   - Per-IP rate limits apply.
   - At most 2 reports are computed at once.
6. Measured on Render, warm, with 10- and 50-ticker portfolios:
   - **p50 under 8 s**, or we switch to the two-call design (section 9).
   - Peak memory and gzipped payload size are recorded in this spec.
7. Every existing test still passes. Where a test asserted a buggy number fixed here (section 7), the test changes with a comment saying why.

### Not in scope

- The React app (B).
- Cold start, hosting and retiring Streamlit (C).
- ETF look-through.
- User accounts or authentication.
- Switching "expected return" from log to arithmetic mean. This is a model change and is listed under section 10.

## 2. The risk score

**Definition.** The score comes from the Monte Carlo run described in section 4:
- V = portfolio value today.
- P5 = the 5th percentile of simulated value after 252 trading days (1 year).
- P95 = the 95th percentile of simulated value after 252 trading days.

```
bad_year_loss  L = max(0, 1 - P5 / V)          # fraction lost in a 1-in-20 bad year
score            = clip(round(1 + 98 * L / 0.60), 1, 99)
loss_dollars     = V - P5    (shown as "could lose $X", floored at 0)
gain_dollars     = P95 - V   (shown as "could gain $Y")
```

**What it means:**
- Each point is about 0.6% of value lost in a bad year.
- A portfolio that loses nothing in a bad year scores 1.
- A portfolio that loses 60% or more scores 99.

It is linear on purpose. Tam has to be able to explain it in one sentence: "score 40 means a 1-in-20 bad year costs you about 24%."

**The S&P 500 score:**
- SPY gets the same simulation over the **same window** with the **same bootstrap draws** (the same random day indices).
- The two scores therefore differ only because the holdings differ, not because of sampling noise.

**The score can't disagree with the dollar range.** The score, `loss_dollars` and `gain_dollars` are all read off the same `terminal` array. One function returns all three, so no caller can mix runs.

**Calibration check before ship:**
- Record SPY's score in this spec.
- If SPY lands outside 15–50, the 0.60 constant is wrong for this data. Stop and raise it with Tam rather than quietly tuning it.
- **Measured 2026-10-23:** SPY scores **28** on the fixture snapshot (as of 2026-06-18), in the report for AAPL $30,000 / JPM $20,000 / TLT $40,000 / IEF $10,000 (window 2018-01-02 to 2026-06-18, seed 42). That falls inside 15–50, so 0.60 stays. A 1-in-20 bad year in SPY costs about 17%.

**Property tests:**
- The score rises when L rises (monotonic).
- An all-SPY portfolio scores the same as the benchmark score.
- A single high-volatility stock (TSLA) scores above SPY.
- Intermediate Treasuries (IEF) score below SPY. Long-duration Treasuries do not: the score includes drift, and on the 2018–2026 data TLT scores about 40 against SPY's 30, while a TLT/IEF 50/50 mix ties SPY.
- The mixed 30/20/40/10 portfolio (AAPL, JPM, TLT, IEF) still scores below SPY.
- The score is bounded to 1..99.

## 3. Architecture

```
api/main.py ──► risk_engine/report.py ──► risk_engine/{inputs, simulate, stress,
                     build_report()             benchmark, metrics, optimize,
app.py ──────► (same engine functions)          backtest, factors, portfolio}
```

**New modules** (all pure: no Streamlit, no FastAPI, no I/O except the snapshot passed in):

| Module | One job | Main function(s) |
|---|---|---|
| `risk_engine/inputs.py` | Turn holdings + snapshot into the shared analysis inputs | `prepare(holdings, snap, *, live_fetch, max_live, budget_s) -> Inputs` |
| `risk_engine/simulate.py` | Bootstrap Monte Carlo + risk score | `simulate(pr, value, *, horizons, n_sims, seed, draws=None) -> Simulation`, `risk_score(sim) -> Score` |
| `risk_engine/stress.py` | Historical crisis replay | `replay(prices, tickers, weights, crises) -> list[CrisisResult]` |
| `risk_engine/benchmark.py` | Portfolio vs S&P 500 | `compare(pr, spy_r, value) -> Comparison` (return, vol, Sharpe, max DD, beta, growth paths) |
| `risk_engine/report.py` | Orchestrate every section into one JSON-safe dict | `build_report(holdings, *, confidence, shrinkage, snapshot, live_fetch) -> dict` |

**Additions to existing modules:**

`metrics.py`:
- `return_histogram(pr, bins=80)`: density, `np.histogram`, `density=True`.
- `correlation(ret_sel)`
- `to_dollars(log_r, value)`: uses `expm1`. See section 7.

`backtest.py`:
- `backtest_var` also returns the rolling historical VaR series, so the chart no longer recomputes it.

`optimize.py`:
- `_solve` checks `res.success` and raises `OptimizerError` on non-convergence.

`analyze.py`:
- Rebuilt on `inputs.prepare`.
- Its public behaviour and the `/analyze` response are **unchanged**. The API showcase documents them.

**`Inputs`** (frozen dataclass) holds the outputs of the shared pipeline that sits at the top of `app.py` today:
- `res`, `tickers`, `weights`, `value`
- `window`, `lr`, `pr`, `spy_r`, `prices`, `meta`

Built once per request. Every section reads from it.

## 4. Monte Carlo

- **Method:** i.i.d. bootstrap of daily portfolio log returns, as today. 10,000 paths.
- **One run per request, not one per horizon.** It draws a 756-day (3-year) path matrix once. The 6-month, 1-year, 2-year and 3-year results are all read off that one matrix.
  - The frontend's horizon slider slices the result client-side, with no new call.
  - The 1-year numbers the slider shows are the same numbers the score uses.
- **Seed:** 42. Draw indices are generated once and shared by the portfolio and SPY runs (section 2).
- **Memory.** About 250 MB is estimated today; the 512 MB free instance can't afford that twice.
  - Draw indices are `int32`.
  - Take, cumsum and exp run in place into one preallocated `float64` (N × 756) buffer.
  - Target peak is under 150 MB per call, checked with a `tracemalloc` test.
- **Returned per horizon (126 / 252 / 504 / 756 days):**
  - Terminal P5, P50 and P95
  - P(loss), P(up 20%)
- **Fan chart:** percentile bands (5/25/50/75/95) every 6 trading days. That is 127 points × 5 bands, not 756 × 5. Every horizon is a multiple of 6, so each one lands on a band point.

## 5. `POST /report`

**Request:**

```json
{ "holdings": {"AAPL": 30000, "TLT": 20000},
  "confidence": 0.95,
  "shrinkage": false }
```

- Holdings validation is the same as `AnalyzeRequest`: 2..50 tickers, finite amounts > 0, each at most 1e12, duplicates merged.
- `confidence` is between 0.90 and 0.99.
- `shrinkage` turns on Ledoit-Wolf for the optimizer, replacing the sidebar checkbox.

**Response** (pydantic models in `api/schemas.py`; every series is `{dates: [...], values: [...]}` with ISO dates):

| Key | Contents | Feeds section |
|---|---|---|
| `score` | `portfolio {score, loss_dollars, gain_dollars, bad_year_loss}` (`bad_year_loss` is a fraction), `benchmark {same}`, `horizon_days: 252` | headline |
| `portfolio` | value, holdings `[{ticker, name, sector, type, asset_class, amount, weight}]` | header, profile |
| `risk` | hist/gaussian VaR and CVaR as fractions **and** dollars, confidence, the hist–gauss gap in dollars | 1 Daily downside |
| `profile` | annual vol, Sharpe, max drawdown, mean-log annual return | 2 Profile |
| `benchmark` | the table rows above for portfolio vs SPY, beta, growth series for both | 3 Benchmark |
| `distribution` | histogram edges and densities, mu, std (the frontend draws the normal curve) | 4 Distribution |
| `drawdown` | series | 5 Drawdown |
| `stress` | per crisis: label, context, start, end, status (`ok` / `not_covered` / `too_few_days`), missing tickers, total, worst_day, max_dd, `change_dollars` (signed; negative is a loss), path series | 6 Stress |
| `correlation` | tickers, matrix (rounded to 3 dp; a cell is null for a flat-priced holding) | 7 Correlation |
| `backtest` | per method: observations, breaches, expected, kupiec_p, christoffersen_p, passed; plus series `{dates, var, realized, breach}` | 8 Model validation |
| `risk_contribution` | per ticker: weight, risk share; the biggest over-contributor; the best diversifier (or null) | 9 Risk contribution |
| `factors` | `available: bool`, `reason` if not, betas, alpha, r², variance split | 10 Factors |
| `optimizer` | `available`, `reason`; current / max-Sharpe / min-var `{ret, vol, sharpe, weights}`; frontier points; reallocation rows `[{ticker, asset_class, sector, current, suggested, delta_dollars}]`; sector mix (sectors, equity, diversified, unplaced, non_equity); concentration flag | 11 Optimizer |
| `monte_carlo` | per-horizon terminals and probabilities; bands every 6 days; n_sims, seed | 12 Monte Carlo |
| `data` | as_of, window start and days, window status and message, `assumptions: [str]` | 13 Assumptions, footer |

**Partial results, not 422s.** A section that can't be computed returns `available: false` plus a human reason instead of killing the whole report:
- **Factors:** fewer than 60 overlapping days.
- **Optimizer:** non-convergence.
- **Stress:** a crisis that isn't covered, or has fewer than 5 clean days. These are listed with their status; today they vanish silently.

**Still a 422:** any rejected ticker, or a window shorter than 252 days. The report is meaningless without them. The body is per-ticker problems, same shape as `/analyze`.

`/analyze` has no partial sections, so there a non-converging optimizer is a 422 with an `(optimizer)` problem.

**Size:** daily series of fractions (returns, drawdowns, VaR) keep 6 significant digits. Dollar series (benchmark growth, Monte Carlo bands) are rounded to the cent, because significant digits would round a $2M portfolio to the nearest $10. `GZipMiddleware` (minimum 1 KB) handles the rest.

## 6. Other endpoint changes

**`GET /universe`** — new.
- Returns the snapshot's tickers: `[{ticker, name, sector, asset_class, in_sp500}]`, about 530 rows.
- The frontend fetches it once and searches client-side, so there's no search endpoint to rate-limit.
- Tickers outside the universe still work through the existing live lookup.

**`POST /import`** — new.
- Request: `{format: "csv" | "paste", text}`, with `text` at most 200 KB.
- Response: `{holdings, notes}` from the existing `importer.parse_csv` / `parse_paste`, with last prices from the snapshot.
- It parses only and never analyses.
- It returns at most 50 notes, plus one "…and N more" summary note, so a 200 KB paste of junk can't come back as 100,000 notes.

**`/portfolios` — security fix.** This is live today and **has a privacy hole**:
- `GET /portfolios` lists every saved portfolio from every user.
- IDs are sequential integers, so `GET /portfolios/{id}` can be enumerated.

Fixes:
- Remove `GET /portfolios` (the list).
- New IDs become random 22-character `secrets.token_urlsafe(16)` tokens.
- `GET /portfolios/{id}` stays.
- The API showcase text at `app.py:223` is updated to match.
- The Render database is ephemeral, so no migration is needed. Old integer rows are simply unreachable.

**CORS:**
- `CORSMiddleware` reads allowed origins from env `ALLOWED_ORIGINS` (comma-separated), defaulting to `http://localhost:5173`.
- Methods: GET and POST. No credentials.
- It exposes `Retry-After`, so the browser can read it on a 429 or 503.
- Project C adds the production origin.

**Rate limits.** In-memory, single instance, about 40 lines in `api/limits.py`. No new dependency.

Per-IP fixed window of 60 seconds:

| Endpoint | Limit |
|---|---|
| `/report` | 10 per minute |
| `/analyze` | 10 per minute |
| `/import` | 20 per minute |
| everything else | 60 per minute |

- Over the limit returns 429 with `Retry-After`.
- `GET /portfolios/{id}?analyze=true` runs a full analysis, so it goes through the compute gate and counts against the `/analyze` limit, not the general one.
- **Client IP** is the **last** `X-Forwarded-For` entry, the one Render's proxy adds. The first entry is client-controlled and spoofable. This is verified on Render in the measurement step. If Render's header differs, the rule changes before ship.
- **Compute gate:** `/report` and `/analyze` share a `threading.BoundedSemaphore(2)`. A request that can't get a slot within 15 s returns 503 `{"detail": "busy"}` with `Retry-After: 10`. This protects memory, not just CPU.
  - At most 8 requests wait for a slot (`max_waiting=8`). Beyond that, the request gets an immediate 503 busy instead of tying up another worker thread for 15 s.
- **422 bodies:** a 422 echoes the offending input back, at most 200 characters per string, with lone surrogates replaced, so a huge or unencodable value can't turn the 422 into a 500 or an outsized response.

## 7. Bugs fixed along the way

These were found in the extraction survey. Each fix gets a test that fails on the old code.

| # | Where | Bug | Fix |
|---|---|---|---|
| 1 | VaR dollars, app.py:502-542 | Dollars = log return × value. This overstates loss: a -10% log return is a -9.5% real loss. | `to_dollars` uses `expm1` everywhere |
| 2 | `optimize._solve` | Never checks `res.success`, so a non-converged solve returns garbage weights. The UI's "couldn't converge" message is dead code. | Raise `OptimizerError`; report returns `available: false` |
| 3 | app.py:1052-1057 | `except Exception: opt_ok=False` swallows every error | Catch `OptimizerError` only |
| 4 | Stress, app.py:729 | A crisis with fewer than 5 clean days is silently dropped | Returned with status `too_few_days` and shown |
| 5 | app.py:708, `CRISIS_COLOR` | Hard-coded per label; adding a crisis to config crashes the page | Colour by position in the list |
| 6 | Beta, app.py | `np.cov` (ddof=1) over `np.var` (ddof=0) | ddof=1 for both |
| 7 | Monte Carlo text, app.py:1249 | "1 year in 20" is hard-coded for every horizon | Text uses the chosen horizon |
| 8 | API showcase, app.py:238-261 | Hard-coded sample numbers "computed 2026-08-23" | Left as is. It was already labelled "computed 2026-08-23", so no change was needed (live numbers come with the React page) |
| 9 | `backtest` | The historical rolling VaR is computed twice | `backtest_var` returns the series |

Fixes 1 and 6 change numbers Streamlit shows today, by about 1% and a hair respectively. They are corrections, not regressions.

## 8. Testing

1. **Characterization first.** Before any inline block is deleted from `app.py`, a test pins its current output on the fixture snapshot. The engine function must reproduce it exactly, except for the fixes in section 7, which are asserted separately.
2. **Unit tests** for every new function, with hand-checkable numbers on tiny inputs. For example, stress on a 2-ticker, 6-day price table is worked out by hand in the test.
3. **Score properties** as listed in section 2.
4. **Determinism:** same `/report` request twice gives an identical JSON body.
5. **Memory:** `tracemalloc` peak for `simulate` at 756 days is under 150 MB.
6. **API tests:**
   - The `/report` schema round-trips.
   - Partial sections (factor gap, optimizer failure mocked, uncovered crisis).
   - 422 shapes.
   - CORS preflight from an allowed origin passes; a disallowed origin gets no `Access-Control-Allow-Origin` header.
   - 429 after the limit, with an injected clock.
   - The semaphore returns 503 when it's full.
   - `/universe`, `/import`.
   - The `/portfolios` list returns 404/405, and IDs are not sequential.
7. **Streamlit smoke tests** still pass, now exercising the engine path.
8. The full suite (370 today) is green locally **and** in CI before any push.

## 9. Ship and measure (the 8-second gate)

**Ship:**
- Built on a branch.
- Ship-check, then pushed **only on Tam's word**.
- Render redeploys from main.

**Measure on the live service.** Recorded here, with the date:

| Measure | Rule |
|---|---|
| warm `/report` p50 and p95, 10 tickers and 50 tickers | 10 calls each |
| peak RSS during a 50-ticker report | from Render metrics |
| gzipped response size, 10 and 50 tickers | |
| SPY's score | calibration check, section 2 |
| `X-Forwarded-For` layout on Render | IP rule, section 6 |

**Gate:** if the 50-ticker warm p50 is over 8 s, we split before project B starts:
- A fast `POST /report/summary`: score, range, headline numbers.
- The full `/report`.

That's a known, bounded change because every section is already its own function.

## 10. Known limitations (stated, not fixed here)

**Bootstrap drift.** The Monte Carlo resamples real 2018–2026 returns, a period that was mostly a bull market. So the "bad year" includes that period's upward drift and is milder than a drift-free model would say.
- The S&P 500 score has the same bias, so the *comparison* stays fair.
- The absolute dollar figure leans optimistic.
- The `assumptions` list says so in plain words.
- A drift-neutral variant is a candidate follow-up.

**Volatility clustering.** An i.i.d. bootstrap loses it. Bad days come in runs in reality, so short-horizon tails are understated.

**"Expected return" uses the mean log return × 252.** That understates the arithmetic return by about σ²/2. This affects the optimizer and frontier, and is unchanged here.

**Two different weight conventions:**
- The growth chart assumes daily rebalancing.
- The stress test assumes buy-and-hold.

Both are stated in `assumptions`.

**Rate limits reset on restart.** They are per-process, which is fine for one free instance. They are not a defence against a determined attacker, and nothing here is.
