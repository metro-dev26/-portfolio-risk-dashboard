# Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-picked 24-ticker universe with the S&P 500 plus curated ETFs, refreshed daily through GitHub Releases, and let a user bring in their real portfolio (paste / CSV / search, any USD ticker), with every data failure shown instead of hidden.

**Architecture:** A scheduled GitHub Action builds a data bundle (`prices.csv.gz`, `universe.json`, `factors.csv`, `refresh_report.json`) and publishes it as the `data-latest` release. `risk_engine/data.py` loads that bundle (release → local cache → bundled fallback in `data/fallback/`), and fetches tickers outside it live from Yahoo. New pure modules, `risk_engine/importer.py` (text/CSV → holdings) and `risk_engine/portfolio.py` (holdings → resolved prices → honest date window), are shared by the Streamlit app and the API.

**Tech Stack:** Python 3.11, pandas 2.2, numpy 2.2, Streamlit ≥1.40 (1.54 locally), FastAPI 0.133 + pydantic 2.12, pytest; refresh job only: yfinance + lxml; GitHub Actions + `gh` CLI.

**Spec:** `docs/superpowers/specs/2026-10-03-data-foundation-design.md`

## Global Constraints

- `risk_engine/` stays pure: never imports streamlit, fastapi or pydantic.
- No new runtime dependency in `requirements.txt` or `requirements-api.txt` unless Task 1's results require yfinance in the app (Task 4, Step 7). The refresh job's dependencies live only in `requirements-refresh.txt`.
- Holdings: 2–50 tickers, USD-listed only, ticker format `^[A-Z0-9][A-Z0-9.\-^=]{0,9}$` after upper-casing and `BRK.B → BRK-B`.
- Refresh thresholds: publish blocked if failed + quarantined > 2% of tickers; bad print = |daily move| > 40% that the next close undoes by ≥ 50%; stale = no close in the last 5 rows; adjusted-close mismatch tolerance 0.01% (1e-4 relative); incremental overlap 10 business days; S&P table must have 490–510 rows.
- Window: ≥ 504 return days = ok; 252–503 = runs with a warning; < 252 = refused, naming the ticker that limits the window.
- Stale-data banner when the snapshot is more than 3 business days old.
- Live lookups: API max 5 per request with a 10 s budget; UI max 20 with a 40 s budget.
- No bare `except: pass`. Every failure carries a human-readable reason that reaches the user, the report or the API response.
- User-supplied text (tickers, notes) is `html.escape`d before any `unsafe_allow_html` render.
- `tests/fixtures/golden.json` is never regenerated in this plan. If a golden test fails, the code is wrong.
- Commits: imperative subject in the existing style ("Add …", "Return …"). **No `Co-Authored-By` trailer** (repo owner's rule).
- Code style: module docstrings and short "why" comments as in the existing modules; no comments narrating construction history ("now", "new", "changed from").
- README voice: impersonal, no "I"/"you".

## Spec deviations (decided while planning, recorded in the spec in Task 2)

| Spec said | Plan does | Why |
|---|---|---|
| `prices.parquet` | `prices.csv.gz` | Parquet needs pyarrow (~40 MB) in both runtimes; pandas reads gzip CSV with no new dependency. Measured: gzip shrinks the current price CSV to 37.8%, so ~530 tickers ≈ 8 MB. |
| Separate monthly keepalive workflow | The refresh workflow re-enables itself as its last step | The 60-day rule disables *every* scheduled workflow in the repo, so a separate keepalive would be disabled too. |
| Sector via Yahoo for every ticker | S&P 500 sectors via a fixed GICS → Yahoo name map; Yahoo lookup only for live tickers | Verified 2026-10-03: the Wikipedia table has 503 rows and exactly the 11 GICS sectors; the map is deterministic and costs 0 requests. |
| Response adds `data_as_of`, `window_start`, `crisis_coverage` | Grouped under one `data` object: `as_of`, `window_start`, `window_days`, `window_status`, `crisis_coverage` | One place for provenance; keeps the top-level response shape stable. |
| (silent) | UI live cap 20 / 40 s | Imported portfolios can hold several non-S&P names; results are cached for 6 h per ticker. |
| Step 0 tests Render | Render is checked in the post-deploy matrix (Task 13) | No zero-cost way to run a probe on Render's free tier; the live path degrades to a clean 422 if blocked. |

## Review Focus

1. **Share counts pasted as dollars** ("AAPL 50" meaning 50 shares): the app can't know, so the sidebar must say amounts are dollars. Pinned in Task 10 (caption assertion).
2. **Class-share tickers typed with a dot** (`BRK.B`, `brk.b`): must resolve to `BRK-B`. Pinned in Task 4 (`normalize_ticker`) and Task 5 (paste).
3. **Broker CSV junk rows** ("Cash & Cash Investments", "Account Total", account info above the header): skipped with a note, never a crash. Pinned in Task 5.
4. **Negative / parenthesised amounts** (short positions, `(1,234.00)`): rejected with "short positions are not supported". Pinned in Task 5.
5. **HTML in pasted text** (`<img src=x onerror=alert(1)> 100`): must render escaped in the "Not included" box. Pinned in Task 10.
6. **Holding SPY itself** (the benchmark): no duplicate column in the window. Pinned in Task 6.

---

## File Structure

| Path | Responsibility |
|---|---|
| `risk_engine/config.py` | Constants only: curated ETFs, GICS map, crises, thresholds, release URL |
| `risk_engine/data.py` | Bundle loading (release/cache/fallback), staleness, freshness line, live Yahoo fetch, ticker normalisation |
| `risk_engine/importer.py` | Pure parsers: paste, CSV, editor rows, share-link encode/decode |
| `risk_engine/portfolio.py` | Resolve holdings → prices + metadata + rejections; date window; crisis coverage |
| `risk_engine/analyze.py` | API orchestration on top of `portfolio.py`; raises `PortfolioError` |
| `risk_engine/factors.py` | Unchanged math; `load_factors()` reads the snapshot |
| `api/schemas.py`, `api/main.py` | Format validation, 2–50, 422 with per-ticker problems, `/health` freshness |
| `tools/build_snapshot.py` | Daily/weekly bundle build (pure functions + CLI) |
| `tools/prune_releases.py` | Delete dated data releases older than 14 days |
| `.github/workflows/refresh-data.yml` | Schedule, build, publish, prune, self re-enable |
| `requirements-refresh.txt` | Refresh-job dependencies |
| `data/fallback/` | Bundled bundle shipped with the repo |
| `tests/fixtures/snapshot/` | Frozen 25-ticker bundle all tests run against |
| `app.py` | Sidebar import, "Not included", window, freshness, stress coverage, share link |

---

### Task 1: Feasibility spike — can Yahoo be reached from the cloud?

Throwaway. Nothing from this task is merged. It answers the question that gates the whole design.

**Files (on branch `spike-yahoo` only):**
- Create: `spike/yahoo_probe.py`
- Create: `.github/workflows/spike-yahoo.yml`

**Interfaces:** Produces only a recorded result in the spec (Step 7).

- [ ] **Step 1: Create the throwaway branch**

```bash
cd C:/Users/sujan/portfolio-risk-dashboard
git checkout -b spike-yahoo
```

- [ ] **Step 2: Write the probe**

`spike/yahoo_probe.py`:

```python
"""Throwaway probe: can this machine reach Yahoo (plain urllib and yfinance), and
how does yfinance do at S&P 500 scale? Prints one JSON report."""
import io
import json
import time
import urllib.request

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}
WIKI_UA = {"User-Agent": "MarketPlug data refresh (github.com/metro-dev26/-portfolio-risk-dashboard)"}
SAMPLE = ["AAPL", "MSFT", "VOO", "PLTR", "BRK-B"]


def get(url, headers=UA):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, r.read()


out = {"urllib_chart": {}, "urllib_search": {}}
for s in SAMPLE:
    urls = {
        "urllib_chart": f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=1mo&interval=1d",
        "urllib_search": f"https://query1.finance.yahoo.com/v1/finance/search?q={s}&quotesCount=1&newsCount=0",
    }
    for kind, url in urls.items():
        try:
            code, body = get(url)
            out[kind][s] = f"{code} ({len(body)} bytes)"
        except Exception as e:
            out[kind][s] = f"{type(e).__name__}: {e}"

_, html = get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", WIKI_UA)
table = pd.read_html(io.StringIO(html.decode()), attrs={"id": "constituents"})[0]
tickers = [t.replace(".", "-") for t in table["Symbol"]]

import yfinance as yf

t0 = time.time()
df = yf.download(tickers, start="2018-01-01", auto_adjust=True, progress=False, threads=True)
closes = df["Close"]
empty = [t for t in tickers if t not in closes.columns or closes[t].dropna().empty]
buf = io.BytesIO()
closes.to_csv(buf, compression={"method": "gzip"}, float_format="%.6f")
out["yfinance"] = {
    "requested": len(tickers), "empty": len(empty), "empty_list": empty[:50],
    "seconds": round(time.time() - t0, 1), "rows": len(closes),
    "csv_gz_bytes": buf.tell(),
}
print(json.dumps(out, indent=2))
```

- [ ] **Step 3: Write the spike workflow**

`.github/workflows/spike-yahoo.yml`:

```yaml
name: spike-yahoo
on:
  push:
    branches: [spike-yahoo]
jobs:
  probe:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - run: pip install pandas==2.2.3 yfinance lxml
      - run: pip show yfinance | head -2
      - run: python spike/yahoo_probe.py
```

- [ ] **Step 4: Push the spike branch (ask the repo owner first; this creates a remote branch)**

```bash
git add spike/yahoo_probe.py .github/workflows/spike-yahoo.yml
git commit -m "Spike: probe Yahoo reachability from GitHub Actions"
git push -u origin spike-yahoo
```

- [ ] **Step 5: Read the run**

```bash
gh run list --branch spike-yahoo --limit 1
gh run watch <run-id> --exit-status
gh run view <run-id> --log | sed -n '/"urllib_chart"/,/^}/p'
```

Record: urllib chart/search status per ticker, yfinance `empty` count and list, `seconds`, `csv_gz_bytes`, and the yfinance version printed by `pip show`.

- [ ] **Step 6: Check Streamlit Cloud (urllib already runs there today)**

The live app calls `load_prices(prefer_live=True)` via urllib. Open `https://icn93ppxyjtbgd5sxdjvpb.streamlit.app/` (Playwright MCP: navigate, wait for "Portfolio Risk Dashboard", read the hero eyebrow text). `Yahoo Finance · live` means urllib works from Streamlit Cloud. `frozen snapshot · <date>` means every live fetch failed there.

- [ ] **Step 7: Record results and decide**

Append a `## Step 0 results (YYYY-MM-DD)` section to the spec with the numbers from Steps 5–6, then apply:

| GitHub Actions result | Decision |
|---|---|
| yfinance `empty` ≤ 2% | Proceed. The refresh job uses yfinance. |
| yfinance `empty` > 2% but urllib works | Proceed. `fetch_yf` falls back per ticker to urllib (already in Task 8). Note the measured rate. |
| Both blocked | **STOP.** Return to design with the evidence; do not start Task 2. |

| Streamlit Cloud result | Decision |
|---|---|
| `Yahoo Finance · live` | `fetch_live` keeps urllib. Skip Task 4, Step 7. |
| `frozen snapshot` | Do Task 4, Step 7 (yfinance transport in the app). |

- [ ] **Step 8: Delete the spike branch**

```bash
git checkout main
git push origin --delete spike-yahoo
git branch -D spike-yahoo
git add docs/superpowers/specs/2026-10-03-data-foundation-design.md
git commit -m "Record Yahoo reachability results in the data foundation spec"
```

---

### Task 2: Config constants + frozen test bundle

**Files:**
- Modify: `risk_engine/config.py`
- Create: `tests/fixtures/snapshot/{prices.csv.gz,universe.json,factors.csv,refresh_report.json}`
- Create: `tools/make_test_bundle.py`
- Modify: `tests/test_config.py`
- Modify: `docs/superpowers/specs/2026-10-03-data-foundation-design.md` (deviations table)

**Interfaces:**
- Produces in `risk_engine.config`: `TRADING_DAYS`, `DATA_RELEASE_URL: str`, `BENCHMARK = "SPY"`, `GICS_TO_YAHOO: dict[str, str]`, `CURATED_ETFS: dict[str, tuple[str, str, str]]` (name, sector, asset_class), `CRISES: list[tuple[str, str, str, str]]` (label, start, end, context), `MIN_HOLDINGS = 2`, `MAX_HOLDINGS = 50`, `WINDOW_OK_DAYS = 504`, `WINDOW_MIN_DAYS = 252`, `STALE_BUSINESS_DAYS = 3`, `LIVE_MAX_API = 5`, `LIVE_BUDGET_API_S = 10.0`, `LIVE_MAX_UI = 20`, `LIVE_BUDGET_UI_S = 40.0`.
- Keeps (until Task 12): `TICKERS`, `SECTOR`, `BONDS`, `ASSET_CLASS`, still used by `app.py`, `analyze.py`, `api/schemas.py` and `risk_engine/data.py`.
- Produces `tests/fixtures/snapshot/`: the bundle format every later task reads.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_curated_etfs_are_well_formed():
    assert config.BENCHMARK in config.CURATED_ETFS
    for t, (name, sector, asset_class) in config.CURATED_ETFS.items():
        assert t.isupper() and name and sector
        assert asset_class in ("Equity", "Bond", "Commodity")


def test_gics_map_covers_all_eleven_sectors():
    assert len(config.GICS_TO_YAHOO) == 11
    assert config.GICS_TO_YAHOO["Information Technology"] == "Technology"


def test_crises_are_ordered_windows_and_include_tariff_shock():
    labels = [c[0] for c in config.CRISES]
    assert "2025 Tariff Shock" in labels
    for _, start, end, _ in config.CRISES:
        assert start < end


def test_holding_and_window_limits():
    assert (config.MIN_HOLDINGS, config.MAX_HOLDINGS) == (2, 50)
    assert config.WINDOW_MIN_DAYS < config.WINDOW_OK_DAYS
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: module 'risk_engine.config' has no attribute 'CURATED_ETFS'`.

- [ ] **Step 3: Add the constants**

Replace `risk_engine/config.py` with:

```python
"""Shared constants for the risk engine: curated ETFs, crisis windows, limits and
data thresholds. The stock universe itself is data (universe.json), not code."""

TRADING_DAYS = 252

DATA_RELEASE_URL = ("https://github.com/metro-dev26/-portfolio-risk-dashboard/"
                    "releases/download/data-latest")
BENCHMARK = "SPY"

# Wikipedia's S&P 500 table uses GICS sector names; Yahoo uses its own. Mapping
# GICS onto Yahoo's names keeps listed stocks and live-looked-up tickers in one
# taxonomy, so a sector is never split in two.
GICS_TO_YAHOO = {
    "Information Technology": "Technology",
    "Health Care": "Healthcare",
    "Financials": "Financial Services",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Materials": "Basic Materials",
}

# ETFs whose contents are known: the ones look-through can open and the
# diversification page can suggest. ticker -> (name, sector, asset class)
CURATED_ETFS = {
    "SPY": ("SPDR S&P 500 ETF Trust", "Broad Market", "Equity"),
    "VOO": ("Vanguard S&P 500 ETF", "Broad Market", "Equity"),
    "IVV": ("iShares Core S&P 500 ETF", "Broad Market", "Equity"),
    "VTI": ("Vanguard Total Stock Market ETF", "Broad Market", "Equity"),
    "QQQ": ("Invesco QQQ Trust", "Broad Market", "Equity"),
    "SCHD": ("Schwab U.S. Dividend Equity ETF", "Broad Market", "Equity"),
    "XLK": ("Technology Select Sector SPDR Fund", "Technology", "Equity"),
    "XLV": ("Health Care Select Sector SPDR Fund", "Healthcare", "Equity"),
    "XLF": ("Financial Select Sector SPDR Fund", "Financial Services", "Equity"),
    "XLY": ("Consumer Discretionary Select Sector SPDR Fund", "Consumer Cyclical", "Equity"),
    "XLP": ("Consumer Staples Select Sector SPDR Fund", "Consumer Defensive", "Equity"),
    "XLC": ("Communication Services Select Sector SPDR Fund", "Communication Services", "Equity"),
    "XLI": ("Industrial Select Sector SPDR Fund", "Industrials", "Equity"),
    "XLE": ("Energy Select Sector SPDR Fund", "Energy", "Equity"),
    "XLU": ("Utilities Select Sector SPDR Fund", "Utilities", "Equity"),
    "XLRE": ("Real Estate Select Sector SPDR Fund", "Real Estate", "Equity"),
    "XLB": ("Materials Select Sector SPDR Fund", "Basic Materials", "Equity"),
    "BND": ("Vanguard Total Bond Market ETF", "Aggregate Bonds", "Bond"),
    "AGG": ("iShares Core U.S. Aggregate Bond ETF", "Aggregate Bonds", "Bond"),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "Govt Bonds", "Bond"),
    "IEF": ("iShares 7-10 Year Treasury Bond ETF", "Govt Bonds", "Bond"),
    "SHY": ("iShares 1-3 Year Treasury Bond ETF", "Govt Bonds", "Bond"),
    "LQD": ("iShares iBoxx $ Investment Grade Corporate Bond ETF", "Corp Bonds", "Bond"),
    "VXUS": ("Vanguard Total International Stock ETF", "International", "Equity"),
    "VEA": ("Vanguard FTSE Developed Markets ETF", "International", "Equity"),
    "VWO": ("Vanguard FTSE Emerging Markets ETF", "International", "Equity"),
    "GLD": ("SPDR Gold Shares", "Gold", "Commodity"),
}

# Real crash windows inside the data (2018 onward): label, start, end, context.
# Tariff-shock dates are SPY's close-to-close peak and trough (-18.8%).
CRISES = [
    ("2018 Q4 Selloff", "2018-10-01", "2018-12-24", "Fed tightening + trade-war fears"),
    ("COVID-19 Crash", "2020-02-19", "2020-03-23", "Fastest-ever 30%+ market drop"),
    ("2022 Bear Market", "2022-01-03", "2022-10-12", "Inflation shock + rate hikes"),
    ("2025 Tariff Shock", "2025-02-19", "2025-04-08", "Sweeping US tariffs announced"),
]

MIN_HOLDINGS, MAX_HOLDINGS = 2, 50
WINDOW_OK_DAYS = 504      # two years of shared history: results run normally
WINDOW_MIN_DAYS = 252     # under one year, VaR is too noisy to show
STALE_BUSINESS_DAYS = 3

LIVE_MAX_API, LIVE_BUDGET_API_S = 5, 10.0
LIVE_MAX_UI, LIVE_BUDGET_UI_S = 20, 40.0

# Legacy 24-ticker universe; removed once app.py, analyze.py and the API read universe.json.
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
```

- [ ] **Step 4: Run config tests**

Run: `python -m pytest tests/test_config.py -v`
Expected: all PASS (old and new).

- [ ] **Step 5: Write the bundle builder for the test fixture**

`tools/make_test_bundle.py`:

```python
"""One-off: freeze the June 2026 24-ticker data as the bundle every test runs
against, so swapping the live universe never moves a tested number.
Run once from the repo root: python tools/make_test_bundle.py"""
import json
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine.config import CURATED_ETFS

OUT = "tests/fixtures/snapshot"
STOCK_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Communication Services", "META": "Communication Services",
    "AMZN": "Consumer Cyclical", "TSLA": "Consumer Cyclical", "WMT": "Consumer Defensive",
    "JPM": "Financial Services", "GS": "Financial Services",
    "BAC": "Financial Services", "MS": "Financial Services",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "BA": "Industrials",
}

os.makedirs(OUT, exist_ok=True)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
prices.to_csv(f"{OUT}/prices.csv.gz", compression="gzip")
shutil.copyfile("factors.csv", f"{OUT}/factors.csv")

universe = {t: {"name": t, "sector": s, "type": "stock", "asset_class": "Equity",
                "in_sp500": True, "curated": False} for t, s in STOCK_SECTORS.items()}
for t in ("TLT", "IEF", "AGG", "LQD", "SPY"):
    name, sector, asset_class = CURATED_ETFS[t]
    universe[t] = {"name": name, "sector": sector, "type": "etf", "asset_class": asset_class,
                   "in_sp500": False, "curated": True}
with open(f"{OUT}/universe.json", "w") as f:
    json.dump(universe, f, indent=1, sort_keys=True)

factors_end = pd.read_csv("factors.csv", index_col=0, parse_dates=True).index.max()
report = {"as_of": str(prices.index.max().date()), "built_at": "2026-06-20T00:00:00Z",
          "mode": "full", "n_tickers": len(universe), "failed": {}, "quarantined": {},
          "repulled": [], "universe_fallback": False,
          "factors_through": str(factors_end.date())}
with open(f"{OUT}/refresh_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(f"wrote {OUT}: {prices.shape[1]} tickers, as of {report['as_of']}")
```

- [ ] **Step 6: Run it and check the round trip is exact**

```bash
python tools/make_test_bundle.py
python -c "import pandas as pd; a=pd.read_csv('prices.csv',index_col=0,parse_dates=True); b=pd.read_csv('tests/fixtures/snapshot/prices.csv.gz',index_col=0,parse_dates=True); print((a-b).abs().max().max())"
```

Expected: `wrote tests/fixtures/snapshot: 25 tickers, as of 2026-06-18`, then `0.0`.

- [ ] **Step 7: Record the spec deviations**

In `docs/superpowers/specs/2026-10-03-data-foundation-design.md`, add a section `## Changes made during planning (2026-10-03)` containing the "Spec deviations" table from the top of this plan, verbatim.

- [ ] **Step 8: Commit**

```bash
git add risk_engine/config.py tests/test_config.py tools/make_test_bundle.py tests/fixtures/snapshot docs/superpowers/specs/2026-10-03-data-foundation-design.md
git commit -m "Add universe constants, crisis windows and a frozen test data bundle"
```

---

### Task 3: Snapshot loader (release → cache → fallback)

**Files:**
- Modify: `risk_engine/data.py` (loading half; live fetch comes in Task 4)
- Modify: `risk_engine/factors.py:13-15`
- Create: `data/fallback/` (copy of the test bundle until Task 13)
- Modify: `tests/conftest.py`, `tests/test_data.py`
- Create: `tests/test_snapshot.py`

**Interfaces:**
- Consumes: `config.DATA_RELEASE_URL`, `config.STALE_BUSINESS_DAYS`; bundle layout from Task 2.
- Produces in `risk_engine.data`:
  - `class SnapshotUnavailable(RuntimeError)`
  - `@dataclass(frozen=True) class Snapshot: prices: pd.DataFrame; universe: dict; factors: pd.DataFrame; report: dict; source: str; warning: str | None = None`, with properties `as_of -> str` and `quarantined -> dict[str, str]`
  - `BUNDLE_FILES: tuple[str, ...]`, `FALLBACK_DIR: str`
  - `read_bundle(directory: str, source: str) -> Snapshot`
  - `load_snapshot(*, data_dir=None, release_url=DATA_RELEASE_URL, cache_dir=None, fallback_dir=FALLBACK_DIR, fetch=_download, max_age_s=3600, use_memo=True) -> Snapshot`
  - `is_stale(snap: Snapshot, today: dt.date | None = None) -> bool`
  - `freshness_line(snap: Snapshot) -> str`
  - `load_prices(prefer_live=False) -> (prices, log_returns, label)`: legacy view, deleted in Task 12
- Env vars: `MARKETPLUG_DATA_DIR` pins a bundle directory; `MARKETPLUG_NO_LIVE` disables live lookups (Task 4). `tests/conftest.py` sets both.

- [ ] **Step 1: Point the test session at the frozen bundle**

At the very top of `tests/conftest.py`, before the other imports:

```python
import os

# Every test reads the frozen June 2026 bundle and never touches the network.
os.environ.setdefault("MARKETPLUG_DATA_DIR",
                      os.path.join(os.path.dirname(__file__), "fixtures", "snapshot"))
os.environ.setdefault("MARKETPLUG_NO_LIVE", "1")
```

Then change the `market` fixture so the golden master keeps its exact legacy construction:

```python
@pytest.fixture(scope="session")
def market():
    from risk_engine.data import load_snapshot
    prices = load_snapshot().prices.ffill().dropna()
    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr
```

Remove the now-unused `from risk_engine.data import load_prices` import from `conftest.py`.

- [ ] **Step 2: Write the failing loader tests**

`tests/test_snapshot.py`:

```python
import datetime as dt
import json
import os
import shutil

import pytest

from risk_engine import data

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "snapshot")


def _release_fetcher(src_dir, calls):
    """Serve bundle files from a directory as if they were release assets."""
    def fetch(url):
        calls.append(url)
        with open(os.path.join(src_dir, url.rsplit("/", 1)[1]), "rb") as f:
            return f.read()
    return fetch


def _broken_fetch(url):
    raise OSError("network down")


def test_pinned_dir_wins(monkeypatch):
    snap = data.load_snapshot(use_memo=False)
    assert snap.source == "pinned"
    assert snap.as_of == "2026-06-18"
    assert "SPY" in snap.prices.columns and "AAPL" in snap.universe


def test_release_download_then_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    calls = []
    fetch = _release_fetcher(FIXTURE, calls)
    first = data.load_snapshot(release_url="https://x/rel", cache_dir=str(tmp_path),
                               fetch=fetch, use_memo=False)
    assert first.source == "release"
    assert len(calls) == 4                       # report + the three data files
    calls.clear()
    second = data.load_snapshot(release_url="https://x/rel", cache_dir=str(tmp_path),
                                fetch=fetch, use_memo=False)
    assert second.source == "cache"
    assert calls == ["https://x/rel/refresh_report.json"]   # same as_of: nothing re-downloaded


def test_new_release_replaces_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    newer = tmp_path / "newer"
    shutil.copytree(FIXTURE, newer)
    report = json.loads((newer / "refresh_report.json").read_text())
    report["as_of"] = "2026-06-19"
    (newer / "refresh_report.json").write_text(json.dumps(report))
    cache = tmp_path / "cache"
    data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                       fetch=_release_fetcher(FIXTURE, []), use_memo=False)
    snap = data.load_snapshot(release_url="https://x/rel", cache_dir=str(cache),
                              fetch=_release_fetcher(str(newer), []), use_memo=False)
    assert snap.source == "release" and snap.as_of == "2026-06-19"


def test_download_failure_falls_back_with_a_visible_warning(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    snap = data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=FIXTURE,
                              fetch=_broken_fetch, use_memo=False)
    assert snap.source == "fallback"
    assert "network down" in snap.warning


def test_missing_fallback_raises_loudly(monkeypatch, tmp_path):
    monkeypatch.delenv("MARKETPLUG_DATA_DIR")
    with pytest.raises(data.SnapshotUnavailable, match="missing"):
        data.load_snapshot(cache_dir=str(tmp_path), fallback_dir=str(tmp_path / "nope"),
                           fetch=_broken_fetch, use_memo=False)


def test_staleness_uses_business_days():
    snap = data.load_snapshot(use_memo=False)       # as_of Thu 2026-06-18
    assert not data.is_stale(snap, today=dt.date(2026, 6, 22))   # Mon: 2 business days
    assert data.is_stale(snap, today=dt.date(2026, 6, 24))       # Wed: 4 business days


def test_freshness_line_names_quarantined_tickers():
    snap = data.load_snapshot(use_memo=False)
    assert data.freshness_line(snap) == "Prices as of 2026-06-18 close · 25/25 tickers"
    report = dict(snap.report, quarantined={"XOM": "suspected bad print on 2026-06-17"})
    bad = data.Snapshot(snap.prices.drop(columns=["XOM"]), snap.universe, snap.factors,
                        report, "pinned")
    assert data.freshness_line(bad) == ("Prices as of 2026-06-18 close · 24/25 tickers"
                                        " · 1 held back: XOM")
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_snapshot.py -v`
Expected: FAIL with `AttributeError: module 'risk_engine.data' has no attribute 'load_snapshot'`.

- [ ] **Step 4: Implement the loader**

Replace `risk_engine/data.py` with:

```python
"""Market data access. Snapshot first: a bundle built daily by a GitHub Action and
published as a release, cached on local disk, with a dated bundle in the repo as
the fallback when the download fails. Pure — no UI."""
import dataclasses
import datetime as dt
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine.config import DATA_RELEASE_URL, STALE_BUSINESS_DAYS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_DIR = os.path.join(_REPO, "data", "fallback")
BUNDLE_FILES = ("prices.csv.gz", "universe.json", "factors.csv", "refresh_report.json")
_UA = {"User-Agent": "Mozilla/5.0"}


class SnapshotUnavailable(RuntimeError):
    """No complete bundle could be read from any source."""


@dataclass(frozen=True)
class Snapshot:
    prices: pd.DataFrame      # one column per ticker; NaN before a ticker listed
    universe: dict            # ticker -> name, sector, type, asset_class, in_sp500, curated
    factors: pd.DataFrame     # Fama-French daily factors, decimals
    report: dict              # as_of, failed, quarantined, ... from the build
    source: str               # "pinned" | "release" | "cache" | "fallback"
    warning: str | None = None

    @property
    def as_of(self):
        return self.report["as_of"]

    @property
    def quarantined(self):
        return self.report.get("quarantined", {})


def _bundle_complete(directory):
    return all(os.path.exists(os.path.join(directory, f)) for f in BUNDLE_FILES)


def read_bundle(directory, source):
    missing = [f for f in BUNDLE_FILES if not os.path.exists(os.path.join(directory, f))]
    if missing:
        raise SnapshotUnavailable(f"{directory} is missing {', '.join(missing)}")
    prices = pd.read_csv(os.path.join(directory, "prices.csv.gz"), index_col=0, parse_dates=True)
    factors = pd.read_csv(os.path.join(directory, "factors.csv"), index_col=0, parse_dates=True)
    with open(os.path.join(directory, "universe.json")) as f:
        universe = json.load(f)
    with open(os.path.join(directory, "refresh_report.json")) as f:
        report = json.load(f)
    return Snapshot(prices.sort_index(), universe, factors, report, source)


def _download(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _atomic_write(path, payload):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)


def _from_release(release_url, cache_dir, fetch):
    report_bytes = fetch(f"{release_url}/refresh_report.json")
    remote_as_of = json.loads(report_bytes)["as_of"]
    cached_report = os.path.join(cache_dir, "refresh_report.json")
    if _bundle_complete(cache_dir):
        with open(cached_report) as f:
            if json.load(f).get("as_of") == remote_as_of:
                return read_bundle(cache_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    for name in ("prices.csv.gz", "universe.json", "factors.csv"):
        _atomic_write(os.path.join(cache_dir, name), fetch(f"{release_url}/{name}"))
    # The report goes last: a cache holding the new report always holds the new data.
    _atomic_write(cached_report, report_bytes)
    return read_bundle(cache_dir, "release")


_MEMO = {"snap": None, "at": 0.0}


def load_snapshot(*, data_dir=None, release_url=DATA_RELEASE_URL, cache_dir=None,
                  fallback_dir=FALLBACK_DIR, fetch=_download, max_age_s=3600, use_memo=True):
    """Return the freshest readable bundle. Raises SnapshotUnavailable only when
    even the bundled fallback can't be read."""
    pinned = data_dir or os.environ.get("MARKETPLUG_DATA_DIR")
    if pinned:
        return read_bundle(pinned, "pinned")
    if use_memo and _MEMO["snap"] is not None and time.time() - _MEMO["at"] < max_age_s:
        return _MEMO["snap"]
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "marketplug_cache")
    try:
        snap = _from_release(release_url, cache_dir, fetch)
    except Exception as e:  # network, HTTP status, bad JSON or a corrupt file: use the fallback
        snap = dataclasses.replace(
            read_bundle(fallback_dir, "fallback"),
            warning=f"Could not download today's data ({type(e).__name__}: {e}); "
                    f"showing the bundled copy instead.")
    if use_memo:
        _MEMO.update(snap=snap, at=time.time())
    return snap


def is_stale(snap, today=None):
    today = today or dt.date.today()
    as_of = dt.date.fromisoformat(snap.as_of)
    return int(np.busday_count(as_of, today)) > STALE_BUSINESS_DAYS


def freshness_line(snap):
    n_total = len(snap.universe)
    present = [t for t in snap.universe if t in snap.prices.columns]
    n_now = int(snap.prices[present].iloc[-1].notna().sum()) if present else 0
    line = f"Prices as of {snap.as_of} close · {n_now}/{n_total} tickers"
    held = snap.quarantined
    if held:
        line += f" · {len(held)} held back: {', '.join(sorted(held))}"
    return line


def load_prices(prefer_live=False):
    """Legacy (prices, log_returns, label) view for callers not yet on
    risk_engine.portfolio. `prefer_live` is accepted and ignored."""
    snap = load_snapshot()
    prices = snap.prices.ffill().dropna()
    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr, f"{snap.source} snapshot · {snap.as_of}"
```

- [ ] **Step 5: Point factors at the snapshot**

In `risk_engine/factors.py`, delete the `_FACTORS = ...` line and the `import os` line, then replace `load_factors`:

```python
def load_factors():
    """Return the factor DataFrame (DatetimeIndex; columns Mkt-RF, SMB, HML, RF; decimals)."""
    from risk_engine.data import load_snapshot
    return load_snapshot().factors
```

- [ ] **Step 6: Ship the test bundle as the first fallback**

```bash
mkdir -p data/fallback
cp tests/fixtures/snapshot/* data/fallback/
```

(Task 13 replaces it with the first real release.)

- [ ] **Step 7: Update the legacy data test**

Replace `tests/test_data.py` with:

```python
import numpy as np

from risk_engine.data import load_prices


def test_legacy_view_shape():
    prices, lr, source = load_prices()
    assert prices.shape[0] > 2000          # ~8.5 years of daily rows
    assert "SPY" in prices.columns
    assert lr.shape[0] == prices.shape[0] - 1
    assert "snapshot" in source


def test_log_returns_are_finite():
    _, lr, _ = load_prices()
    assert np.isfinite(lr.to_numpy()).all()
```

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass, including `tests/test_metrics.py::test_metrics_reproduce_golden` and the three app smoke tests.

- [ ] **Step 9: Commit**

```bash
git add risk_engine/data.py risk_engine/factors.py data/fallback tests/conftest.py tests/test_data.py tests/test_snapshot.py
git commit -m "Load market data from a release bundle with cache and labeled fallback"
```

---

### Task 4: Live lookup for tickers outside the snapshot

**Files:**
- Modify: `risk_engine/data.py` (append)
- Create: `tests/test_live.py`

**Interfaces:**
- Produces in `risk_engine.data`:
  - `TICKER_RE: re.Pattern`
  - `normalize_ticker(raw: str) -> str`
  - `@dataclass(frozen=True) class LiveResult: ticker: str; status: str; reason: str = ""; prices: pd.Series | None = None; meta: dict | None = None`, where status ∈ `"ok" | "invalid" | "not_found" | "non_usd" | "unavailable"`
  - `live_meta(quote: dict) -> dict` (same keys as a `universe.json` entry)
  - `fetch_live(sym: str, *, get_json=_get_json, timeout=8.0, start="2018-01-01") -> LiveResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_live.py`:

```python
import pytest

from risk_engine import data

SEARCH_PLTR = {"quotes": [{"symbol": "PLTR", "quoteType": "EQUITY", "sector": "Technology",
                           "longname": "Palantir Technologies Inc."}]}
SEARCH_VOO = {"quotes": [{"symbol": "VOO", "quoteType": "ETF", "shortname": "Vanguard S&P 500"}]}


def chart(currency="USD", closes=(10.0, 11.0, None, 12.0)):
    ts = [1735828200, 1735914600, 1736173800, 1736260200][: len(closes)]
    return {"chart": {"result": [{"meta": {"currency": currency}, "timestamp": ts,
                                  "indicators": {"adjclose": [{"adjclose": list(closes)}]}}]}}


def fake_get(search, chart_payload=None, fail=None):
    def get_json(url, timeout):
        if fail:
            raise fail
        return search if "/search" in url else chart_payload
    return get_json


@pytest.fixture(autouse=True)
def allow_live(monkeypatch):
    monkeypatch.delenv("MARKETPLUG_NO_LIVE", raising=False)


@pytest.mark.parametrize("raw,expected", [
    ("brk.b", "BRK-B"), (" aapl ", "AAPL"), ("BF.B", "BF-B"), ("RELIANCE.NS", "RELIANCE.NS"),
])
def test_normalize_ticker(raw, expected):
    assert data.normalize_ticker(raw) == expected


@pytest.mark.parametrize("bad", ["", "AAPL/../X", "<SCRIPT>", "TOOLONGTICKER1", "A B"])
def test_invalid_symbols_never_reach_the_network(bad):
    def explode(url, timeout):
        raise AssertionError("network called for an invalid symbol")
    r = data.fetch_live(bad, get_json=explode)
    assert r.status == "invalid"


def test_ok_stock_returns_prices_and_meta():
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, chart()))
    assert r.status == "ok"
    assert list(r.prices) == [10.0, 11.0, 12.0]          # the null close is dropped, not filled
    assert r.meta["sector"] == "Technology" and r.meta["asset_class"] == "Equity"
    assert r.meta["in_sp500"] is False


def test_unknown_etf_is_labeled_as_a_black_box():
    r = data.fetch_live("VOO", get_json=fake_get(SEARCH_VOO, chart()))
    assert r.meta["type"] == "etf"
    assert r.meta["sector"] == "Fund (holdings unknown)"


def test_search_miss_is_not_found():
    r = data.fetch_live("XYZQQ", get_json=fake_get({"quotes": []}))
    assert r.status == "not_found"


def test_fuzzy_search_hit_for_a_different_symbol_is_not_found():
    r = data.fetch_live("APPL", get_json=fake_get({"quotes": [{"symbol": "AAPL"}]}))
    assert r.status == "not_found"


def test_non_usd_is_rejected_with_the_currency_named():
    search = {"quotes": [{"symbol": "RELIANCE.NS", "quoteType": "EQUITY"}]}
    r = data.fetch_live("RELIANCE.NS", get_json=fake_get(search, chart(currency="INR")))
    assert r.status == "non_usd" and "INR" in r.reason


def test_network_failure_is_unavailable_with_reason():
    r = data.fetch_live("PLTR", get_json=fake_get(None, fail=TimeoutError("slow")))
    assert r.status == "unavailable" and "TimeoutError" in r.reason


def test_malformed_chart_is_unavailable():
    broken = {"chart": {"result": [{"meta": {"currency": "USD"}}]}}
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, broken))
    assert r.status == "unavailable"


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("MARKETPLUG_NO_LIVE", "1")
    r = data.fetch_live("PLTR", get_json=fake_get(SEARCH_PLTR, chart()))
    assert r.status == "unavailable" and "disabled" in r.reason
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_live.py -v`
Expected: FAIL with `AttributeError: module 'risk_engine.data' has no attribute 'normalize_ticker'`.

- [ ] **Step 3: Implement**

Add `import re` and `import urllib.parse` to the imports of `risk_engine/data.py`, then append:

```python
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,9}$")
_CLASS_SHARE = re.compile(r"^[A-Z]{1,5}\.[A-Z]$")


def normalize_ticker(raw):
    """Upper-case, trim, and turn class-share dots into Yahoo's dashes (BRK.B -> BRK-B).
    Exchange suffixes like .NS keep their dot."""
    t = str(raw).strip().upper()
    return t.replace(".", "-") if _CLASS_SHARE.match(t) else t


@dataclass(frozen=True)
class LiveResult:
    ticker: str
    status: str               # "ok" | "invalid" | "not_found" | "non_usd" | "unavailable"
    reason: str = ""
    prices: pd.Series | None = None
    meta: dict | None = None


def _get_json(url, timeout):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def live_meta(quote):
    kind = {"EQUITY": "stock", "ETF": "etf"}.get(quote.get("quoteType"), "other")
    if kind == "stock":
        sector, asset_class = quote.get("sector") or "Unknown", "Equity"
    elif kind == "etf":
        sector, asset_class = "Fund (holdings unknown)", "Fund"
    else:
        sector, asset_class = quote.get("typeDisp") or "Other", "Other"
    name = quote.get("longname") or quote.get("shortname") or quote["symbol"]
    return {"name": name, "sector": sector, "type": kind, "asset_class": asset_class,
            "in_sp500": False, "curated": False}


def fetch_live(sym, *, get_json=_get_json, timeout=8.0, start="2018-01-01"):
    """Price history + metadata for one ticker outside the snapshot. Never raises:
    every outcome is a LiveResult with a reason a person can read."""
    if os.environ.get("MARKETPLUG_NO_LIVE"):
        return LiveResult(sym, "unavailable", "live lookups are disabled in this environment")
    if not TICKER_RE.fullmatch(sym or ""):
        return LiveResult(sym, "invalid", "not a valid ticker symbol")
    q = urllib.parse.quote(sym, safe="")
    p1 = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(time.time())
    try:
        found = get_json("https://query1.finance.yahoo.com/v1/finance/search"
                         f"?q={q}&quotesCount=5&newsCount=0", timeout)
        quote = next((x for x in (found or {}).get("quotes", [])
                      if str(x.get("symbol", "")).upper() == sym), None)
        if quote is None:
            return LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
        payload = get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
                           f"?period1={p1}&period2={p2}&interval=1d", timeout)
    except Exception as e:  # network, HTTP status or malformed JSON
        return LiveResult(sym, "unavailable", f"data source unavailable ({type(e).__name__})")
    results = ((payload or {}).get("chart") or {}).get("result") or []
    if not results:
        return LiveResult(sym, "not_found", "no price history on Yahoo Finance")
    r = results[0]
    currency = (r.get("meta") or {}).get("currency")
    if currency != "USD":
        return LiveResult(sym, "non_usd", f"priced in {currency or 'an unknown currency'}, not USD")
    try:
        days = [dt.datetime.fromtimestamp(t, dt.timezone.utc).date() for t in r["timestamp"]]
        closes = r["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError):
        return LiveResult(sym, "unavailable", "price history came back in an unexpected format")
    s = pd.Series(closes, index=pd.to_datetime(days), name=sym, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty:
        return LiveResult(sym, "not_found", "no price history on Yahoo Finance")
    return LiveResult(sym, "ok", "", s, live_meta(quote))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_live.py -v`
Expected: all PASS.

- [ ] **Step 5: Smoke-test against real Yahoo (local only, not in CI)**

```bash
python -c "import os; os.environ.pop('MARKETPLUG_NO_LIVE', None); import truststore; truststore.inject_into_ssl(); from risk_engine import data; [print(t, data.fetch_live(t).status, data.fetch_live(t).reason) for t in ['PLTR','XYZQQ','RELIANCE.NS']]"
```

Expected: `PLTR ok`, `XYZQQ not_found …`, `RELIANCE.NS non_usd priced in INR, not USD`.

- [ ] **Step 6: Commit**

```bash
git add risk_engine/data.py tests/test_live.py
git commit -m "Look up tickers outside the snapshot live, with a reason for every failure"
```

- [ ] **Step 7 (only if Task 1 showed Streamlit Cloud cannot reach Yahoo via urllib): add a yfinance transport**

Add `yfinance==<version pinned in Task 1>` to `requirements.txt` and `requirements-api.txt`. In `risk_engine/data.py`, append:

```python
def _yf_history(sym, start):
    import yfinance as yf
    hist = yf.Ticker(sym).history(start=start, auto_adjust=True)["Close"].dropna()
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    return hist.rename(sym)
```

In `fetch_live`, wrap the chart request: on any exception from the urllib chart call, try `_yf_history(sym, start)`. If that succeeds, return `LiveResult(sym, "ok", "", series, live_meta(quote))` (yfinance's `.history` is USD for US listings; keep the currency check by reading `yf.Ticker(sym).fast_info["currency"]` and returning `non_usd` when it isn't `"USD"`). Add a test `test_yfinance_fallback_used_when_chart_fails` that monkeypatches `data._yf_history` to return a 3-point series and a `get_json` whose chart call raises; assert `status == "ok"`. Commit: `git commit -m "Fall back to yfinance when Yahoo's chart endpoint blocks the host"`.

---

### Task 5: Importer — paste, CSV, table rows, share link

**Files:**
- Create: `risk_engine/importer.py`
- Create: `tests/test_importer.py`

**Interfaces:**
- Consumes: `data.normalize_ticker`, `data.TICKER_RE` (Task 4).
- Produces in `risk_engine.importer`:
  - `@dataclass class ParseResult: holdings: dict[str, float]; notes: list[str]`
  - `parse_amount(text: str) -> float | None`
  - `parse_paste(text: str) -> ParseResult`
  - `parse_csv(text: str, *, last_price=None) -> ParseResult`, where `last_price: Callable[[str], float | None]`
  - `from_rows(rows: list[tuple]) -> ParseResult` (editor table rows: ticker, amount)
  - `encode_share(holdings: dict) -> str`, `decode_share(param: str) -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/test_importer.py`:

```python
import math

import pytest

from risk_engine import importer


@pytest.mark.parametrize("text,value", [
    ("10000", 10000.0), ("$25,000", 25000.0), ("8,500.50", 8500.5), ("$ 1 000", 1000.0),
    ("(1,234.00)", -1234.0), ("-5", -5.0), ("abc", None), ("", None), ("1.2.3", None),
])
def test_parse_amount(text, value):
    assert importer.parse_amount(text) == value


def test_paste_forgives_case_dollars_commas_tabs_and_semicolons():
    r = importer.parse_paste("aapl 10000\nVOO, $25,000\nmsft\t8,500.50\nTLT;5000\n\n")
    assert r.holdings == {"AAPL": 10000.0, "VOO": 25000.0, "MSFT": 8500.5, "TLT": 5000.0}
    assert r.notes == []


def test_paste_merges_duplicates_and_says_so():
    r = importer.parse_paste("AAPL 10000\naapl 8000")
    assert r.holdings == {"AAPL": 18000.0}
    assert r.notes == ["AAPL appeared 2 times — merged into $18,000"]


def test_paste_class_share_dot_becomes_dash():
    assert importer.parse_paste("brk.b 5000").holdings == {"BRK-B": 5000.0}


@pytest.mark.parametrize("line,fragment", [
    ("AAPL", "expected 'TICKER AMOUNT'"),
    ("AAPL lots", "no dollar amount"),
    ("AAPL 0", "$0 amount"),
    ("AAPL (1,234.00)", "short positions are not supported"),
    ("TSLA -500", "short positions are not supported"),
    ("<img src=x> 100", "not a valid ticker"),
])
def test_paste_skips_bad_lines_with_a_reason(line, fragment):
    r = importer.parse_paste(line)
    assert r.holdings == {}
    assert len(r.notes) == 1 and fragment in r.notes[0]


SCHWAB_LIKE = (
    '"Positions for account Individual ...123 as of 10/03/2026"\n'
    '\n'
    '"Symbol","Description","Quantity","Price","Market Value"\n'
    '"AAPL","APPLE INC","50","$230.00","$11,500.00"\n'
    '"VOO","VANGUARD S&P 500","20","$600.00","$12,000.00"\n'
    '"Cash & Cash Investments","--","--","--","$1,250.00"\n'
    '"Account Total","--","--","--","$24,750.00"\n'
)


def test_csv_finds_header_below_account_info_and_skips_junk_rows():
    r = importer.parse_csv(SCHWAB_LIKE)
    assert r.holdings == {"AAPL": 11500.0, "VOO": 12000.0}
    assert sum("not a valid ticker" in n for n in r.notes) == 2


def test_csv_handles_byte_order_mark_and_lowercase_headers():
    r = importer.parse_csv("\ufeffticker,amount\nmsft,9000\n".lstrip("\ufeff"))
    assert r.holdings == {"MSFT": 9000.0}


def test_csv_quantity_only_is_valued_at_latest_close_and_labeled():
    r = importer.parse_csv("Symbol,Quantity\nAAPL,10\nZZZZ,5\n",
                           last_price=lambda t: {"AAPL": 200.0}.get(t))
    assert r.holdings == {"AAPL": 2000.0}
    assert any("quantity × latest close" in n for n in r.notes)
    assert any("no price for ZZZZ" in n for n in r.notes)


def test_csv_without_ticker_column_names_the_expected_headers():
    r = importer.parse_csv("Name,Value\nApple,100\n")
    assert r.holdings == {} and "Symbol or Ticker" in r.notes[0]


def test_csv_without_value_or_quantity_column():
    r = importer.parse_csv("Symbol,Description\nAAPL,Apple\n")
    assert r.holdings == {} and "Market Value" in r.notes[0]


def test_from_rows_ignores_blank_editor_rows():
    r = importer.from_rows([("AAPL", 1000.0), (None, math.nan), ("", None), ("msft", 500)])
    assert r.holdings == {"AAPL": 1000.0, "MSFT": 500.0} and r.notes == []


def test_share_link_round_trip():
    h = {"AAPL": 10000.4, "BRK-B": 5000.0}
    assert importer.encode_share(h) == "AAPL:10000,BRK-B:5000"
    assert importer.decode_share("AAPL:10000,BRK-B:5000") == {"AAPL": 10000.0, "BRK-B": 5000.0}
    assert importer.decode_share("") == {}
    assert importer.decode_share("garbage") == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_importer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'risk_engine.importer'`.

- [ ] **Step 3: Implement**

`risk_engine/importer.py`:

```python
"""Turn what people paste, type or export from a broker into {ticker: dollars}.
Nothing is dropped silently: every skipped line or merged duplicate becomes a
note the UI shows. Pure — no UI."""
import csv
import io
import math
import re
from dataclasses import dataclass, field

from risk_engine.data import TICKER_RE, normalize_ticker

TICKER_HEADERS = ("symbol", "ticker")
VALUE_HEADERS = ("market value", "current value", "value", "amount")
QTY_HEADERS = ("quantity", "shares", "qty")

_AMOUNT = re.compile(r"^\(?-?\$?-?[\d,]*\.?\d+\)?$")


@dataclass
class ParseResult:
    holdings: dict = field(default_factory=dict)   # ticker -> dollars, duplicates summed
    notes: list = field(default_factory=list)      # skipped input and merges, for the user


def parse_amount(text):
    """'$25,000.50' -> 25000.5; '(1,234)' and '-5' -> negative; anything else -> None."""
    t = str(text).strip().replace(" ", "")
    if not t or not _AMOUNT.match(t):
        return None
    value = float(re.sub(r"[^\d.]", "", t))
    return -value if (t.startswith("(") or "-" in t) else value


class _Collector:
    def __init__(self):
        self.holdings, self.notes, self.counts = {}, [], {}

    def add(self, raw_ticker, amount, where):
        t = normalize_ticker(raw_ticker)
        if not TICKER_RE.fullmatch(t):
            self.notes.append(f"{where}: '{raw_ticker}' is not a valid ticker — skipped")
        elif amount is None:
            self.notes.append(f"{where}: no dollar amount for {t} — skipped")
        elif amount < 0:
            self.notes.append(f"{where}: {t} has a negative amount "
                              f"(short positions are not supported) — skipped")
        elif amount == 0:
            self.notes.append(f"{where}: {t} has a $0 amount — skipped")
        else:
            self.counts[t] = self.counts.get(t, 0) + 1
            self.holdings[t] = self.holdings.get(t, 0.0) + float(amount)

    def result(self):
        merges = [f"{t} appeared {n} times — merged into ${self.holdings[t]:,.0f}"
                  for t, n in self.counts.items() if n > 1]
        return ParseResult(self.holdings, self.notes + merges)


def parse_paste(text):
    c = _Collector()
    for i, line in enumerate(str(text).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,;]+", line, maxsplit=1) if p]
        if len(parts) != 2:
            c.notes.append(f"line {i}: expected 'TICKER AMOUNT', got '{line}' — skipped")
            continue
        c.add(parts[0], parse_amount(parts[1]), f"line {i}")
    return c.result()


def _find(header, wanted):
    by_name = {h.strip().lower(): h for h in header if h}
    return next((by_name[w] for w in wanted if w in by_name), None)


def parse_csv(text, *, last_price=None):
    """Broker exports often put account details above the header row, so the
    header is the first row that names a ticker column. `last_price(ticker)` values
    quantity-only files; it may return None when no price is known."""
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    hdr_i = next((i for i, r in enumerate(rows) if _find(r, TICKER_HEADERS)), None)
    if hdr_i is None:
        return ParseResult({}, ["no ticker column found — expected a header named Symbol or Ticker"])
    header = rows[hdr_i]
    t_col = header.index(_find(header, TICKER_HEADERS))
    v_name, q_name = _find(header, VALUE_HEADERS), _find(header, QTY_HEADERS)
    if v_name is None and (q_name is None or last_price is None):
        return ParseResult({}, ["no value column found — expected Market Value, Current Value, "
                                "Value or Amount (or Quantity)"])
    c = _Collector()
    if v_name is None:
        c.notes.append("no value column — each holding valued as quantity × latest close")
    col = header.index(v_name if v_name is not None else q_name)
    for i, r in enumerate(rows[hdr_i + 1:], hdr_i + 2):
        if not any(cell.strip() for cell in r):
            continue
        if len(r) <= max(t_col, col):
            c.notes.append(f"row {i}: too few columns — skipped")
            continue
        sym, number = r[t_col].strip(), parse_amount(r[col])
        if v_name is None and number is not None and number > 0:
            price = last_price(normalize_ticker(sym))
            if price is None:
                c.notes.append(f"row {i}: no price for {normalize_ticker(sym)} to value its shares — skipped")
                continue
            number *= price
        c.add(sym, number, f"row {i}")
    return c.result()


def from_rows(rows):
    """Rows from the editable holdings table. A row with no ticker and no amount is
    an empty line the user added, not a mistake, so it is skipped quietly."""
    c = _Collector()
    for i, (ticker, amount) in enumerate(rows, 1):
        blank_ticker = ticker is None or str(ticker).strip() == ""
        blank_amount = amount is None or (isinstance(amount, float) and math.isnan(amount))
        if blank_ticker and blank_amount:
            continue
        c.add("" if blank_ticker else ticker, None if blank_amount else float(amount),
              f"table row {i}")
    return c.result()


def encode_share(holdings):
    return ",".join(f"{t}:{round(a)}" for t, a in holdings.items())


def decode_share(param):
    if not param:
        return {}
    return parse_paste("\n".join(p.replace(":", " ", 1) for p in param.split(","))).holdings
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_importer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add risk_engine/importer.py tests/test_importer.py
git commit -m "Parse pasted and broker-exported holdings, reporting every skipped line"
```

---

### Task 6: Portfolio resolution, honest window, crisis coverage

**Files:**
- Create: `risk_engine/portfolio.py`
- Create: `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `data.Snapshot`, `data.fetch_live`, `data.LiveResult`, `data.normalize_ticker`; `config.CRISES`, `WINDOW_OK_DAYS`, `WINDOW_MIN_DAYS`, `LIVE_MAX_API`, `LIVE_BUDGET_API_S`.
- Produces in `risk_engine.portfolio`:
  - `@dataclass(frozen=True) class Rejection: ticker: str; amount: float; reason: str`
  - `@dataclass class Resolution: holdings: dict[str, float]; prices: pd.DataFrame; meta: dict[str, dict]; source: dict[str, str]; rejected: list[Rejection]`, with property `tickers -> list[str]`
  - `resolve_holdings(holdings: dict[str, float], snap, *, live_fetch=None, max_live=LIVE_MAX_API, budget_s=LIVE_BUDGET_API_S, clock=time.monotonic) -> Resolution`
  - `@dataclass class Window: returns: pd.DataFrame; start: pd.Timestamp; end: pd.Timestamp; first_dates: dict; n_days: int; status: str; limiting: str; message: str`, with status ∈ `"ok" | "short" | "too_short"`
  - `portfolio_window(prices: pd.DataFrame, tickers: list[str], benchmark: pd.Series | None = None) -> Window`
  - `crisis_coverage(prices: pd.DataFrame, tickers: list[str], crises=CRISES) -> list[dict]`, each `{"label", "start", "end", "context", "missing": list[str]}`

- [ ] **Step 1: Write the failing tests**

`tests/test_portfolio.py`:

```python
import numpy as np
import pandas as pd

from risk_engine import data, portfolio
from tests.conftest import HOLDINGS


def snap():
    return data.load_snapshot(use_memo=False)


def ok_live(series):
    def fetch(sym):
        meta = {"name": sym, "sector": "Technology", "type": "stock", "asset_class": "Equity",
                "in_sp500": False, "curated": False}
        return data.LiveResult(sym, "ok", "", series.rename(sym), meta)
    return fetch


def test_known_tickers_come_from_the_snapshot():
    r = portfolio.resolve_holdings({"aapl": 1000, "TLT": 500}, snap(),
                                   live_fetch=lambda s: (_ for _ in ()).throw(AssertionError))
    assert r.tickers == ["AAPL", "TLT"] and r.rejected == []
    assert r.source == {"AAPL": "snapshot", "TLT": "snapshot"}
    assert r.meta["TLT"]["asset_class"] == "Bond"


def test_unknown_ticker_goes_live_and_failures_carry_reasons():
    s = snap()
    calls = []

    def fetch(sym):
        calls.append(sym)
        return data.LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
    r = portfolio.resolve_holdings({"AAPL": 1000, "XYZQQ": 300}, s, live_fetch=fetch)
    assert calls == ["XYZQQ"]
    assert r.rejected == [portfolio.Rejection("XYZQQ", 300, "no such ticker on Yahoo Finance")]


def test_live_cap_and_time_budget():
    s = snap()
    base = s.prices["AAPL"].dropna()
    many = {f"ZZ{i}": 100.0 for i in range(7)}
    r = portfolio.resolve_holdings(many, s, live_fetch=ok_live(base), max_live=5)
    assert len(r.tickers) == 5
    assert all("max 5" in x.reason for x in r.rejected) and len(r.rejected) == 2

    ticks = iter([0.0, 0.0, 11.0, 11.0, 11.0])
    r2 = portfolio.resolve_holdings({"ZZ1": 1, "ZZ2": 1}, s, live_fetch=ok_live(base),
                                    budget_s=10.0, clock=lambda: next(ticks))
    assert r2.tickers == ["ZZ1"] and "time budget" in r2.rejected[0].reason


def test_quarantined_ticker_is_rejected_not_looked_up():
    s = snap()
    held = data.Snapshot(s.prices, s.universe, s.factors,
                         dict(s.report, quarantined={"XOM": "suspected bad print on 2026-06-17"}),
                         "pinned")
    r = portfolio.resolve_holdings({"AAPL": 1, "XOM": 1}, held,
                                   live_fetch=lambda t: (_ for _ in ()).throw(AssertionError))
    assert r.rejected[0].ticker == "XOM" and "bad print" in r.rejected[0].reason


def test_window_matches_legacy_returns_for_full_history_portfolio(market):
    _, legacy_lr = market
    r = portfolio.resolve_holdings({t: 1.0 for t in HOLDINGS}, snap())
    w = portfolio.portfolio_window(r.prices, r.tickers)
    assert w.status == "ok"
    diff = (w.returns[HOLDINGS] - legacy_lr[HOLDINGS]).abs().max().max()
    assert diff < 1e-12


def _young_listing(s, start="2024-04-02"):
    young = s.prices["AAPL"].copy() * 0.3
    young[young.index < start] = np.nan
    return young


def test_young_listing_shortens_the_window_but_never_erases_crisis_history():
    s = snap()
    r = portfolio.resolve_holdings({"AAPL": 1, "TLT": 1, "NEWCO": 1}, s,
                                   live_fetch=ok_live(_young_listing(s)))
    w = portfolio.portfolio_window(r.prices, r.tickers)
    assert w.limiting == "NEWCO" and w.start >= pd.Timestamp("2024-04-02")
    assert w.status == "ok"                          # 2024-04 -> 2026-06 is > 504 days
    cov = {c["label"]: c["missing"] for c in portfolio.crisis_coverage(r.prices, r.tickers)}
    assert cov["COVID-19 Crash"] == ["NEWCO"]
    assert cov["2025 Tariff Shock"] == []


def test_short_and_too_short_windows():
    s = snap()
    short = portfolio.resolve_holdings({"AAPL": 1, "NEWCO": 1}, s,
                                       live_fetch=ok_live(_young_listing(s, "2025-01-02")))
    w = portfolio.portfolio_window(short.prices, short.tickers)
    assert w.status == "short" and "NEWCO" in w.message
    tiny = portfolio.resolve_holdings({"AAPL": 1, "NEWCO": 1}, s,
                                      live_fetch=ok_live(_young_listing(s, "2026-01-02")))
    w2 = portfolio.portfolio_window(tiny.prices, tiny.tickers)
    assert w2.status == "too_short" and "NEWCO" in w2.message


def test_benchmark_is_added_once_even_when_held():
    s = snap()
    r = portfolio.resolve_holdings({"AAPL": 1, "SPY": 1}, s)
    w = portfolio.portfolio_window(r.prices, r.tickers, benchmark=s.prices["SPY"])
    assert list(w.returns.columns).count("SPY") == 1
    r2 = portfolio.resolve_holdings({"AAPL": 1, "TLT": 1}, s)
    w2 = portfolio.portfolio_window(r2.prices, r2.tickers, benchmark=s.prices["SPY"])
    assert "SPY" in w2.returns.columns and w2.returns["SPY"].notna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'risk_engine.portfolio'`.

- [ ] **Step 3: Implement**

`risk_engine/portfolio.py`:

```python
"""Resolve a user's holdings against the snapshot (or live Yahoo for anything
outside it), then find the date window they can honestly be analysed over.
Pure — no UI."""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine import data
from risk_engine.config import (CRISES, LIVE_BUDGET_API_S, LIVE_MAX_API,
                                WINDOW_MIN_DAYS, WINDOW_OK_DAYS)


@dataclass(frozen=True)
class Rejection:
    ticker: str
    amount: float
    reason: str


@dataclass
class Resolution:
    holdings: dict        # accepted ticker -> dollars, input order kept
    prices: pd.DataFrame  # accepted tickers' prices; NaN before each listed
    meta: dict            # ticker -> universe-style metadata
    source: dict          # ticker -> "snapshot" | "live"
    rejected: list        # Rejection, input order

    @property
    def tickers(self):
        return list(self.holdings)


def resolve_holdings(holdings, snap, *, live_fetch=None, max_live=LIVE_MAX_API,
                     budget_s=LIVE_BUDGET_API_S, clock=time.monotonic):
    live_fetch = live_fetch or data.fetch_live
    accepted, meta, source, cols, rejected = {}, {}, {}, {}, []
    started, n_live = clock(), 0
    for raw, amount in holdings.items():
        t = data.normalize_ticker(raw)
        if t in accepted:
            accepted[t] += amount
            continue
        if t in snap.quarantined:
            rejected.append(Rejection(t, amount,
                                      f"held back by today's data checks ({snap.quarantined[t]})"))
            continue
        if t in snap.universe and t in snap.prices.columns:
            accepted[t], meta[t], source[t] = amount, snap.universe[t], "snapshot"
            cols[t] = snap.prices[t]
            continue
        if n_live >= max_live:
            rejected.append(Rejection(t, amount, f"too many tickers outside the dataset "
                                                 f"in one request (max {max_live})"))
            continue
        if clock() - started > budget_s:
            rejected.append(Rejection(t, amount, "lookup time budget used up — try again"))
            continue
        n_live += 1
        res = live_fetch(t)
        if res.status != "ok":
            rejected.append(Rejection(t, amount, res.reason))
            continue
        accepted[t], meta[t], source[t] = amount, res.meta, "live"
        cols[t] = res.prices
    prices = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    return Resolution(accepted, prices, meta, source, rejected)


@dataclass
class Window:
    returns: pd.DataFrame  # log returns over the window; holdings (+ benchmark), no gaps
    start: pd.Timestamp
    end: pd.Timestamp
    first_dates: dict      # ticker -> first date with a price
    n_days: int
    status: str            # "ok" | "short" | "too_short"
    limiting: str          # the holding whose listing date sets the start
    message: str


def portfolio_window(prices, tickers, benchmark=None):
    sub = prices[tickers]
    first = {t: sub[t].first_valid_index() for t in tickers}
    limiting = max(tickers, key=lambda t: first[t])
    # Align on dates where every holding has a price, THEN difference: a gap is
    # never filled with an invented price, and a multi-day move lands on one row.
    aligned = sub.loc[first[limiting]:].dropna()
    returns = np.log(aligned / aligned.shift(1)).iloc[1:].copy()
    if benchmark is not None and benchmark.name not in tickers:
        b = benchmark.reindex(aligned.index).ffill()
        returns[benchmark.name] = np.log(b / b.shift(1)).iloc[1:]
    n = len(returns)
    since = first[limiting].date()
    if n >= WINDOW_OK_DAYS:
        status, message = "ok", ""
    elif n >= WINDOW_MIN_DAYS:
        status = "short"
        message = (f"Results use {n} trading days (since {since}), limited by {limiting}'s "
                   f"listing date. With under two years of shared history, VaR is less reliable.")
    else:
        status = "too_short"
        message = (f"{limiting} has only {n} trading days of history (since {since}). That is "
                   f"under a year, too little for these risk numbers to mean anything. Remove "
                   f"{limiting} or analyse it once it has a year of prices.")
    return Window(returns, aligned.index[0], aligned.index[-1], first, n, status, limiting, message)


def crisis_coverage(prices, tickers, crises=CRISES):
    """For each crisis, which holdings weren't trading yet. A crisis a holding
    missed is reported, never silently dropped."""
    out = []
    for label, start, end, context in crises:
        missing = []
        for t in tickers:
            first = prices[t].first_valid_index()
            if first is None or first > pd.Timestamp(start):
                missing.append(t)
        out.append({"label": label, "start": start, "end": end, "context": context,
                    "missing": missing})
    return out
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_portfolio.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add risk_engine/portfolio.py tests/test_portfolio.py
git commit -m "Resolve holdings and analyse each portfolio over the window its holdings share"
```

---

### Task 7: API on the new data layer

**Files:**
- Modify: `risk_engine/analyze.py` (full rewrite)
- Modify: `api/schemas.py:1-40` (validation) and `AnalyzeResponse`
- Modify: `api/main.py` (`/health`, `/analyze`, `/portfolios/{pid}`)
- Modify: `tests/test_analyze.py`, `tests/test_schemas.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `portfolio.resolve_holdings`, `portfolio.portfolio_window`, `portfolio.crisis_coverage`, `data.load_snapshot`, `data.is_stale`, `data.normalize_ticker`, `data.TICKER_RE`, `config.BENCHMARK`, `config.MIN_HOLDINGS`, `config.MAX_HOLDINGS`, `config.LIVE_MAX_API`, `config.LIVE_BUDGET_API_S`.
- Produces:
  - `risk_engine.analyze.PortfolioError(ValueError)` with `.problems: list[dict]` (`{"ticker", "reason"}`)
  - `analyze_portfolio(holdings, *, confidence=0.95, include_backtest=True, snapshot=None, live_fetch=None) -> dict` with keys `metrics, factors, optimizer, data[, backtest]`
  - `data` block: `{"as_of": str, "window_start": str, "window_days": int, "window_status": str, "crisis_coverage": dict[str, list[str]]}`
  - API: 422 body for engine-level problems = `{"detail": {"problems": [...]}}`; `/health` → `{"status", "data_as_of", "stale", "data_source"}`, or 503 `{"status": "degraded", "detail": ...}`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_analyze.py`, change the structure test's expected key set and add tests:

```python
def test_analyze_returns_expected_structure():
    r = analyze_portfolio(HOLDINGS)
    assert set(r) == {"metrics", "factors", "optimizer", "backtest", "data"}
    assert {"hist_var", "gaussian_var", "sharpe", "annual_vol", "max_drawdown"} <= set(r["metrics"])
    assert set(r["factors"]) == {"betas", "alpha_annual", "r2", "variance_split"}
    assert {"current_sharpe", "max_sharpe_weights", "min_variance_weights", "top_sector"} <= set(r["optimizer"])
    assert set(r["backtest"]) == {"historical", "gaussian"}
    assert r["data"]["as_of"] == "2026-06-18" and r["data"]["window_status"] == "ok"
    assert r["data"]["crisis_coverage"]["COVID-19 Crash"] == []


def test_unresolvable_ticker_raises_with_reason_per_ticker():
    import pytest
    from risk_engine.analyze import PortfolioError
    with pytest.raises(PortfolioError) as e:
        analyze_portfolio({"AAPL": 1000, "XYZQQ": 500})
    assert e.value.problems[0]["ticker"] == "XYZQQ"
    assert "disabled" in e.value.problems[0]["reason"]     # tests run with live lookups off
```

Leave `test_asymmetric_weights_pin_ticker_ordering` unchanged: its "Technology" expectation still holds under the Yahoo taxonomy.

In `tests/test_schemas.py`, replace the `test_bad_holdings_rejected` parametrize list:

```python
@pytest.mark.parametrize("holdings", [
    {"AAPL": 20000},                          # fewer than 2
    {"AAPL": 20000, "MSFT": -5},              # non-positive amount
    {"AAPL": 20000, "AAPL/../X": 1},          # not a ticker format
    {"AAPL": 20000, "<script>": 1},           # not a ticker format
    {f"T{i}": 1 for i in range(51)},          # more than 50
])
def test_bad_holdings_rejected(holdings):
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=holdings)


def test_tickers_are_normalized_and_merged():
    req = AnalyzeRequest(holdings={"brk.b": 1000, "BRK-B": 500, "aapl": 10})
    assert req.holdings == {"BRK-B": 1500.0, "AAPL": 10.0}
```

In `tests/test_api.py`, add (TestClient with raw JSON, the transport path production uses):

```python
import json as _json

import pandas as pd
import pytest

from risk_engine import data


def _ok_live(monkeypatch):
    base = data.load_snapshot(use_memo=False).prices["AAPL"].dropna() * 0.5

    def fake(sym, **_):
        if sym == "XYZQQ":
            return data.LiveResult(sym, "not_found", "no such ticker on Yahoo Finance")
        if sym == "RELIANCE.NS":
            return data.LiveResult(sym, "non_usd", "priced in INR, not USD")
        meta = {"name": sym, "sector": "Technology", "type": "stock", "asset_class": "Equity",
                "in_sp500": False, "curated": False}
        return data.LiveResult(sym, "ok", "", base.rename(sym), meta)
    monkeypatch.setattr(data, "fetch_live", fake)


def _post(body):
    return client.post("/analyze", content=_json.dumps(body),
                       headers={"Content-Type": "application/json"})


def test_unknown_real_ticker_is_fetched_live(monkeypatch):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 10000, "ZZZZ": 5000}})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["as_of"] == "2026-06-18"


@pytest.mark.parametrize("bad,fragment", [
    ("XYZQQ", "no such ticker"), ("RELIANCE.NS", "INR"),
])
def test_unresolvable_ticker_is_422_with_per_ticker_reason(monkeypatch, bad, fragment):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 10000, bad: 5000}})
    assert r.status_code == 422
    problems = r.json()["detail"]["problems"]
    assert problems[0]["ticker"] == bad and fragment in problems[0]["reason"]


def test_more_than_five_live_lookups_is_422(monkeypatch):
    _ok_live(monkeypatch)
    r = _post({"holdings": {"AAPL": 1, **{f"ZZ{i}": 1 for i in range(6)}}})
    assert r.status_code == 422
    assert any("max 5" in p["reason"] for p in r.json()["detail"]["problems"])


@pytest.mark.parametrize("holdings", [
    {f"T{i}": 1 for i in range(51)}, {"AAPL": 1, "A/B": 1}, {"AAPL": 1, "": 1},
])
def test_shape_errors_are_422(holdings):
    assert _post({"holdings": holdings}).status_code == 422


def test_health_reports_data_freshness():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data_as_of"] == "2026-06-18"
    assert body["stale"] is True and body["data_source"] == "pinned"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_analyze.py tests/test_schemas.py tests/test_api.py -v`
Expected: FAIL (no `data` key; `PortfolioError` missing; 51 holdings accepted as unknown tickers; `/health` lacks `data_as_of`).

- [ ] **Step 3: Rewrite `risk_engine/analyze.py`**

```python
"""Pure portfolio-analysis orchestration for the API: resolve holdings against
the snapshot (bounded live lookups for anything outside it), choose the window
the holdings share, run the engine, return a JSON-serializable dict.
No UI, no transport."""
import numpy as np

from risk_engine import backtest, data, metrics, optimize, portfolio
from risk_engine import factors as fac
from risk_engine.config import BENCHMARK, LIVE_BUDGET_API_S, LIVE_MAX_API


class PortfolioError(ValueError):
    """The portfolio can't be analysed as given; `problems` says why, per ticker."""

    def __init__(self, problems):
        super().__init__("; ".join(f"{p['ticker']}: {p['reason']}" for p in problems))
        self.problems = problems


def analyze_portfolio(holdings, *, confidence=0.95, include_backtest=True,
                      snapshot=None, live_fetch=None):
    snap = snapshot or data.load_snapshot()
    res = portfolio.resolve_holdings(holdings, snap, live_fetch=live_fetch or data.fetch_live,
                                     max_live=LIVE_MAX_API, budget_s=LIVE_BUDGET_API_S)
    if res.rejected:
        raise PortfolioError([{"ticker": r.ticker, "reason": r.reason} for r in res.rejected])
    tickers = res.tickers
    win = portfolio.portfolio_window(res.prices, tickers, benchmark=snap.prices[BENCHMARK])
    if win.status == "too_short":
        raise PortfolioError([{"ticker": win.limiting, "reason": win.message}])

    lr = win.returns
    amounts = np.array([res.holdings[t] for t in tickers], dtype=float)
    weights = amounts / amounts.sum()
    pr = metrics.portfolio_returns(lr, tickers, weights)

    hist_var, hist_cvar = metrics.historical_var_cvar(pr, confidence)
    gauss_var, gauss_cvar = metrics.gaussian_var_cvar(pr, confidence)
    result = {
        "metrics": {
            "hist_var": hist_var,
            "hist_cvar": hist_cvar,
            "gaussian_var": gauss_var,
            "gaussian_cvar": gauss_cvar,
            "sharpe": metrics.sharpe_ratio(pr),
            "annual_vol": metrics.annualized_vol(pr),
            "max_drawdown": metrics.max_drawdown(pr),
        },
    }

    reg = fac.factor_regression(pr, snap.factors)
    result["factors"] = {
        "betas": reg["betas"],
        "alpha_annual": reg["alpha_annual"],
        "r2": reg["r2"],
        "variance_split": fac.variance_attribution(reg),
    }

    mu = optimize.annualized_mean(lr, tickers)
    cov = optimize.sample_cov(lr, tickers)
    w_ms = optimize.max_sharpe_weights(mu, cov)
    w_mv = optimize.min_variance_weights(cov)
    _, _, cur_sharpe = optimize.perf(weights, mu, cov)
    _, _, ms_sharpe = optimize.perf(w_ms, mu, cov)
    sector_pct = {}
    for t, w in zip(tickers, weights):
        s = res.meta[t]["sector"]
        sector_pct[s] = sector_pct.get(s, 0.0) + float(w)
    top_sector = max(sector_pct, key=sector_pct.get)
    result["optimizer"] = {
        "current_sharpe": float(cur_sharpe),
        "max_sharpe_value": float(ms_sharpe),
        "max_sharpe_weights": {t: float(w) for t, w in zip(tickers, w_ms)},
        "min_variance_weights": {t: float(w) for t, w in zip(tickers, w_mv)},
        "top_sector": top_sector,
        "top_sector_pct": float(sector_pct[top_sector]),
    }

    result["data"] = {
        "as_of": snap.as_of,
        "window_start": str(win.start.date()),
        "window_days": int(win.n_days),
        "window_status": win.status,
        "crisis_coverage": {c["label"]: c["missing"]
                            for c in portfolio.crisis_coverage(res.prices, tickers)},
    }

    if include_backtest:
        rows = backtest.backtest_var(pr, confidence)
        result["backtest"] = {
            r["method"]: {
                "breaches": int(r["breaches"]),
                "expected": float(r["expected"]),
                "kupiec_p": float(r["kupiec_p"]),
                "christoffersen_p": float(r["christoffersen_p"]),
                "passed": bool(r["passed"]),
            }
            for r in rows
        }

    return result
```

The `live_fetch or data.fetch_live` lookup happens at call time, so tests can monkeypatch `data.fetch_live`.

- [ ] **Step 4: Update `api/schemas.py`**

Replace the imports and `validate_holdings`:

```python
"""pydantic request/response models for the risk API. Validation is free credibility:
bad input becomes a clean 422 instead of a 500 deep in the engine."""
import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from risk_engine.config import MAX_HOLDINGS, MIN_HOLDINGS
from risk_engine.data import TICKER_RE, normalize_ticker

# A holding above this is nonsensical for a portfolio and only serves to probe
# for overflow; reject it rather than let a degenerate weight poison the math.
_MAX_AMOUNT = 1e12


def validate_holdings(v):
    if not (MIN_HOLDINGS <= len(v) <= MAX_HOLDINGS):
        raise ValueError(f"holdings must contain between {MIN_HOLDINGS} and {MAX_HOLDINGS} tickers")
    out = {}
    for ticker, amount in v.items():
        t = normalize_ticker(ticker)
        if not TICKER_RE.fullmatch(t):
            raise ValueError(f"not a valid ticker symbol: {ticker!r}")
        # inf/NaN pass a naive `> 0` check but divide into a NaN weight, which the
        # engine would silently return as a zeros analysis. Reject non-finite first.
        if not math.isfinite(amount):
            raise ValueError(f"amount for {t} must be a finite number")
        if amount <= 0:
            raise ValueError(f"amount for {t} must be > 0")
        if amount > _MAX_AMOUNT:
            raise ValueError(f"amount for {t} exceeds the maximum of {_MAX_AMOUNT:.0f}")
        out[t] = out.get(t, 0.0) + amount
    return out
```

Add a response model and field after `class Backtest`:

```python
class DataInfo(BaseModel):
    as_of: str
    window_start: str
    window_days: int
    window_status: str
    crisis_coverage: dict[str, list[str]]
```

and in `AnalyzeResponse` add `data: DataInfo` above `backtest`.

- [ ] **Step 5: Update `api/main.py`**

Add `from risk_engine import data` and `from risk_engine.analyze import PortfolioError, analyze_portfolio` (replacing the old analyze import). Replace `health`, `analyze` and `get_portfolio`:

```python
@app.get("/health")
def health():
    try:
        snap = data.load_snapshot()
    except data.SnapshotUnavailable as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "detail": str(e)})
    return {"status": "ok", "data_as_of": snap.as_of, "stale": data.is_stale(snap),
            "data_source": snap.source}


def _analyze_or_422(holdings, **kwargs):
    try:
        return analyze_portfolio(holdings, **kwargs)
    except PortfolioError as e:
        raise HTTPException(status_code=422, detail={"problems": e.problems})


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return _analyze_or_422(req.holdings, confidence=req.confidence,
                           include_backtest=req.include_backtest)


@app.get("/portfolios/{pid}")
def get_portfolio(pid: int, analyze: bool = Query(False)):
    row = store.get_portfolio(pid)
    if row is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    if analyze:
        row["analysis"] = _analyze_or_422(row["holdings"])
    return row
```

- [ ] **Step 6: Run the suite**

Run: `python -m pytest -q`
Expected: all PASS, including the August inf/NaN raw-JSON tests.

- [ ] **Step 7: Commit**

```bash
git add risk_engine/analyze.py api/schemas.py api/main.py tests/test_analyze.py tests/test_schemas.py tests/test_api.py
git commit -m "Serve any USD ticker through the API with per-ticker 422s and data freshness"
```

---

### Task 8: Snapshot build tool

**Files:**
- Create: `tools/build_snapshot.py`
- Create: `requirements-refresh.txt`
- Create: `tests/test_build_snapshot.py`
- Delete: `tools/snapshot_factors.py` (its fetch logic moves into `fetch_factors`)

**Interfaces:**
- Consumes: `config.BENCHMARK`, `config.CURATED_ETFS`, `config.GICS_TO_YAHOO`; `data.read_bundle`, `data.BUNDLE_FILES`.
- Produces in `tools/build_snapshot.py` (importable; tests add `tools/` to `sys.path`):
  - constants `START, SP500_MIN, SP500_MAX, OVERLAP_DAYS, ADJ_TOL, SPIKE, REVERSAL, STALE_ROWS, MAX_BAD_FRAC`
  - `parse_sp500(html: str) -> list[tuple[str, str, str]]` (ticker, name, Yahoo sector)
  - `build_universe(rows, previous: dict | None) -> tuple[dict, bool]` (universe, used_previous)
  - `fetch_yf(tickers: list[str], start: str) -> tuple[dict[str, pd.Series], dict[str, str]]`
  - `merge_incremental(stored: pd.DataFrame, fresh: dict[str, pd.Series], tol=ADJ_TOL) -> tuple[pd.DataFrame, list[str]]`
  - `quality_gate(prices: pd.DataFrame, spike=SPIKE, reversal=REVERSAL, stale_rows=STALE_ROWS) -> dict[str, str]`
  - `fetch_factors() -> pd.DataFrame`
  - `write_bundle(out_dir, prices, universe, factors, report) -> None`
  - `run(mode, prev_dir, out_dir, *, fetch=fetch_yf, get_html=None, get_factors=fetch_factors) -> dict` (the report; raises `SystemExit` when publishing must be blocked)
  - CLI: `python tools/build_snapshot.py --mode {incremental,full} --prev-dir DIR --out-dir DIR`

- [ ] **Step 1: Write the failing tests**

`tests/test_build_snapshot.py`:

```python
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
import build_snapshot as bs  # noqa: E402

WIKI = """<table id="constituents"><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
<tr><td>AAPL</td><td>Apple Inc.</td><td>Information Technology</td></tr>
<tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td></tr></table>"""


def series(values, start="2026-01-02"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_parse_sp500_maps_gics_to_yahoo_and_dots_to_dashes():
    assert bs.parse_sp500(WIKI) == [("AAPL", "Apple Inc.", "Technology"),
                                    ("BRK-B", "Berkshire Hathaway", "Financial Services")]


def test_universe_keeps_previous_list_when_table_looks_wrong():
    rows = [(f"T{i}", "x", "Technology") for i in range(400)]
    prev = {"OLD": {"name": "old"}}
    uni, used_prev = bs.build_universe(rows, prev)
    assert used_prev and uni == prev
    with pytest.raises(RuntimeError, match="400 rows"):
        bs.build_universe(rows, None)


def test_universe_adds_curated_etfs():
    rows = [(f"T{i}", "x", "Technology") for i in range(500)]
    uni, used_prev = bs.build_universe(rows, None)
    assert not used_prev
    assert uni["SPY"]["curated"] and uni["TLT"]["asset_class"] == "Bond"
    assert uni["T0"]["in_sp500"] and len(uni) == 500 + len(bs.CURATED_ETFS)


def test_incremental_appends_new_closes():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    fresh = {"A": series([12, 13], start="2026-01-06")}
    merged, repull = bs.merge_incremental(stored, fresh)
    assert repull == [] and list(merged["A"]) == [10, 11, 12, 13]


def test_rescaled_history_triggers_full_repull():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    fresh = {"A": series([11.88, 13], start="2026-01-06")}   # a dividend re-scaled history by 1%
    merged, repull = bs.merge_incremental(stored, fresh)
    assert repull == ["A"] and list(merged["A"]) == [10, 11, 12]


def test_no_overlap_also_triggers_repull():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    _, repull = bs.merge_incremental(stored, {"A": series([20, 21], start="2026-02-02")})
    assert repull == ["A"]


def test_failed_fetch_keeps_stored_history():
    stored = pd.DataFrame({"A": series([10, 11, 12])})
    merged, _ = bs.merge_incremental(stored, {})
    assert list(merged["A"]) == [10, 11, 12]


def test_bad_print_that_reverses_is_quarantined_but_real_crash_is_not():
    prices = pd.DataFrame({
        "BADTICK": series([100, 101, 300, 102, 103, 104, 105]),
        "CRASH": series([100, 101, 50, 49, 48, 50, 51]),
        "OK": series([100, 101, 102, 103, 104, 105, 106]),
    })
    q = bs.quality_gate(prices)
    assert set(q) == {"BADTICK"} and "bad print" in q["BADTICK"]


def test_stale_ticker_is_quarantined():
    vals = [100.0] * 10
    prices = pd.DataFrame({"LIVE": series(vals), "DEAD": series(vals[:4] + [np.nan] * 6)})
    q = bs.quality_gate(prices)
    assert set(q) == {"DEAD"} and "no new close" in q["DEAD"]


def _fake_world(n_good=100, n_fail=0, benchmark_fails=False):
    rows = [(f"T{i}", "x", "Technology") for i in range(n_good + n_fail)]
    html = ("<table id='constituents'><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>"
            + "".join(f"<tr><td>{t}</td><td>x</td><td>Information Technology</td></tr>"
                      for t, _, _ in rows) + "</table>")
    failing = {f"T{i}" for i in range(n_good, n_good + n_fail)}
    if benchmark_fails:
        failing.add("SPY")

    def fetch(tickers, start):
        idx = pd.bdate_range("2018-01-02", "2026-06-18")
        got = {t: pd.Series(np.linspace(50, 150, len(idx)), index=idx, name=t)
               for t in tickers if t not in failing}
        return got, {t: "no data returned" for t in tickers if t in failing}

    factors = pd.DataFrame({"Mkt-RF": [0.0], "SMB": [0.0], "HML": [0.0], "RF": [0.0]},
                           index=pd.to_datetime(["2026-05-29"]))
    return html, fetch, factors


@pytest.fixture
def small_bounds(monkeypatch):
    monkeypatch.setattr(bs, "SP500_MIN", 1)
    monkeypatch.setattr(bs, "SP500_MAX", 10_000)


def test_full_run_writes_a_readable_bundle(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    report = bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["as_of"] == "2026-06-18" and report["failed"] == {}
    from risk_engine import data
    snap = data.read_bundle(str(tmp_path / "out"), "release")
    assert "T0" in snap.prices.columns and "SPY" in snap.universe


def test_too_many_failures_blocks_publishing(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(n_good=100, n_fail=5)     # ~4% of ~132 tickers
    with pytest.raises(SystemExit, match="not publishing"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)


def test_benchmark_failure_always_blocks_publishing(tmp_path, small_bounds):
    html, fetch, factors = _fake_world(benchmark_fails=True)
    with pytest.raises(SystemExit, match="benchmark"):
        bs.run("full", str(tmp_path / "none"), str(tmp_path / "out"),
               fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)


def test_incremental_without_previous_bundle_runs_full(tmp_path, small_bounds):
    html, fetch, factors = _fake_world()
    report = bs.run("incremental", str(tmp_path / "none"), str(tmp_path / "out"),
                    fetch=fetch, get_html=lambda: html, get_factors=lambda: factors)
    assert report["mode"] == "full"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_build_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_snapshot'`.

- [ ] **Step 3: Implement**

`requirements-refresh.txt` (pin yfinance to the version Task 1 recorded):

```
numpy==2.2.4
pandas==2.2.3
lxml
yfinance==<version from Task 1>
```

`tools/build_snapshot.py`:

```python
"""Build the data bundle the app and API read: S&P 500 + curated ETFs, adjusted
daily closes since 2018, Fama-French factors, and a report naming every ticker
that failed or was held back. Run by .github/workflows/refresh-data.yml; also
runnable by hand:

    python tools/build_snapshot.py --mode full --prev-dir prev --out-dir out
"""
import argparse
import datetime as dt
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from risk_engine import data  # noqa: E402
from risk_engine.config import BENCHMARK, CURATED_ETFS, GICS_TO_YAHOO  # noqa: E402

START = "2018-01-01"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_Factors_daily_CSV.zip")
UA = {"User-Agent": "MarketPlug data refresh (github.com/metro-dev26/-portfolio-risk-dashboard)"}

SP500_MIN, SP500_MAX = 490, 510
OVERLAP_DAYS = 10
ADJ_TOL = 1e-4          # relative gap on overlapping closes that means history was re-scaled
SPIKE, REVERSAL = 0.40, 0.5
STALE_ROWS = 5
MAX_BAD_FRAC = 0.02


def _http(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_sp500(html):
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})[0]
    return [(str(r["Symbol"]).strip().replace(".", "-"), str(r["Security"]).strip(),
             GICS_TO_YAHOO.get(str(r["GICS Sector"]).strip(), "Unknown"))
            for _, r in table.iterrows()]


def build_universe(rows, previous):
    """A table far from 500 rows means the Wikipedia page changed shape, not that
    the index did; keep yesterday's list rather than publish a broken one."""
    if not (SP500_MIN <= len(rows) <= SP500_MAX):
        if previous is None:
            raise RuntimeError(f"S&P 500 table has {len(rows)} rows and there is no "
                               f"previous universe to fall back on")
        return previous, True
    universe = {t: {"name": n, "sector": s, "type": "stock", "asset_class": "Equity",
                    "in_sp500": True, "curated": False} for t, n, s in rows}
    for t, (name, sector, asset_class) in CURATED_ETFS.items():
        universe[t] = {"name": name, "sector": sector, "type": "etf",
                       "asset_class": asset_class, "in_sp500": False, "curated": True}
    return universe, False


def _naive_daily(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.Series(s.to_numpy(dtype=float), index=idx.normalize(), name=s.name)


def _fetch_urllib(t, start):
    p1 = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    payload = json.loads(_http("https://query1.finance.yahoo.com/v8/finance/chart/"
                               f"{t}?period1={p1}&period2={p2}&interval=1d", timeout=15))
    r = payload["chart"]["result"][0]
    days = [dt.datetime.fromtimestamp(x, dt.timezone.utc).date() for x in r["timestamp"]]
    closes = r["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(closes, index=pd.to_datetime(days), name=t, dtype=float).dropna()


def fetch_yf(tickers, start):
    """yfinance in one batch (it handles Yahoo's checks on cloud hosts); any ticker
    it misses gets one retry over plain urllib. -> (series by ticker, reason by ticker)."""
    import yfinance as yf
    got, failed = {}, {}
    df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    closes = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    for t in tickers:
        s = closes[t].dropna() if t in closes.columns else pd.Series(dtype=float)
        if len(s):
            got[t] = _naive_daily(s.rename(t))
            continue
        try:
            s = _fetch_urllib(t, start)
            if len(s):
                got[t] = s
            else:
                failed[t] = "no data returned"
        except Exception as e:  # recorded per ticker in the report, never swallowed
            failed[t] = f"{type(e).__name__}: {e}"
    return got, failed


def merge_incremental(stored, fresh, tol=ADJ_TOL):
    """Extend stored history with fresh closes. Returns (merged, repull): repull
    names tickers whose overlapping closes disagree (or don't overlap at all),
    meaning a dividend or split re-scaled the history; those need a full fetch."""
    cols, repull = {}, []
    for t in sorted(set(stored.columns) | set(fresh)):
        old = stored[t].dropna() if t in stored.columns else pd.Series(dtype=float)
        new = fresh.get(t)
        if new is None or old.empty:
            cols[t] = old if new is None else new
            continue
        overlap = old.index.intersection(new.index)
        if len(overlap) == 0 or (np.abs(new[overlap] / old[overlap] - 1) > tol).any():
            repull.append(t)
            cols[t] = old
            continue
        cols[t] = pd.concat([old, new[new.index > old.index.max()]])
    return pd.DataFrame(cols).sort_index(), repull


def quality_gate(prices, spike=SPIKE, reversal=REVERSAL, stale_rows=STALE_ROWS):
    """-> {ticker: reason} to hold back. A bad print jumps more than `spike` and the
    next close undoes at least `reversal` of the jump; a real crash doesn't bounce
    back overnight. Stale = no close in the table's last `stale_rows` rows."""
    held = {}
    recent = prices.index[-stale_rows:] if len(prices) >= stale_rows else prices.index
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            held[t] = "no prices"
            continue
        if s.index.max() < recent[0]:
            held[t] = f"no new close since {s.index.max().date()}"
            continue
        prev, nxt = s.shift(1), s.shift(-1)
        jump = s / prev - 1
        undone = (s - nxt) / (s - prev)
        bad = (jump.abs() > spike) & (undone >= reversal)
        if bad.any():
            held[t] = f"suspected bad print on {bad[bad].index[0].date()}"
    return held


def fetch_factors():
    z = zipfile.ZipFile(io.BytesIO(_http(FF_URL)))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hdr = next(i for i, line in enumerate(lines) if line.strip().startswith(",Mkt-RF"))
    rows = []
    for line in lines[hdr + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit() and parts[0] >= "20180101":
            rows.append([parts[0]] + [float(x) / 100.0 for x in parts[1:]])   # percent -> decimal
    df = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date").sort_index()


def write_bundle(out_dir, prices, universe, factors, report):
    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(os.path.join(out_dir, "prices.csv.gz"), compression="gzip", float_format="%.6f")
    factors.to_csv(os.path.join(out_dir, "factors.csv"))
    with open(os.path.join(out_dir, "universe.json"), "w") as f:
        json.dump(universe, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "refresh_report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)


def _read_prev(prev_dir):
    try:
        return data.read_bundle(prev_dir, "previous")
    except data.SnapshotUnavailable:
        return None


def run(mode, prev_dir, out_dir, *, fetch=fetch_yf, get_html=None, get_factors=fetch_factors):
    get_html = get_html or (lambda: _http(SP500_URL).decode("utf-8"))
    prev = _read_prev(prev_dir)
    if prev is None:
        mode = "full"
    universe, used_prev = build_universe(parse_sp500(get_html()), prev.universe if prev else None)
    tickers = sorted(universe)

    repulled = []
    if mode == "full":
        got, failed = fetch(tickers, START)
        prices = pd.DataFrame(got).sort_index()
    else:
        stored = prev.prices[[t for t in prev.prices.columns if t in universe]]
        since = (stored.index.max() - pd.offsets.BDay(OVERLAP_DAYS)).strftime("%Y-%m-%d")
        got, failed = fetch([t for t in tickers if t in stored.columns], since)
        prices, repulled = merge_incremental(stored, got)
        need_full = repulled + [t for t in tickers if t not in stored.columns]
        if need_full:
            got_full, failed_full = fetch(need_full, START)
            prices = pd.concat([prices.drop(columns=[t for t in got_full if t in prices.columns]),
                                pd.DataFrame(got_full)], axis=1).sort_index()
            failed.update(failed_full)

    if BENCHMARK in failed or BENCHMARK not in prices.columns:
        raise SystemExit(f"benchmark {BENCHMARK} failed to download — not publishing")
    held = {t: r for t, r in quality_gate(prices).items() if t not in failed}
    if BENCHMARK in held:
        raise SystemExit(f"benchmark {BENCHMARK} held back ({held[BENCHMARK]}) — not publishing")
    prices = prices.drop(columns=[t for t in held if t in prices.columns])
    factors = get_factors() if (mode == "full" or prev is None) else prev.factors

    report = {
        "as_of": str(prices[BENCHMARK].dropna().index.max().date()),
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "n_tickers": len(tickers),
        "failed": failed,
        "quarantined": held,
        "repulled": repulled,
        "universe_fallback": used_prev,
        "factors_through": str(factors.index.max().date()),
    }
    write_bundle(out_dir, prices, universe, factors, report)
    bad = len(set(failed) | set(held))
    print(json.dumps({k: report[k] for k in ("as_of", "mode", "n_tickers")}
                     | {"failed": len(failed), "held_back": len(held), "repulled": len(repulled)}))
    if bad / len(tickers) > MAX_BAD_FRAC:
        raise SystemExit(f"{bad}/{len(tickers)} tickers failed or held back "
                         f"({bad / len(tickers):.1%} > {MAX_BAD_FRAC:.0%}) — not publishing")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["incremental", "full"], default="incremental")
    ap.add_argument("--prev-dir", default="prev")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()
    run(args.mode, args.prev_dir, args.out_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_build_snapshot.py -v`
Expected: all PASS. `lxml` is installed locally; CI gets it in Step 5.

- [ ] **Step 5: Give CI the parser dependency**

`pd.read_html` needs lxml in CI too. Add `lxml` to `requirements-dev.txt`.

- [ ] **Step 6: Remove the superseded factors tool and run the suite**

```bash
git rm tools/snapshot_factors.py
python -m pytest -q
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tools/build_snapshot.py requirements-refresh.txt requirements-dev.txt tests/test_build_snapshot.py
git commit -m "Add the daily data build: incremental fetch, quality gate, publish threshold"
```

---

### Task 9: Refresh workflow + release pruning

**Files:**
- Create: `.github/workflows/refresh-data.yml`
- Create: `tools/prune_releases.py`
- Create: `tests/test_prune_releases.py`

**Interfaces:**
- Consumes: `tools/build_snapshot.py` CLI (Task 8).
- Produces: release `data-latest` (assets `prices.csv.gz`, `universe.json`, `factors.csv`, `refresh_report.json`), dated releases `data-YYYY-MM-DD`; `tools/prune_releases.py` with `tags_to_prune(tags: list[str], today: dt.date, keep_days: int = 14) -> list[str]`.

- [ ] **Step 1: Write the failing test**

`tests/test_prune_releases.py`:

```python
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "tools"))
from prune_releases import tags_to_prune  # noqa: E402


def test_only_dated_data_releases_older_than_the_window_are_pruned():
    tags = ["data-2026-09-01", "data-2026-09-20", "data-latest", "v1.0", "data-garbage"]
    assert tags_to_prune(tags, dt.date(2026, 10, 3), keep_days=14) == ["data-2026-09-01"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_prune_releases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prune_releases'`.

- [ ] **Step 3: Implement the pruner**

`tools/prune_releases.py`:

```python
"""Delete dated data releases (data-YYYY-MM-DD) older than the retention window.
data-latest and any non-data release are never touched. Needs the gh CLI."""
import argparse
import datetime as dt
import json
import re
import subprocess

_DATED = re.compile(r"data-(\d{4}-\d{2}-\d{2})")


def tags_to_prune(tags, today, keep_days=14):
    old = []
    for tag in tags:
        m = _DATED.fullmatch(tag)
        if m and (today - dt.date.fromisoformat(m.group(1))).days > keep_days:
            old.append(tag)
    return old


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-days", type=int, default=14)
    args = ap.parse_args()
    listing = subprocess.run(["gh", "release", "list", "--limit", "200", "--json", "tagName"],
                             check=True, capture_output=True, text=True).stdout
    tags = [r["tagName"] for r in json.loads(listing)]
    for tag in tags_to_prune(tags, dt.date.today(), args.keep_days):
        subprocess.run(["gh", "release", "delete", tag, "-y", "--cleanup-tag"], check=True)
        print(f"deleted {tag}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/test_prune_releases.py -v`
Expected: PASS.

- [ ] **Step 5: Write the workflow**

`.github/workflows/refresh-data.yml`:

```yaml
name: refresh-data
on:
  schedule:
    - cron: "0 23 * * 1-5"   # weekdays after the US close: incremental
    - cron: "0 23 * * 0"     # Sunday: full rebuild + Fama-French factors
  workflow_dispatch:
    inputs:
      mode:
        description: "incremental or full"
        required: true
        default: "full"
permissions:
  contents: write   # create/upload releases
  actions: write    # re-enable this workflow (beats the 60-day inactivity disable)
concurrency:
  group: refresh-data
  cancel-in-progress: false
jobs:
  refresh:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      GH_TOKEN: ${{ github.token }}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with:
          python-version: "3.11"
      - run: pip install -r requirements-refresh.txt
      - name: Choose mode
        id: mode
        run: |
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            echo "mode=${{ inputs.mode }}" >> "$GITHUB_OUTPUT"
          elif [ "$(date -u +%u)" = "7" ]; then
            echo "mode=full" >> "$GITHUB_OUTPUT"
          else
            echo "mode=incremental" >> "$GITHUB_OUTPUT"
          fi
      - name: Download previous bundle
        run: gh release download data-latest -D prev || echo "no previous bundle; building in full"
      - name: Build
        run: python tools/build_snapshot.py --mode "${{ steps.mode.outputs.mode }}" --prev-dir prev --out-dir out
      - name: Publish dated release and move data-latest
        run: |
          AS_OF=$(python -c "import json; print(json.load(open('out/refresh_report.json'))['as_of'])")
          TAG="data-${AS_OF}"
          gh release view "$TAG" >/dev/null 2>&1 || \
            gh release create "$TAG" --title "Data ${AS_OF}" --notes "Automated data bundle" --prerelease
          gh release upload "$TAG" out/prices.csv.gz out/universe.json out/factors.csv out/refresh_report.json --clobber
          gh release view data-latest >/dev/null 2>&1 || \
            gh release create data-latest --title "Data (latest)" --notes "Always the newest bundle" --prerelease
          # Data first, report last: a reader who sees the new report also gets the new data.
          gh release upload data-latest out/prices.csv.gz out/universe.json out/factors.csv --clobber
          gh release upload data-latest out/refresh_report.json --clobber
      - name: Prune releases older than 14 days
        run: python tools/prune_releases.py --keep-days 14
      - name: Keep the schedule alive
        run: gh api -X PUT "repos/${{ github.repository }}/actions/workflows/refresh-data.yml/enable"
```

- [ ] **Step 6: Lint the workflow locally**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/refresh-data.yml')); print('ok')"`
Expected: `ok`. (PyYAML ships with Streamlit's dependency tree; if it's missing, `pip install pyyaml` for this check only.)

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/refresh-data.yml tools/prune_releases.py tests/test_prune_releases.py
git commit -m "Refresh the data bundle daily and keep 14 days of dated releases"
```

---

### Task 10: App — import, "Not included", window, freshness

**Files:**
- Modify: `app.py:1-35` (docstring, imports), `app.py:274-372` (universe/data/sidebar/holdings/engine start), `app.py:374-378` (hero eyebrow), factor-section caption near `app.py:846`, footer `app.py:1137-1139`
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `data.load_snapshot`, `data.SnapshotUnavailable`, `data.fetch_live`, `data.is_stale`, `data.freshness_line`; `importer.parse_paste`, `parse_csv`, `from_rows`, `encode_share`, `decode_share`; `portfolio.resolve_holdings`, `portfolio.portfolio_window`; `config.BENCHMARK`, `MIN_HOLDINGS`, `MAX_HOLDINGS`, `LIVE_MAX_UI`, `LIVE_BUDGET_UI_S`.
- Produces (module-level names the rest of `app.py` already uses): `selected: list[str]`, `weights: np.ndarray`, `port_val: float`, `lr: pd.DataFrame` (window returns incl. `SPY`), `prices: pd.DataFrame` (holdings' prices with gaps), `SECTOR: dict`, `ASSET_CLASS: dict`; session keys `holdings`, `is_example`, `editor_v`, `import_notes`; query param `p`.

- [ ] **Step 1: Write the failing app tests**

Append to `tests/test_app_smoke.py`:

```python
def _sidebar_text(at):
    return " ".join(str(m.value) for m in at.sidebar.markdown) + " ".join(
        str(c.value) for c in at.sidebar.caption)


def test_example_portfolio_and_freshness_line_show_on_first_visit():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.session_state["is_example"] is True
    assert "example portfolio" in _sidebar_text(at)
    assert "dollars" in _sidebar_text(at)                 # amounts are $, not share counts
    body = " ".join(str(m.value) for m in at.markdown)
    assert "Prices as of 2026-06-18 close" in body


def test_stale_data_shows_a_banner():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert any("refresh looks behind" in str(w.value) for w in at.warning)


def test_paste_with_a_bad_ticker_lists_it_as_not_included():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nMSFT 5000\nXYZQQ 3000").run()
    at.button(key="load_paste").click().run()
    assert not at.exception
    text = _sidebar_text(at)
    assert "XYZQQ" in text and "$3,000 excluded" in text
    assert at.session_state["holdings"] == {"AAPL": 10000.0, "MSFT": 5000.0, "XYZQQ": 3000.0}


def test_pasted_html_is_escaped():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 10000\nMSFT 5000\n<img src=x onerror=alert(1)> 100").run()
    at.button(key="load_paste").click().run()
    text = _sidebar_text(at)
    assert "&lt;img" in text and "<img" not in text


def test_zero_amounts_never_become_equal_weights():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="paste_box").input("AAPL 0\nMSFT 0").run()
    at.button(key="load_paste").click().run()
    assert not at.exception
    assert any("at least 2 holdings" in str(w.value) for w in at.warning)


def test_shared_link_restores_the_portfolio():
    at = AppTest.from_file(APP, default_timeout=60)
    at.query_params["p"] = "AAPL:50000,TLT:50000"
    at.run()
    assert not at.exception
    assert at.session_state["holdings"] == {"AAPL": 50000.0, "TLT": 50000.0}
    assert at.session_state["is_example"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: the new tests FAIL (no `paste_box`, no `is_example`); the original three still pass.

- [ ] **Step 3: Update imports and the module docstring**

In `app.py`, change the docstring's last two lines from "Every number on this page is computed live from real market data.\nNothing is hardcoded." to:

```
Every number on this page is computed from end-of-day market data refreshed
each US trading day. Nothing is hardcoded.
```

Add `import html` on the line after `import warnings`. Replace `from risk_engine.data import load_prices` with:

```python
from risk_engine import data, importer, portfolio
from risk_engine.config import (BENCHMARK, CRISES, LIVE_BUDGET_UI_S, LIVE_MAX_UI,
                                MAX_HOLDINGS, MIN_HOLDINGS)
```

- [ ] **Step 4: Replace `app.py:274-372`**

Delete from the `# ── UNIVERSE (US large-caps + bond ETFs, priced in USD) ───────` header through the line `max_dd = metrics.max_drawdown(pr)` and its preceding `# ── RISK ENGINE` header, and insert:

```python
# ── DATA ──────────────────────────────────────────────────────
try:
    snap = data.load_snapshot()
except data.SnapshotUnavailable as e:
    st.error(f"Market data is unavailable right now ({e}). Please try again later.")
    st.stop()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _live(sym):
    return data.fetch_live(sym)


def _last_price(sym):
    if sym in snap.prices.columns:
        s = snap.prices[sym].dropna()
        return float(s.iloc[-1]) if len(s) else None
    r = _live(sym)
    return float(r.prices.iloc[-1]) if r.status == "ok" else None


EXAMPLE = {"AAPL": 20000.0, "MSFT": 20000.0, "JPM": 20000.0, "XOM": 20000.0, "TLT": 20000.0}

if "holdings" not in st.session_state:
    _shared = importer.decode_share(st.query_params.get("p", ""))
    st.session_state.holdings = _shared or dict(EXAMPLE)
    st.session_state.is_example = not _shared
    st.session_state.editor_v = 0
    st.session_state.import_notes = []


def _replace_holdings(parsed):
    st.session_state.holdings = parsed.holdings
    st.session_state.is_example = False
    st.session_state.editor_v += 1          # a fresh key makes the table show the new rows
    st.session_state.import_notes = parsed.notes


# ── SIDEBAR: build the portfolio ──────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:DM Mono;font-size:9px;letter-spacing:0.2em;
                color:#4d9fff;text-transform:uppercase;margin-bottom:4px;'>Portfolio Risk Dashboard</div>
    <div style='font-family:Syne;font-size:22px;font-weight:800;margin-bottom:20px;'>Build Your Portfolio</div>
    """, unsafe_allow_html=True)

    tab_paste, tab_csv, tab_search = st.tabs(["Paste", "Upload CSV", "Search"])
    with tab_paste:
        _txt = st.text_area("One holding per line: ticker, then dollars",
                            placeholder="AAPL 10000\nVOO, $25,000\nmsft 8,500", key="paste_box")
        if st.button("Load pasted holdings", key="load_paste"):
            _replace_holdings(importer.parse_paste(_txt))
    with tab_csv:
        _up = st.file_uploader("Broker export (.csv)", type="csv", key="csv_up")
        if _up is not None and st.button("Load CSV", key="load_csv"):
            _replace_holdings(importer.parse_csv(
                _up.getvalue().decode("utf-8-sig", errors="replace"), last_price=_last_price))
    with tab_search:
        _pick = st.selectbox("Find a ticker", sorted(snap.universe), index=None,
                             format_func=lambda t: f"{t} — {snap.universe[t]['name']}",
                             key="search_pick")
        _other = st.text_input("…or type any US-listed ticker", key="search_other")
        _amt = st.number_input("Amount ($)", min_value=0.0, value=10000.0, step=1000.0,
                               key="search_amt")
        if st.button("Add holding", key="add_holding"):
            _sym = (_other or _pick or "").strip()
            if _sym:
                _base = {} if st.session_state.is_example else dict(st.session_state.holdings)
                _replace_holdings(importer.from_rows(list(_base.items()) + [(_sym, _amt)]))

    if st.session_state.is_example:
        st.caption("Showing an example portfolio — paste, upload or search to use your own.")
    st.caption("Amounts are US dollars (market value), not share counts.")
    _h = st.session_state.holdings
    _edited = st.data_editor(
        pd.DataFrame({"Ticker": list(_h), "Amount $": list(_h.values())}),
        num_rows="dynamic", hide_index=True, width="stretch",
        key=f"amounts_{st.session_state.editor_v}",
        column_config={"Amount $": st.column_config.NumberColumn(
            min_value=0.0, step=1000.0, format="$%d")},
    )

    conf = st.select_slider(
        "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

_parsed = importer.from_rows(list(zip(_edited["Ticker"], _edited["Amount $"])))
_notes = st.session_state.import_notes + _parsed.notes
if len(_parsed.holdings) > MAX_HOLDINGS:
    st.error(f"This tool analyses up to {MAX_HOLDINGS} holdings; the table has "
             f"{len(_parsed.holdings)}. Combine or remove some.")
    st.stop()

with st.spinner("Looking up tickers outside the dataset..."):
    res = portfolio.resolve_holdings(_parsed.holdings, snap, live_fetch=_live,
                                     max_live=LIVE_MAX_UI, budget_s=LIVE_BUDGET_UI_S)

if res.rejected or _notes:
    _items = "".join(f"<li><b>{html.escape(r.ticker)}</b> — {html.escape(r.reason)} · "
                     f"${r.amount:,.0f} excluded</li>" for r in res.rejected)
    _items += "".join(f"<li>{html.escape(n)}</li>" for n in _notes)
    st.sidebar.markdown(f"""<div class='insight warn'><div class='insight-text'>
        <strong>Not included</strong><ul>{_items}</ul></div></div>""", unsafe_allow_html=True)

if snap.warning:
    st.warning(snap.warning)
if data.is_stale(snap):
    st.warning(f"These prices are from {snap.as_of} — the daily refresh looks behind, so the "
               f"numbers below are not current.")

if len(res.tickers) < MIN_HOLDINGS:
    st.warning("Add at least 2 holdings with an amount above $0 to build a portfolio.")
    st.stop()

selected = res.tickers
amounts = np.array([res.holdings[t] for t in selected], dtype=float)
port_val = float(amounts.sum())
weights = amounts / port_val
SECTOR = {t: res.meta[t]["sector"] for t in selected}
ASSET_CLASS = {t: res.meta[t]["asset_class"] for t in selected}

win = portfolio.portfolio_window(res.prices, selected, benchmark=snap.prices[BENCHMARK])
if win.status == "too_short":
    st.error(win.message)
    st.stop()
if win.status == "short":
    st.warning(win.message)
lr = win.returns
prices = res.prices
st.query_params["p"] = importer.encode_share(res.holdings)

# ── RISK ENGINE ───────────────────────────────────────────────
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

Holdings are always positive after `importer.from_rows`, so the old all-zero → equal-weights branch has nothing to catch.

- [ ] **Step 5: Hero, factor caption, footer**

Hero eyebrow (formerly `{source} · {len(ret_sel):,} trading days · end-of-day, refreshed hourly`):

```python
    <div class='hero-eyebrow'>{html.escape(data.freshness_line(snap))} · {len(ret_sel):,} trading days analysed</div>
```

In the factor section, directly after the existing `st.caption(...)` that explains the three factors, add:

```python
st.caption(f"Factor data through {fac.load_factors().index.max().date()} — Ken French "
           f"publishes monthly, so it trails prices by about a month.")
```

In the footer, replace `(3) Data is end-of-day, refreshed hourly — not live intraday.` with `(3) Prices are end-of-day closes, refreshed once each US trading day — not live intraday.`

- [ ] **Step 6: Run the app tests and the suite**

Run: `python -m pytest tests/test_app_smoke.py -v` then `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 7: Look at it**

Run: `python -m streamlit run app.py --server.port 8801`. With Playwright, screenshot the sidebar and hero. Check: the three tabs render, the example caption shows, the "Not included" box appears after pasting `XYZQQ 100` (with live lookups on locally it should say "no such ticker on Yahoo Finance"), the freshness line sits in the hero, and the dark theme holds for the new widgets. Fix any visual breakage before committing.

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Let users paste, upload or search their real portfolio, and show what was left out"
```

---

### Task 11: App — stress-test coverage + tariff shock

**Files:**
- Modify: `app.py` stress section (from `# ── STRESS TESTING: HISTORICAL CRISES` through the `else: st.info(...)` that closes it, formerly `app.py:600-672`)
- Modify: `tests/test_app_smoke.py`

**Interfaces:**
- Consumes: `config.CRISES` (imported in Task 10), `portfolio.crisis_coverage`, module-level `prices`, `selected`, `weights`, `port_val`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_app_smoke.py`:

```python
def test_stress_section_includes_tariff_shock():
    at = AppTest.from_file(APP, default_timeout=60).run()
    body = " ".join(str(m.value) for m in at.markdown)
    assert "2025 Tariff Shock" in body
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_app_smoke.py::test_stress_section_includes_tariff_shock -v`
Expected: FAIL (the local `CRISES` list in `app.py` has three entries).

- [ ] **Step 3: Rewrite the crisis selection**

Delete the local `CRISES = [...]` block in the stress section. Replace the two `results = ...` lines with:

```python
_coverage = portfolio.crisis_coverage(prices, selected, CRISES)
results = [(c["label"], c["context"], stress(c["start"], c["end"]))
           for c in _coverage if not c["missing"]]
results = [(label, ctx, r) for label, ctx, r in results if r is not None]
_uncovered = [c for c in _coverage if c["missing"]]
```

Change the plot palette line to `palette = ["#ffb347", "#ff3d5a", "#4d9fff", "#b07cff"]`.

Replace the closing `else: st.info(...)` with:

```python
else:
    st.info("None of the crisis windows is fully covered by these holdings' price history.")

if _uncovered:
    st.caption("Not in this stress test: " + "; ".join(
        f"{c['label']} — {', '.join(html.escape(t) for t in c['missing'])} not listed yet"
        for c in _uncovered))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_app_smoke.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_smoke.py
git commit -m "Add the 2025 tariff shock and label crises a holding was not listed for"
```

---

### Task 12: Remove the legacy universe and document the data pipeline

**Files:**
- Modify: `risk_engine/config.py` (delete the legacy block)
- Modify: `risk_engine/data.py` (delete `load_prices`)
- Modify: `tests/test_config.py`, `tests/test_data.py`
- Delete: `prices.csv`, `factors.csv` (repo root)
- Modify: `tools/snapshot_reference.py:15` (read the fixture)
- Modify: `README.md`

**Interfaces:** Removes `config.TICKERS/SECTOR/BONDS/ASSET_CLASS` and `data.load_prices`. Nothing may still import them.

- [ ] **Step 1: Confirm nothing still uses the legacy names**

Run: `git grep -n "load_prices\|config import TICKERS\|config import SECTOR\|ASSET_CLASS\[\|from risk_engine.config import.*BONDS" -- ':!docs'`
Expected: hits only in `risk_engine/config.py`, `risk_engine/data.py`, `tests/test_config.py`, `tests/test_data.py`. `app.py` defines its own `SECTOR`/`ASSET_CLASS` dicts from `res.meta` (Task 10), which is fine. Any other hit is a missed caller: fix it before continuing.

- [ ] **Step 2: Delete the legacy code and tests**

- In `risk_engine/config.py`, delete from `# Legacy 24-ticker universe; …` to the end of the file.
- In `risk_engine/data.py`, delete `load_prices`.
- In `tests/test_config.py`, delete `test_tickers_are_unique_and_nonempty`, `test_bonds_are_subset_of_tickers` and `test_every_ticker_has_a_sector_and_asset_class`.
- `git rm tests/test_data.py prices.csv factors.csv` (`test_data.py`'s coverage now lives in `tests/test_snapshot.py`).
- In `tools/snapshot_reference.py`, change the read to `pd.read_csv("tests/fixtures/snapshot/prices.csv.gz", index_col=0, parse_dates=True).ffill().dropna()` and update its docstring's run note accordingly. Do **not** run it: the golden file stays frozen.

- [ ] **Step 3: Run the suite**

Run: `python -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: README**

Add a `## Data` section after Features (impersonal voice, no "I"/"you"):

```markdown
## Data

- **Universe:** every S&P 500 constituent plus 27 curated ETFs (broad market, the 11 sector
  SPDRs, Treasury/aggregate/corporate bonds, international, gold). Any other USD-listed ticker
  is looked up live from Yahoo Finance when a portfolio includes it.
- **Refresh:** a GitHub Action rebuilds the data bundle after each US trading day (full
  rebuild and Fama-French factor refresh on Sundays) and publishes it as the `data-latest`
  release. Tickers with suspected bad prints or no recent closes are held back and named;
  if more than 2% of tickers fail, nothing is published and the previous day's data stays up.
- **Freshness is always shown:** the dashboard header states the data date and how many
  tickers are current, and a banner appears when the data is more than three business days old.
- **Portfolio input:** paste `TICKER AMOUNT` lines, upload a broker CSV (Symbol/Ticker plus
  Market Value, Value or Quantity columns), or search. Anything that can't be used is listed
  with the reason and the dollar amount left out.
- **History window:** each portfolio is analysed over the dates all of its holdings traded.
  Under two years of shared history shows a reliability warning; under one year is refused.
```

Update the Tech Stack line so it no longer says "live via urllib + snapshot fallback". It becomes: "data: daily-built release bundle (GitHub Actions + yfinance), live Yahoo lookup for tickers outside it".

- [ ] **Step 5: Commit**

```bash
git add risk_engine/config.py risk_engine/data.py tests/test_config.py tools/snapshot_reference.py README.md
git commit -m "Remove the hand-picked universe and document the data pipeline"
```

---

### Task 13: Ship and verify live

Every step here runs against production. Nothing in this task is claimed from local green.

- [ ] **Step 1: Pre-push review**

Run the `ship-check` skill on the branch diff (repo owner's standing rule). Fix what it flags, run `python -m pytest -q`, and commit any fixes.

- [ ] **Step 2: Push (ask the repo owner first)**

```bash
git fetch origin
git status -sb          # must show main ahead of origin/main only, never diverged
git push origin main
```

- [ ] **Step 3: Watch CI go green**

```bash
gh run list --workflow tests --limit 1
gh run watch <run-id> --exit-status
```

Expected: success in a clean environment.

- [ ] **Step 4: First real data build**

```bash
gh workflow run refresh-data -f mode=full
gh run list --workflow refresh-data --limit 1
gh run watch <run-id> --exit-status
gh release download data-latest -p refresh_report.json -D /tmp/rel --clobber
python -c "import json; r=json.load(open('/tmp/rel/refresh_report.json')); print(r['as_of'], r['n_tickers'], len(r['failed']), len(r['quarantined'])); print(r['failed']); print(r['quarantined'])"
```

If the run fails at the 2% gate, read the failed list before touching any threshold. Raising the threshold to make a run pass is not a fix.

- [ ] **Step 5: Replace the bundled fallback with the real bundle**

```bash
gh release download data-latest -D data/fallback --clobber
ls -la data/fallback
git add data/fallback
git commit -m "Bundle the first full S&P 500 data snapshot as the offline fallback"
git push origin main
```

Record `prices.csv.gz` size from `ls -la`.

- [ ] **Step 6: Verify the Streamlit app live**

With Playwright on `https://icn93ppxyjtbgd5sxdjvpb.streamlit.app/` (wait out the cold start):
1. The hero shows `Prices as of <recent date> close · N/M tickers`, with no stale banner.
2. The example portfolio loads and every section renders without an error box.
3. Paste `AAPL 10000`, `MSFT 5000`, `XYZQQ 3000` → "Not included" lists XYZQQ with "$3,000 excluded".
4. Paste `AAPL 10000`, `GEV 5000` (listed 2024) → stress test caption says COVID-19 is not covered for GEV; the 2025 Tariff Shock card shows.
5. Search tab: add `PLTR`. Its Sector shows Technology, which means the live lookup works on Streamlit Cloud.
6. Measure boot time (navigation start to the hero rendering) on a cold and a warm load.

Screenshot each step.

- [ ] **Step 7: Verify the Render API live**

```bash
API=https://marketplug-risk-api.onrender.com
curl -s $API/health
post() { curl -s -o /dev/null -w "%{http_code}\n" -X POST $API/analyze -H "Content-Type: application/json" -d "$1"; }
post '{"holdings":{"AAPL":10000,"MSFT":5000}}'          # 200
post '{"holdings":{"AAPL":10000,"PLTR":5000}}'          # 200 if Render reaches Yahoo, else 422 "unavailable"
post '{"holdings":{"AAPL":10000,"XYZQQ":5000}}'         # 422
post '{"holdings":{"AAPL":10000,"RELIANCE.NS":5000}}'   # 422
post '{"holdings":{"AAPL":10000,"A/B":5000}}'           # 422
post '{"holdings":{"AAPL":1e400,"MSFT":5000}}'          # 422 (August regression)
```

`/health` must show a recent `data_as_of`, `"stale": false`, `"data_source": "release"` or `"cache"`. If PLTR returns 422 "data source unavailable", Render can't reach Yahoo: record that, and apply Task 4, Step 7 to the API's requirements.

- [ ] **Step 8: Record measurements in the spec**

Append `## Post-deploy measurements (YYYY-MM-DD)` to the spec: real failure and held-back counts from the first build, bundle size, app cold/warm boot time, the Render live-lookup result, and whether the 2% threshold held. Commit and push: `git commit -m "Record first production data build and boot-time measurements"`.

- [ ] **Step 9: Two weeks later (calendar reminder, not a blocker)**

Check that `refresh-data` ran every weekday, that dated releases older than 14 days are gone, and that the repo's Actions tab shows no "This workflow will be disabled soon" notice.
