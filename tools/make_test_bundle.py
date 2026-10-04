"""One-off: froze the June 2026 25-ticker data as the bundle every test runs
against (tests/fixtures/snapshot), so swapping the live universe never moves a tested
number. It read the repo-root prices.csv and factors.csv, which are no longer in the
repository, so it cannot be re-run."""
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
