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
