"""Market data access. Snapshot first: a bundle built daily by a GitHub Action and
published as a release, cached on local disk, with a dated bundle in the repo as
the fallback when the download fails. Pure — no UI."""
import dataclasses
import datetime as dt
import gzip
import json
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_engine.config import DATA_RELEASE_URL, STALE_BUSINESS_DAYS

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_DIR = os.path.join(_REPO, "data", "fallback")
BUNDLE_FILES = ("prices.csv.gz", "universe.json", "factors.csv", "refresh_report.json")
FALLBACK_RETRY_S = 300
_UA = {"User-Agent": "Mozilla/5.0"}
_CACHE_DIR = None


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
    """Read a bundle from disk. Raises SnapshotUnavailable on missing or corrupt files."""
    missing = [f for f in BUNDLE_FILES if not os.path.exists(os.path.join(directory, f))]
    if missing:
        raise SnapshotUnavailable(f"{directory} is missing {', '.join(missing)}")

    try:
        prices = pd.read_csv(os.path.join(directory, "prices.csv.gz"), index_col=0, parse_dates=True)
    except (OSError, gzip.BadGzipFile, EOFError, ValueError, zlib.error) as e:
        raise SnapshotUnavailable(
            f"{directory}: prices.csv.gz is unreadable ({type(e).__name__}: {e})") from e

    try:
        factors = pd.read_csv(os.path.join(directory, "factors.csv"), index_col=0, parse_dates=True)
    except (OSError, EOFError, ValueError) as e:
        raise SnapshotUnavailable(
            f"{directory}: factors.csv is unreadable ({type(e).__name__}: {e})") from e

    try:
        with open(os.path.join(directory, "universe.json")) as f:
            universe = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise SnapshotUnavailable(
            f"{directory}: universe.json is unreadable ({type(e).__name__}: {e})") from e

    try:
        with open(os.path.join(directory, "refresh_report.json")) as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise SnapshotUnavailable(
            f"{directory}: refresh_report.json is unreadable ({type(e).__name__}: {e})") from e

    if not isinstance(report, dict):
        raise SnapshotUnavailable(
            f"{directory}: refresh_report.json must be a dict, got {type(report).__name__}")

    if "as_of" not in report:
        raise SnapshotUnavailable(f"{directory}: refresh_report.json missing 'as_of'")

    return Snapshot(prices.sort_index(), universe, factors, report, source)


def _download(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_default_cache_dir():
    """Return the default private cache directory, creating it once."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = tempfile.mkdtemp(prefix="marketplug-cache-")
    return _CACHE_DIR


def _atomic_write(path, payload):
    """Write payload atomically using tempfile.mkstemp."""
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirname)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _from_release(release_url, cache_dir, fetch):
    """Download and cache a release bundle. Validates before committing to cache."""
    report_bytes = fetch(f"{release_url}/refresh_report.json")
    remote_as_of = json.loads(report_bytes)["as_of"]

    cached_report = os.path.join(cache_dir, "refresh_report.json")
    if _bundle_complete(cache_dir):
        try:
            with open(cached_report) as f:
                cached_report_data = json.load(f)
            # Tolerate non-dict reports (treat as a cache miss)
            if isinstance(cached_report_data, dict) and cached_report_data.get("as_of") == remote_as_of:
                # Try to read the cached bundle; if it fails, fall through to redownload
                try:
                    return read_bundle(cache_dir, "cache")
                except SnapshotUnavailable:
                    pass  # Cache is corrupt; redownload
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # Cache report unreadable; redownload

    # Download all files to a staging directory, validate, then move to cache
    os.makedirs(cache_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=".staging-", dir=cache_dir)
    try:
        for name in ("prices.csv.gz", "universe.json", "factors.csv"):
            _atomic_write(os.path.join(staging, name), fetch(f"{release_url}/{name}"))
        _atomic_write(os.path.join(staging, "refresh_report.json"), report_bytes)

        # Validate the staging bundle before committing to cache
        snap = read_bundle(staging, "release")

        # Move staging files to cache
        for name in BUNDLE_FILES:
            src = os.path.join(staging, name)
            dst = os.path.join(cache_dir, name)
            os.replace(src, dst)

        return snap
    finally:
        # Clean up staging directory
        try:
            shutil.rmtree(staging)
        except OSError:
            pass


_MEMO = {}


def load_snapshot(*, data_dir=None, release_url=DATA_RELEASE_URL, cache_dir=None,
                  fallback_dir=FALLBACK_DIR, fetch=_download, max_age_s=3600, use_memo=True,
                  clock=time.time):
    """Return the freshest readable bundle. Raises SnapshotUnavailable only when
    even the bundled fallback can't be read.

    Memoizes per (release_url, cache_dir, fallback_dir). Release/cache results
    memoized for max_age_s; fallback results for FALLBACK_RETRY_S (300s).
    clock parameter is for testability."""
    pinned = data_dir or os.environ.get("MARKETPLUG_DATA_DIR")
    if pinned:
        return read_bundle(pinned, "pinned")

    cache_dir = cache_dir or _get_default_cache_dir()
    memo_key = (release_url, cache_dir, fallback_dir)
    now = clock()

    # Check memo: different TTL for fallback vs release/cache
    if use_memo and memo_key in _MEMO:
        snap, at, is_fallback = _MEMO[memo_key]
        ttl = FALLBACK_RETRY_S if is_fallback else max_age_s
        if now - at < ttl:
            return snap

    try:
        snap = _from_release(release_url, cache_dir, fetch)
        is_fallback = False
    except Exception as e:  # network, HTTP status, bad JSON or a corrupt file: use the fallback
        try:
            snap = read_bundle(fallback_dir, "fallback")
            snap = dataclasses.replace(
                snap,
                warning=f"Could not download today's data ({type(e).__name__}: {e}); "
                        f"showing the bundled copy instead.")
            is_fallback = True
        except SnapshotUnavailable as fallback_err:
            # Both release and fallback failed; include both in the message
            raise SnapshotUnavailable(
                f"Release failed ({type(e).__name__}: {e}); "
                f"fallback also failed ({fallback_err})") from e

    if use_memo:
        _MEMO[memo_key] = (snap, now, is_fallback)
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


TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,10}$")
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
