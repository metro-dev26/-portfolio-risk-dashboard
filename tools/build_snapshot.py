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
import urllib.parse
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
QUALITY_WINDOW_ROWS = 20   # bad prints are looked for in the latest rows only; older moves stand
STALE_ROWS = 5
MAX_BAD_FRAC = 0.02
FALLBACK_GIVE_UP = 10   # consecutive failed urllib retries before the rest are skipped

FACTOR_COLUMNS = ["Mkt-RF", "SMB", "HML", "RF"]
FRENCH_MISSING = (-99.99, -999.0)   # the data library's missing-value markers, in percent


def _http(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _parse_table(html):
    """-> (rows, skipped, unmapped): the rows parse_sp500 returns, how many table rows
    had no symbol or one that is not a valid ticker, and the tickers whose GICS sector
    has no Yahoo name."""
    # flavor pinned: pandas otherwise retries with html5lib when the table is missing,
    # and the ImportError hides the real problem
    table = pd.read_html(io.StringIO(html), attrs={"id": "constituents"}, flavor="lxml")[0]
    rows, unmapped, skipped = [], [], 0
    for _, r in table.iterrows():
        if pd.isna(r["Symbol"]) or not str(r["Symbol"]).strip():
            skipped += 1
            continue
        ticker = str(r["Symbol"]).strip().replace(".", "-")
        if not data.TICKER_RE.fullmatch(ticker):
            skipped += 1
            continue
        sector = GICS_TO_YAHOO.get(str(r["GICS Sector"]).strip())
        if sector is None:
            unmapped.append(ticker)
        rows.append((ticker, str(r["Security"]).strip(), sector or data.UNKNOWN_SECTOR))
    return rows, skipped, sorted(unmapped)


def parse_sp500(html):
    return _parse_table(html)[0]


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


def _no_closes():
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def _clean_closes(s):
    """Finite closes only, one per date (the latest wins), oldest first."""
    s = s[np.isfinite(s.to_numpy(dtype=float))]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _naive_daily(s):
    idx = pd.to_datetime(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return _clean_closes(pd.Series(s.to_numpy(dtype=float), index=idx.normalize(), name=s.name))


def _fetch_urllib(t, start):
    p1 = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp())
    p2 = int(dt.datetime.now(dt.timezone.utc).timestamp())
    payload = json.loads(_http("https://query1.finance.yahoo.com/v8/finance/chart/"
                               f"{urllib.parse.quote(t, safe='')}?period1={p1}&period2={p2}&interval=1d",
                               timeout=15))
    r = payload["chart"]["result"][0]
    days = [dt.datetime.fromtimestamp(x, dt.timezone.utc).date() for x in r["timestamp"]]
    closes = r["indicators"]["adjclose"][0]["adjclose"]
    return _clean_closes(pd.Series(closes, index=pd.to_datetime(days), name=t, dtype=float))


def _close_columns(df):
    """Per-ticker closes from a yf.download frame; empty when nothing came back."""
    if "Close" not in df.columns.get_level_values(0):
        return pd.DataFrame()
    closes = df["Close"]
    return closes if isinstance(df.columns, pd.MultiIndex) else closes.to_frame()


def fetch_yf(tickers, start):
    """yfinance in one batch (it handles Yahoo's checks on cloud hosts); any ticker
    it misses gets one retry over plain urllib, abandoned after FALLBACK_GIVE_UP
    consecutive failures so an outage can't stall the build ticker by ticker.
    -> (series by ticker, reason by ticker)."""
    import yfinance as yf
    got, failed = {}, {}
    if not tickers:
        return got, failed
    try:
        frame = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    except Exception as e:  # nothing can be built without prices; the cause goes to the log
        raise SystemExit(f"yfinance download failed ({type(e).__name__}: {e}) — not publishing") from e
    closes = _close_columns(frame)
    streak = 0
    for t in tickers:
        s = _naive_daily(closes[t].rename(t)) if t in closes.columns else _no_closes()
        if len(s):
            got[t] = s
            continue
        if streak >= FALLBACK_GIVE_UP:
            failed[t] = (f"no data returned; urllib retry skipped after "
                         f"{FALLBACK_GIVE_UP} consecutive retry failures")
            continue
        try:
            s = _fetch_urllib(t, start)
        except Exception as e:  # recorded per ticker in the report, never swallowed
            failed[t] = f"{type(e).__name__}: {e}"
            streak += 1
            continue
        if len(s):
            got[t] = s
            streak = 0
        else:
            failed[t] = "no data returned"
            streak += 1
    return got, failed


def merge_incremental(stored, fresh, tol=ADJ_TOL):
    """Extend stored history with fresh closes. Returns (merged, repull): repull
    names tickers whose overlapping closes disagree (or don't overlap at all),
    meaning a dividend or split re-scaled the history; those need a full fetch."""
    cols, repull = {}, []
    for t in sorted(set(stored.columns) | set(fresh)):
        old = _clean_closes(stored[t]) if t in stored.columns else _no_closes()
        new = _clean_closes(fresh[t]) if t in fresh else _no_closes()
        if new.empty:
            cols[t] = old
            continue
        if old.empty:
            cols[t] = new
            continue
        overlap = old.index.intersection(new.index)
        if len(overlap) == 0 or (np.abs(new[overlap] / old[overlap] - 1) > tol).any():
            repull.append(t)
            cols[t] = old
            continue
        cols[t] = pd.concat([old, new[new.index > old.index.max()]])
    return pd.DataFrame(cols).sort_index(), repull


def quality_gate(prices, spike=SPIKE, reversal=REVERSAL, stale_rows=STALE_ROWS,
                 window_rows=QUALITY_WINDOW_ROWS):
    """-> {ticker: reason} to hold back. A bad print jumps more than `spike` and the
    next close undoes at least `reversal` of the jump; a real crash doesn't bounce
    back overnight. Only jumps in the table's last `window_rows` rows are judged, so a
    real move that is already in the history (a meme-stock day) is not re-flagged by
    every later build. Stale = no close in the table's last `stale_rows` rows."""
    held = {}
    recent = prices.index[-stale_rows:]
    window = prices.index[-window_rows:]
    for t in prices.columns:
        s = prices[t].dropna()
        if s.empty:
            held[t] = "no prices"
            continue
        if s.index.max() < recent[0]:
            held[t] = f"no new close since {s.index.max().date()}"
            continue
        prev = s.shift(1)
        delta = s - prev
        jump = delta / prev
        undone = (s - s.shift(-1)) / delta.where(delta != 0)
        bad = (jump.abs() > spike) & (undone >= reversal) & (s.index >= window[0])
        if bad.any():
            held[t] = f"suspected bad print on {bad[bad].index[0].date()}"
    return held


def fetch_factors():
    z = zipfile.ZipFile(io.BytesIO(_http(FF_URL)))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hdr = next((i for i, line in enumerate(lines) if line.strip().startswith(",Mkt-RF")), None)
    if hdr is None:
        raise RuntimeError("Fama-French file has no ',Mkt-RF' header row; its format changed")
    rows = []
    for line in lines[hdr + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit() and parts[0] >= "20180101":
            values = [float(x) for x in parts[1:]]
            rows.append([parts[0]] + [np.nan if v in FRENCH_MISSING else v / 100.0   # percent -> decimal
                                      for v in values])
    df = pd.DataFrame(rows, columns=["Date", *FACTOR_COLUMNS])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date").sort_index()


def _complete_factors(factors):
    """The API cannot serve a factor table with gaps, so incomplete rows are dropped
    and counted; a table with the wrong columns or no complete row blocks publishing."""
    if list(factors.columns) != FACTOR_COLUMNS:
        raise SystemExit(f"factor columns are {list(factors.columns)}, expected "
                         f"{FACTOR_COLUMNS} — not publishing")
    complete = factors.replace([np.inf, -np.inf], np.nan).dropna()
    if complete.empty:
        raise SystemExit("factor table has no row free of missing values — not publishing")
    return complete, len(factors) - len(complete)


def write_bundle(out_dir, prices, universe, factors, report):
    os.makedirs(out_dir, exist_ok=True)
    prices.to_csv(os.path.join(out_dir, "prices.csv.gz"), compression="gzip", float_format="%.6f")
    factors.to_csv(os.path.join(out_dir, "factors.csv"))
    with open(os.path.join(out_dir, "universe.json"), "w") as f:
        json.dump(universe, f, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "refresh_report.json"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)


def _read_prev(prev_dir):
    """-> (previous bundle or None, how the report describes it)."""
    try:
        return data.read_bundle(prev_dir, "previous"), "used"
    except data.SnapshotUnavailable as e:
        if not any(os.path.exists(os.path.join(prev_dir, f)) for f in data.BUNDLE_FILES):
            return None, "none"
        return None, f"rejected: {e}"


def _fetch_history(mode, prev, universe, tickers, fetch):
    """-> (prices, failed, repulled). Incremental mode pulls a short overlapping window
    for stored tickers, then a full history for new and re-scaled ones."""
    if mode == "full":
        got, failed = fetch(tickers, START)
        return pd.DataFrame(got).sort_index(), failed, []
    stored = prev.prices[[t for t in prev.prices.columns if t in universe]]
    since = (stored.index.max() - pd.offsets.BDay(OVERLAP_DAYS)).strftime("%Y-%m-%d")
    got, failed = fetch([t for t in tickers if t in stored.columns], since)
    prices, repulled = merge_incremental(stored, got)
    need_full = repulled + [t for t in tickers if t not in stored.columns]
    if need_full:
        got_full, failed_full = fetch(need_full, START)
        failed.update(failed_full)
        if got_full:
            prices = pd.concat([prices.drop(columns=[t for t in got_full if t in prices.columns]),
                                pd.DataFrame(got_full)], axis=1).sort_index()
    return prices, failed, repulled


def _block(out_dir, report, headline, failed, held):
    """Refuse to publish: leave only blocked_report.json in out_dir and exit with the
    reason and up to 10 offenders, the benchmark first."""
    offenders = sorted(set(failed) | set(held), key=lambda t: (t != BENCHMARK, t))
    lines = [f"  {t}: " + "; ".join(r for r in (failed.get(t), held.get(t)) if r)
             for t in offenders[:10]]
    if len(offenders) > 10:
        lines.append(f"  ... and {len(offenders) - 10} more (see blocked_report.json)")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "blocked_report.json"), "w") as f:
        json.dump({**report, "blocked_reason": headline}, f, indent=1, sort_keys=True)
    raise SystemExit("\n".join([f"{headline} — not publishing", *lines]))


def run(mode, prev_dir, out_dir, *, fetch=fetch_yf, get_html=None, get_factors=fetch_factors):
    get_html = get_html or (lambda: _http(SP500_URL).decode("utf-8"))
    prev, prev_status = _read_prev(prev_dir)
    if prev is None or prev.prices.empty:
        mode = "full"
    try:
        rows, rows_skipped, unmapped_sectors = _parse_table(get_html())
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"S&P 500 list unavailable ({type(e).__name__}: {e}) — not publishing") from e
    try:
        universe, used_prev = build_universe(rows, prev.universe if prev else None)
    except RuntimeError as e:
        raise SystemExit(f"{e} — not publishing") from e
    tickers = sorted(universe)
    if not tickers:
        raise SystemExit("universe is empty — not publishing")

    prices, failed, repulled = _fetch_history(mode, prev, universe, tickers, fetch)
    for t in tickers:
        if t not in prices.columns and t not in failed:
            failed[t] = "no data returned"
    benchmark_missing = BENCHMARK in failed or BENCHMARK not in prices.columns
    held = quality_gate(prices)
    prices = prices.drop(columns=list(held))

    as_of = str(prices[BENCHMARK].dropna().index.max().date()) if BENCHMARK in prices.columns else None
    report = {
        "as_of": as_of,
        "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "n_tickers": len(tickers),
        "failed": failed,
        "quarantined": held,
        "repulled": repulled,
        "universe_fallback": used_prev,
        "previous_bundle": prev_status,
        "sp500_rows_skipped": rows_skipped,
        "unmapped_sectors": unmapped_sectors,
    }
    if benchmark_missing:
        _block(out_dir, report, f"benchmark {BENCHMARK} failed to download", failed, held)
    if BENCHMARK in held:
        _block(out_dir, report, f"benchmark {BENCHMARK} held back ({held[BENCHMARK]})", failed, held)
    bad = len(set(failed) | set(held))
    if bad / len(tickers) > MAX_BAD_FRAC:
        _block(out_dir, report, f"{bad}/{len(tickers)} tickers failed or held back "
                                f"({bad / len(tickers):.1%} > {MAX_BAD_FRAC:.0%})", failed, held)

    if mode == "full":
        try:
            raw_factors = get_factors()
        except Exception as e:  # the reason goes to the log; nothing is published without factors
            raise SystemExit(f"factors unavailable ({type(e).__name__}: {e}) — not publishing") from e
    else:
        raw_factors = prev.factors
    factors, factor_rows_dropped = _complete_factors(raw_factors)
    report["factors_through"] = str(factors.index.max().date())
    report["factor_rows_dropped"] = factor_rows_dropped

    write_bundle(out_dir, prices, universe, factors, report)
    print(json.dumps({k: report[k] for k in ("as_of", "mode", "n_tickers", "factor_rows_dropped")}
                     | {"failed": len(failed), "held_back": len(held), "repulled": len(repulled)}))
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
