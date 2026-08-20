"""Price loading. Live pull via Python's own SSL stack (survives HTTPS-inspecting
networks that break curl-based clients); frozen snapshot as fallback. Pure — no UI."""
import os
import numpy as np
import pandas as pd
from risk_engine.config import TICKERS

_SNAPSHOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prices.csv")


def _fetch(sym):
    import urllib.request, json, datetime
    p1 = int(datetime.datetime(2018, 1, 1).timestamp())
    p2 = int(datetime.datetime.now().timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={p1}&period2={p2}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    d = json.loads(urllib.request.urlopen(req, timeout=15).read())
    r = d["chart"]["result"][0]
    idx = pd.to_datetime([datetime.date.fromtimestamp(t) for t in r["timestamp"]])
    cl = r["indicators"]["adjclose"][0]["adjclose"]
    return pd.Series(cl, index=idx, name=sym)


def load_prices(prefer_live=True):
    """Return (prices, log_returns, source_label)."""
    prices, source = None, None

    if prefer_live:
        try:
            series = {}
            for t in TICKERS + ["SPY"]:
                try:
                    s = _fetch(t)
                    if s.notna().sum() > 500:
                        series[t] = s
                except Exception:
                    pass
            if len(series) >= 2:
                prices = pd.concat(series.values(), axis=1).ffill().dropna()
                source = "Yahoo Finance · live"
        except Exception:
            prices = None

    if prices is None or prices.shape[1] < 2:
        if os.path.exists(_SNAPSHOT):
            prices = pd.read_csv(_SNAPSHOT, index_col=0, parse_dates=True).ffill().dropna()
            source = f"frozen snapshot · {prices.index.max().date()}"
        else:
            return None, None, None

    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr, source
