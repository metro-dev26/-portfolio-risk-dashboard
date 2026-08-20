"""Universe definition and shared constants for the risk engine."""

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

TRADING_DAYS = 252
