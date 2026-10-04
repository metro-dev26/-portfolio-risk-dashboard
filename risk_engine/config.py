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

# Funds that spread across many sectors: they count as equity, but a portfolio cannot be
# concentrated in them as a sector.
DIVERSIFIED_FUND_SECTORS = frozenset({"Broad Market", "International"})

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
BACKTEST_MIN_OBS = 100    # fewer out-of-sample days than this and a pass/fail verdict means nothing
FACTOR_MIN_OBS = 60       # fewer days shared with the factor data and the loadings are noise
STALE_BUSINESS_DAYS = 3

LIVE_MAX_API, LIVE_BUDGET_API_S = 5, 10.0
LIVE_MAX_UI, LIVE_BUDGET_UI_S = 20, 40.0
