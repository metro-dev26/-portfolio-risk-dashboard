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
        out[t] = out.get(t, 0.0) + amount
        # Held to the cap after merging: spellings of one ticker add up.
        if out[t] > _MAX_AMOUNT:
            raise ValueError(f"amount for {t} exceeds the maximum of {_MAX_AMOUNT:.0f}")
    # Counted after merging: "aapl" and "AAPL" are one holding, not two.
    if not (MIN_HOLDINGS <= len(out) <= MAX_HOLDINGS):
        raise ValueError(f"holdings must contain between {MIN_HOLDINGS} and {MAX_HOLDINGS} "
                         f"distinct tickers")
    return out


class AnalyzeRequest(BaseModel):
    holdings: dict[str, float]
    confidence: float = Field(default=0.95, ge=0.90, le=0.99)
    include_backtest: bool = True

    @field_validator("holdings")
    @classmethod
    def _check(cls, v):
        return validate_holdings(v)


class Metrics(BaseModel):
    hist_var: float
    hist_cvar: float
    gaussian_var: float
    gaussian_cvar: float
    sharpe: float
    annual_vol: float
    max_drawdown: float


class Factors(BaseModel):
    betas: dict[str, float]
    alpha_annual: float
    r2: float
    variance_split: dict[str, float]
    observations: int


class Optimizer(BaseModel):
    current_sharpe: float
    max_sharpe_value: float
    max_sharpe_weights: dict[str, float]
    min_variance_weights: dict[str, float]
    top_sector: str
    top_sector_pct: float
    non_equity_pct: dict[str, float]    # bonds and gold, kept out of the sector check


class BacktestRow(BaseModel):
    observations: int
    breaches: int
    expected: float
    kupiec_p: float
    christoffersen_p: float
    passed: Optional[bool]    # null when there are too few observations to grade


class Backtest(BaseModel):
    historical: BacktestRow
    gaussian: BacktestRow


class DataInfo(BaseModel):
    as_of: str
    window_start: str
    window_days: int
    window_status: str
    window_message: str       # why the window is short; empty when it is not
    crisis_coverage: dict[str, list[str]]


class AnalyzeResponse(BaseModel):
    metrics: Metrics
    factors: Factors
    optimizer: Optimizer
    data: DataInfo
    backtest: Optional[Backtest] = None


class Problem(BaseModel):
    ticker: str
    reason: str


class ProblemDetail(BaseModel):
    problems: list[Problem]


class ProblemResponse(BaseModel):
    """422 body when the engine can't use the holdings."""
    detail: ProblemDetail


class UnavailableResponse(BaseModel):
    """503 body from /analyze when no market-data snapshot can be read."""
    detail: str


class DegradedResponse(BaseModel):
    """503 body from /health."""
    status: str
    detail: str


class PortfolioIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    holdings: dict[str, float]

    @field_validator("holdings")
    @classmethod
    def _check(cls, v):
        return validate_holdings(v)


class PortfolioOut(BaseModel):
    id: int
    name: str
    holdings: dict[str, float]
    created_at: str
