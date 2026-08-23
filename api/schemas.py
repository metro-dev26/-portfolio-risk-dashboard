"""pydantic request/response models for the risk API. Validation is free credibility:
bad input becomes a clean 422 instead of a 500 deep in the engine."""
import math
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from risk_engine.config import TICKERS

_UNIVERSE = set(TICKERS)

# A holding above this is nonsensical for a portfolio and only serves to probe
# for overflow; reject it rather than let a degenerate weight poison the math.
_MAX_AMOUNT = 1e12


def validate_holdings(v):
    if not (2 <= len(v) <= 12):
        raise ValueError("holdings must contain between 2 and 12 tickers")
    for ticker, amount in v.items():
        if ticker not in _UNIVERSE:
            raise ValueError(f"unknown ticker: {ticker}")
        # inf/NaN pass a naive `> 0` check but divide into a NaN weight, which the
        # engine would silently return as a zeros analysis. Reject non-finite first.
        if not math.isfinite(amount):
            raise ValueError(f"amount for {ticker} must be a finite number")
        if amount <= 0:
            raise ValueError(f"amount for {ticker} must be > 0")
        if amount > _MAX_AMOUNT:
            raise ValueError(f"amount for {ticker} exceeds the maximum of {_MAX_AMOUNT:.0f}")
    return v


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


class Optimizer(BaseModel):
    current_sharpe: float
    max_sharpe_value: float
    max_sharpe_weights: dict[str, float]
    min_variance_weights: dict[str, float]
    top_sector: str
    top_sector_pct: float


class BacktestRow(BaseModel):
    breaches: int
    expected: float
    kupiec_p: float
    christoffersen_p: float
    passed: bool


class Backtest(BaseModel):
    historical: BacktestRow
    gaussian: BacktestRow


class AnalyzeResponse(BaseModel):
    metrics: Metrics
    factors: Factors
    optimizer: Optimizer
    backtest: Optional[Backtest] = None


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
