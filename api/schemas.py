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
    """503 body from /analyze and /report when no market-data snapshot can be read."""
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


class ReportRequest(BaseModel):
    holdings: dict[str, float]
    confidence: float = Field(default=0.95, ge=0.90, le=0.99)
    shrinkage: bool = False     # Ledoit-Wolf covariance for the optimizer

    @field_validator("holdings")
    @classmethod
    def _check(cls, v):
        return validate_holdings(v)


class Series(BaseModel):
    dates: list[str]
    values: list[float]


class ScoreOut(BaseModel):
    score: int
    bad_year_loss: float
    loss_dollars: float
    gain_dollars: float


class ScoreSection(BaseModel):
    portfolio: ScoreOut
    benchmark: ScoreOut
    horizon_days: int


class HoldingOut(BaseModel):
    ticker: str
    name: str
    sector: str
    type: str
    asset_class: str
    amount: float
    weight: float
    source: str


class PortfolioSection(BaseModel):
    value: float
    holdings: list[HoldingOut]


class RiskSection(BaseModel):
    confidence: float
    hist_var: float
    hist_cvar: float
    gaussian_var: float
    gaussian_cvar: float
    dollars: dict[str, float]
    gap_dollars: float


class ProfileSection(BaseModel):
    annual_return: float
    annual_vol: float
    sharpe: float
    max_drawdown: float
    trading_days: int


class CompareRow(BaseModel):
    label: str
    portfolio: float
    benchmark: float
    higher_is_better: bool
    kind: str


class BenchmarkSection(BaseModel):
    rows: list[CompareRow]
    beta: float
    growth: dict[str, Series]


class DistributionSection(BaseModel):
    edges: list[float]
    density: list[float]
    mean: float
    std: float


class CrisisOut(BaseModel):
    label: str
    context: str
    start: str
    end: str
    status: str
    missing: list[str]
    total: Optional[float]
    worst_day: Optional[float]
    max_dd: Optional[float]
    loss_dollars: Optional[float]
    path: Optional[Series]


class CorrelationSection(BaseModel):
    tickers: list[str]
    matrix: list[list[Optional[float]]]     # null where a holding's price never moved


class BacktestSeries(BaseModel):
    dates: list[str]
    var: list[float]
    realized: list[float]
    breach: list[int]


class BacktestSection(BaseModel):
    window: int
    min_observations: int
    methods: dict[str, BacktestRow]
    series: BacktestSeries


class RiskShare(BaseModel):
    ticker: str
    weight: float
    risk_share: float


class RiskContributionSection(BaseModel):
    rows: list[RiskShare]
    biggest_driver: str
    diversifier: Optional[str]


class FactorsSection(BaseModel):
    available: bool
    reason: str
    data_through: Optional[str]
    betas: Optional[dict[str, float]]
    alpha_annual: Optional[float]
    r2: Optional[float]
    variance_split: Optional[dict[str, float]]
    observations: int


class SectorMixOut(BaseModel):
    sectors: dict[str, float]
    equity: float
    diversified: float
    unplaced: float
    non_equity: dict[str, float]
    top_sector: Optional[str]
    top_sector_pct: float
    concentrated: bool


class OptPoint(BaseModel):
    ret: float
    vol: float
    sharpe: float
    weights: dict[str, float]


class FrontierPoint(BaseModel):
    vol: float
    ret: float


class ReallocRow(BaseModel):
    ticker: str
    asset_class: str
    sector: str
    current: float
    suggested: float
    delta_dollars: float


class OptimizerSection(BaseModel):
    available: bool
    reason: str
    shrinkage: bool
    sector_mix: SectorMixOut
    current: Optional[OptPoint]
    max_sharpe: Optional[OptPoint]
    min_variance: Optional[OptPoint]
    frontier: list[FrontierPoint]
    reallocation: list[ReallocRow]
    adds: list[str]
    trims: list[str]


class HorizonOut(BaseModel):
    label: str
    days: int
    p5: float
    p50: float
    p95: float
    prob_loss: float
    prob_up20: float


class MonteCarloSection(BaseModel):
    n_sims: int
    seed: int
    value: float
    band_days: list[int]
    bands: dict[str, list[float]]
    horizons: list[HorizonOut]


class RejectedOut(BaseModel):
    ticker: str
    amount: float
    reason: str


class ReportData(BaseModel):
    as_of: str
    window_start: str
    window_end: str
    window_days: int
    window_status: str
    window_message: str
    rejected: list[RejectedOut]
    assumptions: list[str]


class ReportResponse(BaseModel):
    score: ScoreSection
    portfolio: PortfolioSection
    risk: RiskSection
    profile: ProfileSection
    benchmark: BenchmarkSection
    distribution: DistributionSection
    drawdown: Series
    stress: list[CrisisOut]
    correlation: CorrelationSection
    backtest: BacktestSection
    risk_contribution: RiskContributionSection
    factors: FactorsSection
    optimizer: OptimizerSection
    monte_carlo: MonteCarloSection
    data: ReportData
