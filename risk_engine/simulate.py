"""Bootstrap Monte Carlo and the 1-99 risk score. Pure.

One run draws the longest horizon once. Every shorter horizon, the fan-chart bands
and the risk score are read off that one run, so they can never disagree. Passing
the same `draws` to two portfolios (yours and the S&P 500) gives both the same
random days, so their scores differ only because the holdings do."""
from dataclasses import dataclass

import numpy as np

HORIZONS = {"6 months": 126, "1 year": 252, "2 years": 504, "3 years": 756}
SCORE_HORIZON = 252
N_SIMS = 10_000
SEED = 42
BAND_STEP = 6                     # every horizon is a multiple of 6, so each lands on a band point
BAND_PCTS = (5, 25, 50, 75, 95)
FULL_SCALE_LOSS = 0.60            # a bad-year loss this large or larger scores 99


def draw_indices(n_obs, *, days=max(HORIZONS.values()), n_sims=N_SIMS, seed=SEED):
    """Which historical day each simulated day replays. int32 halves the memory."""
    return np.random.default_rng(seed).integers(0, n_obs, size=(n_sims, days), dtype=np.int32)


@dataclass(frozen=True)
class Simulation:
    value: float
    band_days: list    # 0, 6, ..., the longest horizon
    bands: dict        # percentile -> simulated value at each band day
    horizons: dict     # days -> p5, p50, p95, prob_loss, prob_up20


@dataclass(frozen=True)
class Score:
    score: int
    bad_year_loss: float   # fraction lost at the 5th percentile, floored at 0
    loss_dollars: float
    gain_dollars: float


def simulate(daily_log_returns, value, draws):
    r = np.asarray(daily_log_returns, dtype=float)
    paths = r[draws]                           # the one n_sims x days buffer
    np.cumsum(paths, axis=1, out=paths)
    np.exp(paths, out=paths)
    paths *= value
    days = draws.shape[1]
    band_days = list(range(0, days + 1, BAND_STEP))
    at = np.percentile(paths[:, [d - 1 for d in band_days[1:]]], BAND_PCTS, axis=0)
    bands = {p: [float(value)] + row.tolist() for p, row in zip(BAND_PCTS, at)}
    horizons = {}
    for d in HORIZONS.values():
        if d > days:
            continue
        terminal = paths[:, d - 1]
        p5, p50, p95 = np.percentile(terminal, [5, 50, 95])
        horizons[d] = {"p5": float(p5), "p50": float(p50), "p95": float(p95),
                       "prob_loss": float((terminal < value).mean()),
                       "prob_up20": float((terminal >= value * 1.2).mean())}
    return Simulation(float(value), band_days, bands, horizons)


def score_from_loss(loss):
    return int(min(99, max(1, round(1 + 98 * loss / FULL_SCALE_LOSS))))


def risk_score(sim):
    h = sim.horizons[SCORE_HORIZON]
    loss = max(0.0, 1.0 - h["p5"] / sim.value)
    return Score(score_from_loss(loss), loss, max(0.0, sim.value - h["p5"]), h["p95"] - sim.value)
