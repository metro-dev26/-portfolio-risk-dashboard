import tracemalloc

import numpy as np
import pytest

from risk_engine import metrics, simulate as sim_mod
from risk_engine.simulate import draw_indices, risk_score, score_from_loss, simulate


def _legacy_terminal(hist_r, value, draws):
    """app.py:1187-1192 math, verbatim, on the given draws."""
    sim_paths = value * np.exp(np.cumsum(hist_r[draws], axis=1))
    terminal = sim_paths[:, -1]
    t5, t50, t95 = np.percentile(terminal, [5, 50, 95])
    return t5, t50, t95, float((terminal < value).mean()), float((terminal >= value * 1.2).mean())


def test_simulation_reproduces_the_old_math_on_the_same_draws(market):
    _, lr = market
    r = lr["AAPL"].to_numpy()
    draws = np.random.default_rng(42).integers(0, len(r), size=(2000, 252))
    s = simulate(r, 50_000, draws)
    t5, t50, t95, ploss, pup = _legacy_terminal(r, 50_000, draws)
    h = s.horizons[252]
    assert (h["p5"], h["p50"], h["p95"]) == pytest.approx((t5, t50, t95))
    assert (h["prob_loss"], h["prob_up20"]) == pytest.approx((ploss, pup))


def test_every_horizon_lands_on_a_band_point_and_bands_start_at_value(market):
    _, lr = market
    r = lr["SPY"].to_numpy()
    s = simulate(r, 10_000, draw_indices(len(r), n_sims=500))
    assert s.band_days[0] == 0 and s.band_days[-1] == 756 and len(s.band_days) == 127
    assert set(sim_mod.HORIZONS.values()) <= set(s.band_days)
    for p in sim_mod.BAND_PCTS:
        assert s.bands[p][0] == 10_000
        assert len(s.bands[p]) == 127
    one_year = s.band_days.index(252)
    assert s.bands[5][one_year] == pytest.approx(s.horizons[252]["p5"])
    assert s.bands[95][one_year] == pytest.approx(s.horizons[252]["p95"])


def test_same_draws_give_identical_results(market):
    _, lr = market
    r = lr["MSFT"].to_numpy()
    a = simulate(r, 1000, draw_indices(len(r), n_sims=300))
    b = simulate(r, 1000, draw_indices(len(r), n_sims=300))
    assert a.horizons == b.horizons and a.bands == b.bands


def test_score_is_linear_bounded_and_monotone():
    assert score_from_loss(0.0) == 1
    assert score_from_loss(0.30) == 50
    assert score_from_loss(0.60) == 99
    assert score_from_loss(0.95) == 99
    losses = np.linspace(0, 0.7, 50)
    scores = [score_from_loss(x) for x in losses]
    assert scores == sorted(scores)


def test_score_reads_the_same_run_as_the_dollar_range():
    s = sim_mod.Simulation(value=100_000, band_days=[0], bands={},
                           horizons={252: {"p5": 76_000.0, "p50": 105_000.0, "p95": 131_000.0,
                                           "prob_loss": 0.3, "prob_up20": 0.2}})
    sc = risk_score(s)
    assert sc.bad_year_loss == pytest.approx(0.24)
    assert sc.loss_dollars == pytest.approx(24_000)
    assert sc.gain_dollars == pytest.approx(31_000)
    assert sc.score == score_from_loss(0.24) == 40


def test_a_bad_year_that_still_gains_scores_one_and_loses_nothing():
    s = sim_mod.Simulation(value=100, band_days=[0], bands={},
                           horizons={252: {"p5": 101.0, "p50": 104.0, "p95": 108.0,
                                           "prob_loss": 0.0, "prob_up20": 0.0}})
    sc = risk_score(s)
    assert sc.score == 1 and sc.loss_dollars == 0.0 and sc.bad_year_loss == 0.0


def _score(lr, tickers, weights):
    pr = metrics.portfolio_returns(lr, tickers, np.asarray(weights, dtype=float))
    draws = draw_indices(len(pr), n_sims=3000)
    return risk_score(simulate(pr.to_numpy(), 100_000, draws)).score


def test_volatile_stock_scores_above_spy_and_bonds_below(market):
    _, lr = market
    spy = _score(lr, ["SPY"], [1.0])
    assert _score(lr, ["TSLA"], [1.0]) > spy
    # The score includes drift: intermediate Treasuries score below SPY, long-duration ones (TLT)
    # score well above it on this snapshot, so only IEF is asserted.
    assert _score(lr, ["IEF"], [1.0]) < spy


def test_peak_memory_stays_under_150_mb_at_full_size(market):
    _, lr = market
    r = lr["SPY"].to_numpy()
    tracemalloc.start()
    try:
        simulate(r, 100_000, draw_indices(len(r)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 150 * 2**20, f"peak {peak / 2**20:.0f} MB"

