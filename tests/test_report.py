import dataclasses
import json

import numpy as np
import pytest

from risk_engine import data, optimize
from risk_engine.inputs import prepare
from risk_engine.report import build_report, report_from_inputs

H = {"AAPL": 30000, "JPM": 20000, "TLT": 40000, "IEF": 10000}
SECTIONS = {"score", "portfolio", "risk", "profile", "benchmark", "distribution", "drawdown",
            "stress", "correlation", "backtest", "risk_contribution", "factors", "optimizer",
            "monte_carlo", "data"}


@pytest.fixture(scope="module")
def rep():
    return build_report(H)


def test_report_has_every_section_and_is_json_safe(rep):
    assert set(rep) == SECTIONS
    json.dumps(rep, allow_nan=False)        # raises on numpy types or NaN


def test_headline_score_is_read_off_the_one_year_monte_carlo(rep):
    one_year = next(h for h in rep["monte_carlo"]["horizons"] if h["days"] == 252)
    s, v = rep["score"]["portfolio"], rep["portfolio"]["value"]
    assert s["loss_dollars"] == pytest.approx(max(0.0, v - one_year["p5"]))
    assert s["gain_dollars"] == pytest.approx(one_year["p95"] - v)
    assert 1 <= s["score"] <= 99 and 1 <= rep["score"]["benchmark"]["score"] <= 99
    assert rep["score"]["horizon_days"] == 252


def test_bond_heavy_portfolio_scores_below_the_sp500(rep):
    assert rep["score"]["portfolio"]["score"] < rep["score"]["benchmark"]["score"]


def test_var_dollars_use_the_simple_return(rep):
    r, v = rep["risk"], rep["portfolio"]["value"]
    assert r["dollars"]["hist_var"] == pytest.approx(np.expm1(r["hist_var"]) * v)
    assert r["gap_dollars"] == pytest.approx(r["dollars"]["hist_var"] - r["dollars"]["gaussian_var"])


def test_every_crisis_is_present_with_a_status(rep):
    from risk_engine.config import CRISES
    assert [c["label"] for c in rep["stress"]] == [c[0] for c in CRISES]
    ok = [c for c in rep["stress"] if c["status"] == "ok"]
    assert ok and all(c["loss_dollars"] == pytest.approx(c["total"] * rep["portfolio"]["value"]) for c in ok)
    assert all(c["path"]["values"][0] == 0.0 for c in ok)     # return from start


def test_series_are_parallel_lists_of_iso_dates(rep):
    s = rep["drawdown"]
    assert len(s["dates"]) == len(s["values"]) > 500
    assert s["dates"][0] > rep["data"]["window_start"]       # first return is the day after the first price
    assert len(s["dates"][0]) == 10


def test_risk_shares_sum_to_one_and_name_a_driver(rep):
    rc = rep["risk_contribution"]
    assert sum(row["risk_share"] for row in rc["rows"]) == pytest.approx(1.0)
    assert rc["biggest_driver"] in H


def test_optimizer_section_and_sector_mix(rep):
    o = rep["optimizer"]
    assert o["available"] is True and o["reason"] == ""
    assert sum(o["max_sharpe"]["weights"].values()) == pytest.approx(1.0)
    assert o["frontier"] and all(0 < p["vol"] < 1 for p in o["frontier"])
    assert o["sector_mix"]["sectors"] == {"Technology": pytest.approx(0.3), "Financial Services": pytest.approx(0.2)}
    assert o["sector_mix"]["non_equity"] == {"Bond": pytest.approx(0.5)}
    assert o["sector_mix"]["concentrated"] is False


def test_unconverged_optimizer_is_a_partial_section_not_an_error(monkeypatch):
    def boom(*a, **k):
        raise optimize.OptimizerError("the optimizer did not converge (test)")
    monkeypatch.setattr(optimize, "max_sharpe_weights", boom)
    rep = build_report(H)
    assert rep["optimizer"]["available"] is False
    assert "did not converge" in rep["optimizer"]["reason"]
    assert rep["optimizer"]["max_sharpe"] is None
    assert rep["optimizer"]["sector_mix"]["non_equity"] == {"Bond": pytest.approx(0.5)}


def test_thin_factor_overlap_is_a_partial_section():
    snap = data.load_snapshot(use_memo=False)
    bare = dataclasses.replace(snap, factors=snap.factors.iloc[:30])
    rep = build_report(H, snapshot=bare)
    assert rep["factors"]["available"] is False and "at least 60" in rep["factors"]["reason"]
    assert rep["factors"]["betas"] is None
    assert rep["score"]["portfolio"]["score"] >= 1        # the rest still computed


def test_shrinkage_changes_the_optimizer_only():
    a, b = build_report(H), build_report(H, shrinkage=True)
    assert a["score"] == b["score"] and a["risk"] == b["risk"]
    assert b["optimizer"]["shrinkage"] is True


def test_portfolio_holding_spy_scores_against_itself_cleanly():
    rep = build_report({"SPY": 50000, "AAPL": 50000})
    assert rep["score"]["benchmark"]["score"] >= 1
    assert rep["benchmark"]["beta"] > 0.5


def test_misaligned_benchmark_dates_are_refused_before_the_shared_draws():
    inp = prepare(H, data.load_snapshot(use_memo=False))
    skewed = dataclasses.replace(inp, spy_r=inp.spy_r.iloc[1:])
    with pytest.raises(ValueError, match="same dates"):
        report_from_inputs(skewed)
