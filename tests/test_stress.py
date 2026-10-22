import numpy as np
import pandas as pd
import pytest

from risk_engine import stress


def _prices():
    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    return pd.DataFrame({"A": [100, 110, 99, 120, 120, 132, 132, 132],
                         "B": [50, 50, 50, 50, 25, 25, 25, 25]}, idx, dtype=float)


CRISIS = [("Test crash", "2020-01-01", "2020-01-10", "context")]


def test_buy_and_hold_path_worked_by_hand():
    (r,) = stress.replay(_prices(), ["A", "B"], np.array([0.5, 0.5]), CRISIS)
    # A/A0 = 1, 1.1, 0.99, 1.2, 1.2, 1.32, 1.32, 1.32 ; B/B0 = 1,1,1,1,0.5,0.5,0.5,0.5
    expected = [1.0, 1.05, 0.995, 1.1, 0.85, 0.91, 0.91, 0.91]
    assert r.status == "ok" and r.missing == []
    assert list(r.path.round(6)) == pytest.approx(expected)
    assert r.total == pytest.approx(-0.09)
    assert r.max_dd == pytest.approx(0.85 / 1.1 - 1)
    assert r.worst_day == pytest.approx(0.85 / 1.1 - 1)


def test_weights_and_the_two_drawdown_measures_are_not_interchangeable():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    p = pd.DataFrame({"A": [100, 90, 80, 85, 105], "B": [10] * 5}, idx, dtype=float)
    (r,) = stress.replay(p, ["A", "B"], np.array([0.7, 0.3]), CRISIS)   # exactly MIN_CRISIS_DAYS
    # 0.7 * A/A0 + 0.3 = 1, 0.93, 0.86, 0.895, 1.035
    assert r.status == "ok"
    assert r.total == pytest.approx(0.035)
    assert r.worst_day == pytest.approx(0.86 / 0.93 - 1)
    assert r.max_dd == pytest.approx(0.86 - 1)


def test_a_gap_day_inside_the_window_is_skipped_not_scored_as_a_crash():
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    p = pd.DataFrame({"A": [100.0] * 6, "B": [10, 10, np.nan, 10, 10, 10]}, idx)
    (r,) = stress.replay(p, ["A", "B"], np.array([0.7, 0.3]), CRISIS)
    assert r.status == "ok" and len(r.path) == 5
    assert r.worst_day == pytest.approx(0.0) and r.max_dd == pytest.approx(0.0)


def test_a_crisis_a_holding_missed_is_returned_not_dropped():
    p = _prices()
    p.loc[:"2020-01-03", "B"] = np.nan          # B lists after the crisis start
    (r,) = stress.replay(p, ["A", "B"], np.array([0.5, 0.5]), CRISIS)
    assert r.status == "not_covered" and r.missing == ["B"]
    assert r.total is None and r.path is None


def test_a_crisis_with_too_few_clean_days_is_returned_with_that_status():
    short = [("Blip", "2020-01-01", "2020-01-06", "ctx")]     # 4 business days
    (r,) = stress.replay(_prices(), ["A", "B"], np.array([0.5, 0.5]), short)
    assert r.status == "too_few_days" and r.total is None


def test_real_crises_all_come_back_in_order(market):
    prices, _ = market
    out = stress.replay(prices, ["AAPL", "TLT"], np.array([0.6, 0.4]))
    from risk_engine.config import CRISES
    assert [r.label for r in out] == [c[0] for c in CRISES]
    assert all(r.status == "ok" for r in out)
    covid = out[1]
    assert covid.total < 0 and covid.max_dd < 0
