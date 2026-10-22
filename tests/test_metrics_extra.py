import numpy as np
import pandas as pd
import pytest

from risk_engine import metrics


def test_to_dollars_uses_the_simple_return_not_the_log_return():
    # A -10% log return is a 9.516% real loss, not 10%.
    assert metrics.to_dollars(-0.10, 100_000) == pytest.approx(-9516.258, abs=1e-3)
    assert metrics.to_dollars(0.0, 100_000) == 0.0


def test_return_histogram_is_a_density_over_the_returns():
    r = pd.Series([-0.02, -0.01, 0.0, 0.0, 0.01, 0.03])
    h = metrics.return_histogram(r, bins=5)
    assert len(h["edges"]) == 6 and len(h["density"]) == 5
    assert h["edges"][0] == pytest.approx(-0.02) and h["edges"][-1] == pytest.approx(0.03)
    widths = np.diff(h["edges"])
    assert float(np.sum(np.array(h["density"]) * widths)) == pytest.approx(1.0)


def test_correlation_matches_pandas_and_keeps_ticker_order(market):
    _, lr = market
    c = metrics.correlation(lr, ["MSFT", "AAPL"])
    assert c["tickers"] == ["MSFT", "AAPL"]
    assert c["matrix"][0][0] == pytest.approx(1.0)
    assert c["matrix"][0][1] == pytest.approx(lr["MSFT"].corr(lr["AAPL"]))
