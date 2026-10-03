import numpy as np

from risk_engine.data import load_prices


def test_legacy_view_shape():
    prices, lr, source = load_prices()
    assert prices.shape[0] > 2000          # ~8.5 years of daily rows
    assert "SPY" in prices.columns
    assert lr.shape[0] == prices.shape[0] - 1
    assert "snapshot" in source


def test_log_returns_are_finite():
    _, lr, _ = load_prices()
    assert np.isfinite(lr.to_numpy()).all()
