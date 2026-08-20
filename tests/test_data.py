from risk_engine.data import load_prices


def test_snapshot_load_shape():
    prices, lr, source = load_prices(prefer_live=False)
    assert prices is not None
    assert prices.shape[0] > 2000          # ~8.5 years of daily rows
    assert "SPY" in prices.columns
    assert lr.shape[0] == prices.shape[0] - 1   # one row lost to differencing
    assert "snapshot" in source


def test_log_returns_are_finite():
    _, lr, _ = load_prices(prefer_live=False)
    import numpy as np
    assert np.isfinite(lr.to_numpy()).all()
