import numpy as np
from risk_engine.factors import load_factors, FACTOR_NAMES


def test_factors_load_shape_and_range():
    f = load_factors()
    assert list(f.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert f.index.min().year <= 2018
    assert f.index.max().year >= 2026
    assert len(f) > 1500


def test_factor_values_are_decimals_not_percent():
    f = load_factors()
    # daily factor returns as decimals are tiny; if still in percent they'd exceed 1 often
    assert f[FACTOR_NAMES].abs().to_numpy().max() < 0.5
    assert np.isfinite(f.to_numpy()).all()
