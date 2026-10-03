import json
import os
import numpy as np
import pandas as pd
import pytest

# Every test reads the frozen June 2026 bundle and never touches the network.
os.environ.setdefault("MARKETPLUG_DATA_DIR",
                      os.path.join(os.path.dirname(__file__), "fixtures", "snapshot"))
os.environ.setdefault("MARKETPLUG_NO_LIVE", "1")

HOLDINGS = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
CONF = 0.95


@pytest.fixture(scope="session")
def market():
    from risk_engine.data import load_snapshot
    prices = load_snapshot().prices.ffill().dropna()
    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr


@pytest.fixture(scope="session")
def ref_weights():
    return np.ones(len(HOLDINGS)) / len(HOLDINGS)


@pytest.fixture(scope="session")
def golden():
    with open("tests/fixtures/golden.json") as f:
        return json.load(f)


@pytest.fixture
def normal_returns():
    """A long, well-behaved normal return series with known parameters."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0, 0.01, 50_000))
