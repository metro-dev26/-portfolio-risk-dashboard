import json
import numpy as np
import pandas as pd
import pytest
from risk_engine.data import load_prices

HOLDINGS = ["AAPL", "MSFT", "JPM", "XOM", "TLT"]
CONF = 0.95


@pytest.fixture(scope="session")
def market():
    prices, lr, _ = load_prices(prefer_live=False)
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
