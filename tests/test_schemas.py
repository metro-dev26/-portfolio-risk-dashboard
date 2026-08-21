import pytest
from pydantic import ValidationError

from api.schemas import AnalyzeRequest, AnalyzeResponse
from risk_engine.analyze import analyze_portfolio

GOOD = {"AAPL": 20000, "MSFT": 20000, "JPM": 20000}


def test_good_request_defaults():
    req = AnalyzeRequest(holdings=GOOD)
    assert req.confidence == 0.95 and req.include_backtest is True


@pytest.mark.parametrize("holdings", [
    {"AAPL": 20000},                       # fewer than 2
    {"AAPL": 20000, "MSFT": -5},           # non-positive amount
    {"AAPL": 20000, "NOTREAL": 20000},     # unknown ticker
    {f"AAPL{i}": 1 for i in range(13)},    # more than 12 (and unknown)
])
def test_bad_holdings_rejected(holdings):
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=holdings)


@pytest.mark.parametrize("conf", [0.5, 0.995, 1.0])
def test_bad_confidence_rejected(conf):
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=GOOD, confidence=conf)


def test_engine_output_satisfies_response_model():
    out = analyze_portfolio({"AAPL": 20000, "MSFT": 20000, "JPM": 20000})
    AnalyzeResponse(**out)   # ties engine output to the API's response schema
