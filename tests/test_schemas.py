import pytest
from pydantic import ValidationError

from api.schemas import AnalyzeRequest, AnalyzeResponse
from risk_engine.analyze import analyze_portfolio

GOOD = {"AAPL": 20000, "MSFT": 20000, "JPM": 20000}


def test_good_request_defaults():
    req = AnalyzeRequest(holdings=GOOD)
    assert req.confidence == 0.95 and req.include_backtest is True


@pytest.mark.parametrize("holdings", [
    {"AAPL": 20000},                          # fewer than 2
    {"AAPL": 20000, "aapl": 5},               # two keys, one holding once merged
    {"AAPL": 20000, "MSFT": -5},              # non-positive amount
    {"AAPL": 20000, "AAPL/../X": 1},          # not a ticker format
    {"AAPL": 20000, "<script>": 1},           # not a ticker format
    {f"T{i}": 1 for i in range(51)},          # more than 50
])
def test_bad_holdings_rejected(holdings):
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=holdings)


def test_tickers_are_normalized_and_merged():
    req = AnalyzeRequest(holdings={"brk.b": 1000, "BRK-B": 500, "aapl": 10})
    assert req.holdings == {"BRK-B": 1500.0, "AAPL": 10.0}


def test_merged_amount_is_held_to_the_maximum():
    """Each spelling is under the cap; together they are not."""
    with pytest.raises(ValidationError, match="exceeds the maximum"):
        AnalyzeRequest(holdings={"brk.b": 4e11, "BRK-B": 4e11, "Brk-B": 4e11, "AAPL": 1})


def test_fifty_holdings_are_accepted():
    assert len(AnalyzeRequest(holdings={f"T{i}": 1 for i in range(50)}).holdings) == 50


@pytest.mark.parametrize("holdings", [
    {"AAPL": float("inf"), "MSFT": 20000},    # inf divides into a NaN weight
    {"AAPL": float("nan"), "MSFT": 20000},    # NaN propagates through the math
    {"AAPL": float("-inf"), "MSFT": 20000},   # -inf is non-finite and non-positive
    {"AAPL": 1e13, "MSFT": 20000},            # over the sane maximum
])
def test_non_finite_or_oversized_amounts_rejected(holdings):
    """Security: non-finite amounts previously passed `> 0` and produced a silent
    all-zeros 200 instead of a clean 422. Reject them at the schema."""
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=holdings)


@pytest.mark.parametrize("conf", [0.5, 0.995, 1.0])
def test_bad_confidence_rejected(conf):
    with pytest.raises(ValidationError):
        AnalyzeRequest(holdings=GOOD, confidence=conf)


def test_engine_output_satisfies_response_model():
    out = analyze_portfolio({"AAPL": 20000, "MSFT": 20000, "JPM": 20000})
    AnalyzeResponse(**out)   # ties engine output to the API's response schema


def test_response_model_accepts_omitted_backtest():
    """The Optional backtest path: include_backtest=False omits the key, and the
    response model must validate cleanly with backtest = None (not require it)."""
    out = analyze_portfolio(GOOD, include_backtest=False)
    assert "backtest" not in out
    assert AnalyzeResponse(**out).backtest is None


@pytest.mark.parametrize("conf", [0.90, 0.99])
def test_confidence_bounds_are_inclusive(conf):
    """ge/le bounds must be inclusive at the exact edges."""
    assert AnalyzeRequest(holdings=GOOD, confidence=conf).confidence == conf
