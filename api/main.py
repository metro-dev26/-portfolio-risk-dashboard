"""FastAPI transport over the pure risk engine. Handlers are thin: validate
(via pydantic) -> call the engine/store -> return a typed model. The auto-generated
Swagger UI at /docs is the point. Educational tool, not financial advice."""
import functools
import logging
import math
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api import store
from api.schemas import (AnalyzeRequest, AnalyzeResponse, DegradedResponse, ImportRequest,
                         ImportResponse, PortfolioIn, PortfolioOut, ProblemResponse,
                         ReportRequest, ReportResponse, UnavailableResponse, UniverseRow)
from risk_engine import data, importer
from risk_engine.analyze import analyze_portfolio
from risk_engine.inputs import PortfolioError
from risk_engine.report import build_report

log = logging.getLogger(__name__)

app = FastAPI(
    title="MarketPlug Risk API",
    description="Documented REST service over a tested portfolio-risk engine. "
                "Educational — not financial advice.",
    version="1.0.0",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _json_safe(obj):
    """Replace non-finite floats (inf/-inf/NaN) with their string form. A 422
    error echoes the offending input back; when that input is inf/NaN, Starlette's
    JSON encoder (allow_nan=False) raises and the clean 422 becomes a 500. Scrub
    the whole error payload so validation failures always serialize."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


@app.exception_handler(RequestValidationError)
async def _on_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422,
                        content={"detail": _json_safe(jsonable_encoder(exc.errors()))})


@app.get("/health", responses={
    503: {"model": DegradedResponse,
          "description": "No market-data snapshot could be read; the service cannot analyse anything."}})
def health():
    try:
        snap = data.load_snapshot()
    except data.SnapshotUnavailable as e:
        log.exception("health check: no market-data snapshot is readable")
        return JSONResponse(status_code=503, content={"status": "degraded", "detail": str(e)})
    return {"status": "ok", "data_as_of": snap.as_of, "stale": data.is_stale(snap),
            "data_source": snap.source}


@contextmanager
def _engine_refusals_as_http_errors():
    """The one mapping from the engine's refusals to HTTP: holdings it can't use are a 422
    with one problem per ticker, and no readable snapshot is a 503."""
    try:
        yield
    except PortfolioError as e:
        raise HTTPException(status_code=422, detail={"problems": e.problems})
    except data.SnapshotUnavailable as e:
        raise HTTPException(status_code=503, detail=f"market data is unavailable: {e}")


def _analyze_or_http_error(holdings, **kwargs):
    with _engine_refusals_as_http_errors():
        return analyze_portfolio(holdings, **kwargs)


@app.post("/analyze", response_model=AnalyzeResponse, responses={
    422: {"model": ProblemResponse,
          "description": "The holdings were refused. Request-schema failures return `detail` "
                         "as a list of validation errors; holdings the engine can't use return "
                         "`detail.problems`, one `{ticker, reason}` per ticker (a data-wide "
                         "problem is reported under the ticker `(factors)`)."},
    503: {"model": UnavailableResponse,
          "description": "No market-data snapshot could be read."}})
def analyze(req: AnalyzeRequest):
    return _analyze_or_http_error(req.holdings, confidence=req.confidence,
                                  include_backtest=req.include_backtest)


def _report_or_http_error(holdings, **kwargs):
    with _engine_refusals_as_http_errors():
        return build_report(holdings, **kwargs)


@app.post("/report", response_model=ReportResponse, responses={
    422: {"model": ProblemResponse,
          "description": "The holdings were refused: `detail.problems` gives one "
                         "`{ticker, reason}` per ticker."},
    503: {"model": UnavailableResponse, "description": "No market-data snapshot could be read."}})
def report(req: ReportRequest):
    """Everything the dashboard shows for one portfolio, in one call: the 1-99 risk score
    beside the S&P 500's, the bad-year dollar range, and every detail section."""
    return _report_or_http_error(req.holdings, confidence=req.confidence, shrinkage=req.shrinkage)


def _snapshot_or_http_error():
    with _engine_refusals_as_http_errors():
        return data.load_snapshot()


@app.get("/universe", response_model=list[UniverseRow], responses={
    503: {"model": UnavailableResponse, "description": "No market-data snapshot could be read."}})
def universe():
    """Every ticker in today's dataset, for search-as-you-type. Tickers outside it still
    work in /report through a live lookup."""
    snap = _snapshot_or_http_error()
    return [{"ticker": t, "name": m["name"], "sector": m["sector"],
             "asset_class": m["asset_class"], "in_sp500": bool(m.get("in_sp500", False))}
            for t, m in sorted(snap.universe.items())
            if t in snap.prices.columns and t not in snap.quarantined]


@app.post("/import", response_model=ImportResponse, responses={
    503: {"model": UnavailableResponse,
          "description": "A CSV of share counts needs the dataset's closing prices and none "
                         "could be read. Pasted text never needs them."}})
def import_holdings(req: ImportRequest):
    """Parse a broker CSV export or pasted 'TICKER AMOUNT' lines into holdings. Share
    counts are valued at the dataset's latest close; nothing is analysed here."""
    if req.format == "paste":
        parsed = importer.parse_paste(req.text)
    else:
        snap = _snapshot_or_http_error()

        @functools.cache  # a 200 KB file can repeat one symbol 25,000 times
        def last_price(sym):
            if sym not in snap.prices.columns:
                return None
            closes = snap.prices[sym].dropna()
            return float(closes.iloc[-1]) if len(closes) else None

        parsed = importer.parse_csv(req.text, last_price=last_price)
    return {"holdings": parsed.holdings, "notes": parsed.notes}


@app.post("/portfolios", response_model=PortfolioOut)
def create_portfolio(p: PortfolioIn):
    pid = store.save_portfolio(p.name, p.holdings)
    return store.get_portfolio(pid)


@app.get("/portfolios", response_model=list[PortfolioOut])
def get_portfolios():
    return store.list_portfolios()


@app.get("/portfolios/{pid}")
def get_portfolio(pid: int, analyze: bool = Query(False)):
    row = store.get_portfolio(pid)
    if row is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    if analyze:
        row["analysis"] = _analyze_or_http_error(row["holdings"])
    return row
