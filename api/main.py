"""FastAPI transport over the pure risk engine. Handlers are thin: validate
(via pydantic) -> call the engine/store -> return a typed model. The auto-generated
Swagger UI at /docs is the point. Educational tool, not financial advice."""
import functools
import itertools
import logging
import math
import os
import re
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api import limits, store
from api.schemas import (AnalyzeRequest, AnalyzeResponse, DegradedResponse, ImportRequest,
                         ImportResponse, PortfolioIn, PortfolioOut, ProblemResponse,
                         ReportRequest, ReportResponse, UnavailableResponse, UniverseRow)
from risk_engine import data, importer
from risk_engine.analyze import analyze_portfolio
from risk_engine.inputs import PortfolioError
from risk_engine.report import build_report

log = logging.getLogger(__name__)

_MAX_ECHO_CHARS = 200
_MAX_ECHO_ITEMS = 20
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")
_MAX_IMPORT_NOTES = 50
_MAX_IMPORT_BODY = 1_500_000      # 200,000 characters at the worst case of 6 bytes each (\uXXXX)
_MAX_BODY = 64 * 1024             # every other route takes at most 50 holdings or a short name

app = FastAPI(
    title="MarketPlug Risk API",
    description="Documented REST service over a tested portfolio-risk engine. "
                "Educational — not financial advice.",
    version="1.0.0",
)

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
                   if o.strip()]
limiter = limits.RateLimiter({"/report": 10, "/analyze": 10, "/import": 20}, default=60)
gate = limits.ComputeGate(slots=2, wait_s=15.0, max_waiting=8)
_TRUE_SPELLINGS = {"1", "true", "t", "yes", "y", "on"}    # what a bool query parameter accepts


def _rate_limit_path(request: Request):
    """The path whose allowance a request spends. Reading a saved portfolio with
    analyze=true runs a full analysis, so it spends /analyze's allowance, not the general one."""
    path = request.url.path
    if path.startswith("/portfolios/") and any(
            v.strip().lower() in _TRUE_SPELLINGS for v in request.query_params.getlist("analyze")):
        return "/analyze"
    return path


# Added first, so it runs innermost: an oversized body is refused before any route parses it,
# and the 413 travels out through the outer middleware like any other response.
app.add_middleware(limits.BodyLimit, caps={"/import": _MAX_IMPORT_BODY}, default=_MAX_BODY)


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if request.method != "OPTIONS":
        retry = limiter.check(limits.client_ip(request), _rate_limit_path(request))
        if retry is not None:
            return JSONResponse(status_code=429, content={"detail": "too many requests"},
                                headers={"Retry-After": str(max(1, math.ceil(retry)))})
    return await call_next(request)


app.add_middleware(GZipMiddleware, minimum_size=1000)
# Added last, so it runs outermost: even a 413, 429 or 503 carries CORS headers and the
# browser shows the real error instead of a CORS failure.
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], expose_headers=["Retry-After"],
                   allow_credentials=False)


def _echo_text(text):
    text = _LONE_SURROGATE.sub("\ufffd", text)
    return text if len(text) <= _MAX_ECHO_CHARS else text[:_MAX_ECHO_CHARS - 1] + "…"


def _json_safe(obj):
    """Make a 422 payload serializable and small. It echoes the offending input back,
    and four kinds of input turn the clean 422 into a 500 or an outsized response:
    inf/NaN (Starlette's JSON encoder has allow_nan=False) become their string form, a
    lone UTF-16 surrogate (not encodable as UTF-8) becomes U+FFFD, any string over
    _MAX_ECHO_CHARS is cut short, and any list or dict over _MAX_ECHO_ITEMS keeps its
    first _MAX_ECHO_ITEMS entries plus a count of the rest. Scrub the whole payload, keys
    included, so validation failures always serialize."""
    if isinstance(obj, str):
        return _echo_text(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if isinstance(obj, dict):
        kept = {_json_safe(k): _json_safe(v) for k, v in itertools.islice(obj.items(), _MAX_ECHO_ITEMS)}
        if len(obj) > _MAX_ECHO_ITEMS:
            kept["…"] = f"{len(obj) - _MAX_ECHO_ITEMS} more"
        return kept
    if isinstance(obj, list):
        kept = [_json_safe(v) for v in obj[:_MAX_ECHO_ITEMS]]
        if len(obj) > _MAX_ECHO_ITEMS:
            kept.append(f"…{len(obj) - _MAX_ECHO_ITEMS} more")
        return kept
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


def _gated(fn, *args, **kwargs):
    """Run a heavy computation in one of the compute gate's slots; a full gate is a 503."""
    try:
        with gate.slot():
            return fn(*args, **kwargs)
    except limits.Busy:
        raise HTTPException(status_code=503, detail="busy", headers={"Retry-After": "10"})


@app.post("/analyze", response_model=AnalyzeResponse, responses={
    422: {"model": ProblemResponse,
          "description": "The holdings were refused. Request-schema failures return `detail` "
                         "as a list of validation errors; holdings the engine can't use return "
                         "`detail.problems`, one `{ticker, reason}` per ticker (a data-wide "
                         "problem is reported under the ticker `(factors)`)."},
    503: {"model": UnavailableResponse,
          "description": "No market-data snapshot could be read, or the server is busy (`detail: busy`, "
                         "retry after the `Retry-After` seconds)."}})
def analyze(req: AnalyzeRequest):
    return _gated(_analyze_or_http_error, req.holdings, confidence=req.confidence,
                  include_backtest=req.include_backtest)


def _report_or_http_error(holdings, **kwargs):
    with _engine_refusals_as_http_errors():
        return build_report(holdings, **kwargs)


@app.post("/report", response_model=ReportResponse, responses={
    422: {"model": ProblemResponse,
          "description": "The holdings were refused: `detail.problems` gives one "
                         "`{ticker, reason}` per ticker."},
    503: {"model": UnavailableResponse,
          "description": "No market-data snapshot could be read, or the server is busy (`detail: busy`, "
                         "retry after the `Retry-After` seconds)."}})
def report(req: ReportRequest):
    """Everything the dashboard shows for one portfolio, in one call: the 1-99 risk score
    beside the S&P 500's, the bad-year dollar range, and every detail section."""
    return _gated(_report_or_http_error, req.holdings, confidence=req.confidence,
                  shrinkage=req.shrinkage)


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


def _capped_notes(notes):
    """A 200 KB paste of junk is 100,000 skipped lines. Keep the first _MAX_IMPORT_NOTES and
    say how many more there were."""
    if len(notes) <= _MAX_IMPORT_NOTES:
        return notes
    more = len(notes) - _MAX_IMPORT_NOTES
    return notes[:_MAX_IMPORT_NOTES] + [f"…and {more} more lines were skipped"]


@app.post("/import", response_model=ImportResponse, responses={
    503: {"model": UnavailableResponse,
          "description": "No market-data snapshot could be read. Every CSV import needs one, "
                         "to price share counts; pasted text does not."}})
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
    return {"holdings": parsed.holdings, "notes": _capped_notes(parsed.notes)}


@app.post("/portfolios", response_model=PortfolioOut)
def create_portfolio(p: PortfolioIn):
    pid = store.save_portfolio(p.name, p.holdings)
    return store.get_portfolio(pid)


@app.get("/portfolios/{pid}")
def get_portfolio(pid: str = Path(pattern=r"^[A-Za-z0-9_-]{1,64}$"), analyze: bool = Query(False)):
    row = store.get_portfolio(pid)
    if row is None:
        raise HTTPException(status_code=404, detail="portfolio not found")
    if analyze:
        row["analysis"] = _gated(_analyze_or_http_error, row["holdings"])
    return row
