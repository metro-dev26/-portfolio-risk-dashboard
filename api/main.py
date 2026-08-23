"""FastAPI transport over the pure risk engine. Handlers are thin: validate
(via pydantic) -> call the engine/store -> return a typed model. The auto-generated
Swagger UI at /docs is the point. Educational tool, not financial advice."""
import math

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api import store
from api.schemas import AnalyzeRequest, AnalyzeResponse, PortfolioIn, PortfolioOut
from risk_engine.analyze import analyze_portfolio

app = FastAPI(
    title="MarketPlug Risk API",
    description="Documented REST service over a tested portfolio-risk engine. "
                "Educational — not financial advice.",
    version="1.0.0",
)


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return analyze_portfolio(
        req.holdings, confidence=req.confidence, include_backtest=req.include_backtest
    )


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
        row["analysis"] = analyze_portfolio(row["holdings"])
    return row
