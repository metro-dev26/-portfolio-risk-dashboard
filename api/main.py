"""FastAPI transport over the pure risk engine. Handlers are thin: validate
(via pydantic) -> call the engine/store -> return a typed model. The auto-generated
Swagger UI at /docs is the point. Educational tool, not financial advice."""
from fastapi import FastAPI, HTTPException, Query

from api import store
from api.schemas import AnalyzeRequest, AnalyzeResponse, PortfolioIn, PortfolioOut
from risk_engine.analyze import analyze_portfolio

app = FastAPI(
    title="MarketPlug Risk API",
    description="Documented REST service over a tested portfolio-risk engine. "
                "Educational — not financial advice.",
    version="1.0.0",
)


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
