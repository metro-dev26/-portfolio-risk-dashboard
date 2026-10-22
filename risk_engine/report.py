"""Every dashboard section for one portfolio, as one JSON-safe dict. Pure.

The API serves this dict; the Streamlit page lays it out. Both read the same
numbers, so the two frontends can never disagree. A section that can't be
computed (thin factor data, an optimizer that won't converge) comes back with
`available: false` and a reason instead of failing the whole report."""
import math

import numpy as np

from risk_engine import backtest, benchmark, data, metrics, optimize, simulate, stress
from risk_engine import factors as fac
from risk_engine import portfolio
from risk_engine.config import (BACKTEST_MIN_OBS, CONCENTRATION_LIMIT, FACTOR_MIN_OBS,
                                TRADING_DAYS)
from risk_engine.inputs import prepare

BACKTEST_WINDOW = 250
FRONTIER_POINTS = 40
DIVERSIFIER_MARGIN = 0.02     # risk share this far under capital share counts as a diversifier
REALLOC_THRESHOLD = 0.01      # weight change smaller than this is not worth suggesting

ASSUMPTIONS = [
    "Risk numbers come from this portfolio's own daily returns over the window shown; the "
    "future can differ from the past.",
    "The Monte Carlo replays real past days from 2018 onward, mostly a rising market, so the "
    "bad-year figure leans optimistic. The S&P 500 score replays the same days, so the "
    "comparison stays fair.",
    "Replaying days one at a time ignores that bad days come in clusters, so short-horizon "
    "losses are understated.",
    "Growth charts rebalance to today's weights every day; the stress test holds today's "
    "weights without rebalancing.",
    "Expected returns are average daily log returns times 252, slightly below the arithmetic "
    "average.",
    "Dollar figures convert log returns to real gains and losses with exp(r) - 1.",
    "Educational tool, not financial advice.",
]


def _sig(x, digits=6):
    return float(f"{float(x):.{digits}g}")


def _series(s):
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [_sig(v) for v in s.to_numpy()]}


def _corr_cell(x):
    """A holding with no price movement has no correlation: None, since JSON has no NaN."""
    return round(x, 3) if math.isfinite(x) else None


def _score(sc):
    return {"score": sc.score, "bad_year_loss": sc.bad_year_loss,
            "loss_dollars": sc.loss_dollars, "gain_dollars": sc.gain_dollars}


def _holdings(inp):
    out = []
    for t, w in zip(inp.tickers, inp.weights):
        m = inp.res.meta[t]
        out.append({"ticker": t, "name": m["name"], "sector": m["sector"], "type": m["type"],
                    "asset_class": m["asset_class"], "amount": float(inp.res.holdings[t]),
                    "weight": float(w), "source": inp.res.source[t]})
    return out


def _risk(inp, conf):
    hv, hc = metrics.historical_var_cvar(inp.pr, conf)
    gv, gc = metrics.gaussian_var_cvar(inp.pr, conf)
    fr = {"hist_var": hv, "hist_cvar": hc, "gaussian_var": gv, "gaussian_cvar": gc}
    dollars = {k: metrics.to_dollars(v, inp.value) for k, v in fr.items()}
    return {"confidence": conf, **fr, "dollars": dollars,
            "gap_dollars": dollars["hist_var"] - dollars["gaussian_var"]}


def _stress(inp):
    out = []
    for r in stress.replay(inp.res.prices, inp.tickers, inp.weights):
        ok = r.status == "ok"
        out.append({"label": r.label, "context": r.context, "start": r.start, "end": r.end,
                    "status": r.status, "missing": r.missing, "total": r.total,
                    "worst_day": r.worst_day, "max_dd": r.max_dd,
                    "change_dollars": r.total * inp.value if ok else None,
                    "path": _series(r.path / r.path.iloc[0] - 1.0) if ok else None})
    return out


def _backtest(inp, conf):
    rows = backtest.backtest_var(inp.pr, conf, BACKTEST_WINDOW)
    methods = {r["method"]: {"observations": int(r["observations"]), "breaches": int(r["breaches"]),
                             "expected": float(r["expected"]), "kupiec_p": float(r["kupiec_p"]),
                             "christoffersen_p": float(r["christoffersen_p"]),
                             "passed": bool(r["passed"]) if r["observations"] >= BACKTEST_MIN_OBS else None}
               for r in rows}
    s = next(r["series"] for r in rows if r["method"] == "historical")
    series = {"dates": [d.strftime("%Y-%m-%d") for d in s["dates"]],
              "var": [_sig(v) for v in s["var"]], "realized": [_sig(v) for v in s["realized"]],
              "breach": [int(b) for b in s["breach"]]}
    return {"window": BACKTEST_WINDOW, "min_observations": BACKTEST_MIN_OBS,
            "methods": methods, "series": series}


def _risk_contribution(inp):
    w = inp.weights
    share = optimize.risk_contribution(optimize.sample_cov(inp.lr, inp.tickers), w)
    hot, div = int(np.argmax(share - w)), int(np.argmin(share))
    return {"rows": [{"ticker": t, "weight": float(a), "risk_share": float(b)}
                     for t, a, b in zip(inp.tickers, w, share)],
            "biggest_driver": inp.tickers[hot],
            "diversifier": inp.tickers[div] if share[div] < w[div] - DIVERSIFIER_MARGIN else None}


def _factors(inp):
    snap = inp.snap
    through = str(snap.factors.index.max().date()) if len(snap.factors) else None
    shared = len(inp.pr.index.intersection(snap.factors.index))
    if shared < FACTOR_MIN_OBS:
        return {"available": False, "reason": fac.factor_gap_reason(snap.factors, shared, inp.window.n_days),
                "data_through": through, "betas": None, "alpha_annual": None, "r2": None,
                "variance_split": None, "observations": shared}
    reg = fac.factor_regression(inp.pr, snap.factors)
    return {"available": True, "reason": "", "data_through": through, "betas": reg["betas"],
            "alpha_annual": reg["alpha_annual"], "r2": reg["r2"],
            "variance_split": fac.variance_attribution(reg), "observations": reg["n_obs"]}


def _sector_mix(inp):
    mix = portfolio.sector_mix(inp.tickers, inp.weights, inp.res.meta)
    top = max(mix.sectors, key=mix.sectors.get) if mix.sectors else None
    top_pct = mix.sectors[top] if top else 0.0
    return {"sectors": mix.sectors, "equity": mix.equity, "diversified": mix.diversified,
            "unplaced": mix.unplaced, "non_equity": mix.non_equity, "top_sector": top,
            "top_sector_pct": top_pct, "concentrated": top_pct >= CONCENTRATION_LIMIT}


def _optimizer(inp, shrinkage):
    t, w, v = inp.tickers, inp.weights, inp.value
    mu = optimize.annualized_mean(inp.lr, t)
    cov = optimize.ledoit_wolf_cov(inp.lr, t) if shrinkage else optimize.sample_cov(inp.lr, t)
    base = {"shrinkage": shrinkage, "sector_mix": _sector_mix(inp)}
    try:
        w_ms = optimize.max_sharpe_weights(mu, cov)
        w_mv = optimize.min_variance_weights(cov)
    except optimize.OptimizerError as e:
        return {**base, "available": False, "reason": str(e), "current": None, "max_sharpe": None,
                "min_variance": None, "frontier": [], "reallocation": [], "adds": [], "trims": []}

    def point(weights):
        r, vol, s = optimize.perf(weights, mu, cov)
        return {"ret": r, "vol": vol, "sharpe": s,
                "weights": {k: float(x) for k, x in zip(t, weights)}}

    fr_v, fr_r = optimize.efficient_frontier(mu, cov, n_points=FRONTIER_POINTS)
    delta = w_ms - w
    order_up, order_down = np.argsort(delta)[::-1], np.argsort(delta)
    return {**base, "available": True, "reason": "", "current": point(w),
            "max_sharpe": point(w_ms), "min_variance": point(w_mv),
            "frontier": [{"vol": a / 100, "ret": b / 100} for a, b in zip(fr_v, fr_r)],
            "reallocation": [{"ticker": k, "asset_class": inp.res.meta[k]["asset_class"],
                              "sector": inp.res.meta[k]["sector"], "current": float(a * v),
                              "suggested": float(b * v), "delta_dollars": float((b - a) * v)}
                             for k, a, b in zip(t, w, w_ms)],
            "adds": [t[i] for i in order_up if delta[i] > REALLOC_THRESHOLD][:2],
            "trims": [t[i] for i in order_down if delta[i] < -REALLOC_THRESHOLD][:2]}


def _monte_carlo(sim):
    by_days = {d: label for label, d in simulate.HORIZONS.items()}
    return {"n_sims": simulate.N_SIMS, "seed": simulate.SEED, "value": sim.value,
            "band_days": sim.band_days,
            "bands": {f"p{p}": [_sig(x) for x in vals] for p, vals in sim.bands.items()},
            "horizons": [{"label": by_days[d], "days": d, **h} for d, h in sim.horizons.items()]}


def report_from_inputs(inp, *, confidence=0.95, shrinkage=False):
    if not inp.pr.index.equals(inp.spy_r.index):
        raise ValueError("portfolio and S&P 500 returns must share the same dates: the "
                         "Monte Carlo replays one set of days through both")
    draws = simulate.draw_indices(len(inp.pr))
    sim = simulate.simulate(inp.pr.to_numpy(), inp.value, draws)
    bench = simulate.simulate(inp.spy_r.to_numpy(), inp.value, draws)
    del draws
    cmp_ = benchmark.compare(inp.pr, inp.spy_r, inp.value)
    corr = metrics.correlation(inp.lr, inp.tickers)
    win = inp.window
    return {
        "score": {"portfolio": _score(simulate.risk_score(sim)),
                  "benchmark": _score(simulate.risk_score(bench)),
                  "horizon_days": simulate.SCORE_HORIZON},
        "portfolio": {"value": inp.value, "holdings": _holdings(inp)},
        "risk": _risk(inp, confidence),
        "profile": {"annual_return": float(inp.pr.mean() * TRADING_DAYS),
                    "annual_vol": metrics.annualized_vol(inp.pr),
                    "sharpe": metrics.sharpe_ratio(inp.pr),
                    "max_drawdown": metrics.max_drawdown(inp.pr),
                    "trading_days": int(len(inp.pr))},
        "benchmark": {"rows": cmp_.rows, "beta": cmp_.beta,
                      "growth": {"portfolio": _series(cmp_.growth_portfolio),
                                 "benchmark": _series(cmp_.growth_benchmark)}},
        "distribution": {**metrics.return_histogram(inp.pr), "mean": float(inp.pr.mean()),
                         "std": float(inp.pr.std())},
        "drawdown": _series(metrics.drawdown_series(inp.pr)),
        "stress": _stress(inp),
        "correlation": {"tickers": corr["tickers"],
                        "matrix": [[_corr_cell(x) for x in row] for row in corr["matrix"]]},
        "backtest": _backtest(inp, confidence),
        "risk_contribution": _risk_contribution(inp),
        "factors": _factors(inp),
        "optimizer": _optimizer(inp, shrinkage),
        "monte_carlo": _monte_carlo(sim),
        "data": {"as_of": inp.snap.as_of, "window_start": str(win.start.date()),
                 "window_end": str(win.end.date()), "window_days": int(win.n_days),
                 "window_status": win.status, "window_message": win.message,
                 "rejected": [{"ticker": r.ticker, "amount": r.amount, "reason": r.reason}
                              for r in inp.res.rejected],
                 "assumptions": ASSUMPTIONS},
    }


def build_report(holdings, *, confidence=0.95, shrinkage=False, snapshot=None, live_fetch=None):
    snap = snapshot or data.load_snapshot()
    return report_from_inputs(prepare(holdings, snap, live_fetch=live_fetch),
                              confidence=confidence, shrinkage=shrinkage)
