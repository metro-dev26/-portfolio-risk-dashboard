"""
Portfolio Risk Dashboard
========================
A focused tool for measuring portfolio downside risk the way professionals do —
with real market data, historical (non-Gaussian) tail risk, and the honest gap
between what standard models assume and what the market actually does.

Every number on this page is computed from end-of-day market data refreshed
each US trading day. Nothing is hardcoded.
"""

import html
import time
import warnings
warnings.filterwarnings("ignore")

# Use the operating system's certificate store for SSL. Fixes local machines
# behind antivirus / corporate-network HTTPS inspection; harmless in the cloud.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:   # truststore is optional; a failed injection still surfaces
    pass

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from risk_engine import data, importer, inputs, report, stress
from risk_engine.config import LIVE_BUDGET_UI_S, LIVE_MAX_UI, MAX_HOLDINGS

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── STYLING ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:#05080f; --surface:#0c1220; --card:#111d2e; --border:#1c2d44;
    --green:#05d69e; --red:#ff3d5a; --blue:#4d9fff; --amber:#ffb347;
    --text:#dde4f0; --muted:#5a7088;
}
*, body, html { box-sizing: border-box; }
.stApp { background: var(--bg) !important; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--text); background:var(--bg); }
section[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--border); }
#MainMenu, footer { visibility:hidden; }
header { background:transparent !important; }
/* Keep the sidebar open/collapse control visible so the panel can always be reopened */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
    visibility:visible !important; display:flex !important; z-index:999999; }

.hero {
    background:linear-gradient(135deg,#071428 0%,#0a1e35 60%,#071428 100%);
    border:1px solid var(--border); border-radius:16px;
    padding:40px 48px; margin-bottom:28px; position:relative; overflow:hidden;
}
.hero::after {
    content:''; position:absolute; bottom:-60px; right:-60px;
    width:250px; height:250px; border-radius:50%;
    background:radial-gradient(circle,rgba(77,159,255,0.06),transparent 70%);
}
.hero-eyebrow { font-family:'DM Mono',monospace; font-size:11px; letter-spacing:0.2em;
    text-transform:uppercase; color:var(--blue); margin-bottom:14px; }
.hero-title { font-family:'Syne',sans-serif; font-size:38px; font-weight:800;
    line-height:1.1; margin-bottom:12px;
    background:linear-gradient(135deg,#dde4f0 30%,#4d9fff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-desc { font-size:15px; color:#6a849e; line-height:1.6; max-width:640px; }

.sec { font-family:'DM Mono',monospace; font-size:10px; letter-spacing:0.2em;
    text-transform:uppercase; color:var(--muted);
    border-top:1px solid var(--border); padding-top:12px; margin:28px 0 16px 0; }

.big-stat { background:var(--card); border:1px solid var(--border);
    border-radius:10px; padding:18px 20px; margin:6px 0; text-align:center; }
.big-stat-num { font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
    line-height:1; margin:6px 0; }
.big-stat-label { font-family:'DM Mono',monospace; font-size:10px;
    letter-spacing:0.15em; text-transform:uppercase; color:var(--muted); }
.big-stat-sub { font-size:12px; color:var(--muted); margin-top:4px; }

.insight { display:flex; gap:14px; align-items:flex-start;
    background:rgba(77,159,255,0.05); border:1px solid rgba(77,159,255,0.15);
    border-radius:8px; padding:14px 18px; margin:10px 0; }
.insight-icon { font-size:18px; flex-shrink:0; }
.insight-text { font-size:13px; line-height:1.6; color:#8a9bb8; }
.insight-text strong { color:var(--text); }
.insight.warn { background:rgba(255,179,71,0.05); border-color:rgba(255,179,71,0.2); }
.insight.danger { background:rgba(255,61,90,0.05); border-color:rgba(255,61,90,0.2); }

/* Dark-theme st.code so it sits in the palette instead of a light box */
[data-testid="stCode"] { background:transparent !important; }
[data-testid="stCode"] pre {
    background:var(--card) !important; border:1px solid var(--border) !important;
    border-radius:8px !important; }
[data-testid="stCode"] pre, [data-testid="stCode"] code, [data-testid="stCode"] code * {
    color:#c9d5e8 !important; background:transparent !important;
    font-family:'DM Mono',monospace !important; }
</style>
""", unsafe_allow_html=True)

# ── PAGE TOGGLE: Dashboard vs Beginner's Guide ────────────────
mode = st.radio("view", ["📊 Dashboard", "📖 Beginner's Guide", "🔌 API"],
                horizontal=True, label_visibility="collapsed")

if mode == "📖 Beginner's Guide":
    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>Start here · no finance background needed</div>
        <div class='hero-title'>What this tool does — in plain English</div>
        <div class='hero-desc'>
            Imagine you've put money into a handful of stocks and bonds. Two questions keep you up
            at night: <em>how much could I lose if things go bad?</em> and <em>am I being smart about
            how I've spread my money?</em> This tool answers both — using real market history, not guesses.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>The ideas, explained like you're new</div>", unsafe_allow_html=True)

    GUIDE = [
        ("📉", "Value at Risk (VaR)",
         "On a normal bad day, this is roughly the most you'd expect to lose. Think of it as "
         "\"95% of days, I won't lose more than this much.\" It's a line in the sand for ordinary rough days."),
        ("🌊", "CVaR (Expected Shortfall)",
         "VaR tells you the line. CVaR tells you how bad it gets <em>when you cross it</em> — the average "
         "loss on your worst days. It answers the question that actually matters: when it goes wrong, how wrong?"),
        ("🔔", "The Gaussian gap",
         "Textbooks assume losses follow a neat bell curve. Real markets crash harder and more often than "
         "that. This tool shows you, in dollars, how much risk the textbook quietly ignores."),
        ("⚖️", "Sharpe ratio",
         "Return alone is meaningless without risk. Sharpe is your reward per unit of risk taken. "
         "Higher is better — it means you're being paid well for the bumps you endure. Above 1 is good."),
        ("🤿", "Maximum drawdown",
         "The worst peak-to-valley fall your portfolio ever took. It's the gut-check number: "
         "\"how far underwater did I go, and could I have stomached it without panic-selling?\""),
        ("🔗", "Correlation",
         "Whether your holdings move together or apart. If everything you own rises and falls in sync, "
         "you're not really diversified — you just own one big bet wearing five different hats."),
        ("🔥", "Stress testing",
         "Instead of trusting averages, we replay real disasters — the COVID crash, the 2022 bear market — "
         "and show what <em>your exact portfolio</em> would have suffered. History as a pressure test."),
        ("🎯", "Risk contribution",
         "Surprise: a holding can be 20% of your money but 35% of your risk. This splits your total risk by "
         "who's really driving it. Sometimes a quiet bond is secretly protecting you; sometimes one stock is the storm."),
        ("🧭", "The optimizer (efficient frontier)",
         "For any level of risk, there's a 'best possible' mix that squeezes out the most return. The curve "
         "shows it. If your portfolio sits below the curve, you're taking risk you aren't being paid for."),
        ("🔮", "Monte Carlo simulation",
         "We can't predict the future, so we simulate 10,000 of them by reshuffling real history. The result "
         "is an honest range: 'most likely you land here, but 1 year in 20 it could be this bad.'"),
        ("🏛️", "Benchmark & Beta",
         "Your numbers mean more next to the market (the S&P 500). Beta tells you how wild your portfolio is "
         "vs the market: below 1 means calmer, above 1 means rowdier. It's your speedometer against the index."),
    ]
    for icon, title, body in GUIDE:
        st.markdown(f"""
        <div class='insight'>
            <div class='insight-icon'>{icon}</div>
            <div class='insight-text'><strong>{title}.</strong> {body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>How to use it — three steps</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight'>
        <div class='insight-icon'>1️⃣</div>
        <div class='insight-text'><strong>Build your portfolio.</strong> Switch to the Dashboard tab. In the
        left panel, paste, upload or search for the stocks and funds you hold (or want to test),
        with the dollar amount in each.</div>
    </div>
    <div class='insight'>
        <div class='insight-icon'>2️⃣</div>
        <div class='insight-text'><strong>Read your risk.</strong> The top of the page shows, in real dollars,
        how much a bad day could cost you and how that compares to the market.</div>
    </div>
    <div class='insight'>
        <div class='insight-icon'>3️⃣</div>
        <div class='insight-text'><strong>See how to improve.</strong> Scroll to the optimizer for a suggested
        rebalance, and the simulation for a realistic range of where your money could end up.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight warn'>
        <div class='insight-icon'>🔍</div>
        <div class='insight-text'><strong>One honest note.</strong> This is an educational tool for understanding
        how risk is measured — not financial advice. Every number is built from past data, and the past is a
        guide to the future, never a promise. Knowing that limit is itself the most professional habit in finance.</div>
    </div>""", unsafe_allow_html=True)

    st.stop()

if mode == "🔌 API":
    st.markdown("""
    <div class='hero'>
        <div class='hero-eyebrow'>For developers · a public REST API</div>
        <div class='hero-title'>This risk engine is also an API</div>
        <div class='hero-desc'>
            The same math this dashboard runs is exposed as a public REST API — JSON for other
            <em>programs</em>, not a webpage. One HTTP call returns risk metrics, factor exposures,
            and an optimized allocation. Drop it into a script, a notebook, or your own app.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>What the API does</div>", unsafe_allow_html=True)
    ENDPOINTS = [
        ("📊", "POST /analyze",
         "Send your holdings in dollars — get back VaR &amp; CVaR, Sharpe, max drawdown, "
         "Fama-French factor betas, VaR backtests, and an optimized max-Sharpe mix. One call, the whole engine."),
        ("💾", "POST /portfolios · GET /portfolios/{id}",
         "Save a named portfolio and get back a private id. <code>GET /portfolios/{id}</code> "
         "fetches it; ids are random, and there is no way to list other people's portfolios."),
        ("❤️", "GET /health",
         "A liveness check — returns <code>{\"status\":\"ok\"}</code> when the service is awake."),
    ]
    for icon, title, body in ENDPOINTS:
        st.markdown(f"""
        <div class='insight'>
            <div class='insight-icon'>{icon}</div>
            <div class='insight-text'><strong>{title}.</strong> {body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Example request · POST /analyze</div>", unsafe_allow_html=True)
    st.code('{\n  "holdings": {"AAPL": 10000, "MSFT": 8000, "TLT": 5000}\n}', language="json")

    st.markdown("<div class='sec'>Sample response · computed 2026-08-23</div>", unsafe_allow_html=True)
    _cols = st.columns(4)
    _cards = [
        ("Hist. VaR · 1-day 95%", "-2.12%", "var(--red)", "worst ordinary day"),
        ("Sharpe", "0.78", "var(--text)", "reward per unit of risk"),
        ("Max drawdown", "-31.5%", "var(--red)", "worst peak-to-valley fall"),
        ("Market beta", "0.88", "var(--text)", "Fama-French factor fit"),
    ]
    for _col, (_label, _num, _color, _sub) in zip(_cols, _cards):
        _col.markdown(f"""
        <div class='big-stat'>
            <div class='big-stat-label'>{_label}</div>
            <div class='big-stat-num' style='color:{_color}'>{_num}</div>
            <div class='big-stat-sub'>{_sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight'>
        <div class='insight-icon'>🧭</div>
        <div class='insight-text'><strong>What the optimizer computes.</strong> For this input it finds the
        risk-efficient max-Sharpe mix — <strong>AAPL 71% · MSFT 29% · TLT ~0%</strong> — lifting Sharpe from
        0.78 to <strong>0.80</strong>. It shows the most efficient blend of <em>these exact holdings</em> and
        how to read it. It is not a prediction, and not a recommendation to buy.</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec'>Try it live</div>", unsafe_allow_html=True)
    st.link_button("Open the interactive API docs →",
                   "https://marketplug-risk-api.onrender.com/docs")
    st.markdown("<div class='big-stat-sub' style='margin-top:2px'>"
                "Free-tier API — the first request may take ~40s to wake the server.</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='insight warn'>
        <div class='insight-icon'>🔍</div>
        <div class='insight-text'><strong>One honest note.</strong> This is an educational tool for understanding
        how risk is measured — not financial advice. Every number is built from past market data, and live values
        drift as the data snapshot updates. The past is a guide, never a promise.</div>
    </div>""", unsafe_allow_html=True)

    st.stop()

# ── DATA ──────────────────────────────────────────────────────
try:
    snap = data.load_snapshot()
except data.SnapshotUnavailable as e:
    st.error(f"Market data is unavailable right now ({e}). Please try again later.")
    st.stop()


class _Transient(Exception):
    """A lookup that failed for a reason worth retrying; raised so the cache never keeps it."""

    def __init__(self, result):
        super().__init__(result.reason)
        self.result = result


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _live_settled(sym):
    result = data.fetch_live(sym)
    if result.status == "unavailable":
        raise _Transient(result)
    return result


@st.cache_data(ttl=60, show_spinner=False)
def _live(sym):
    """Yahoo is asked at most once a minute for a ticker it could not answer, and
    answers it did give are kept for six hours."""
    try:
        return _live_settled(sym)
    except _Transient as transient:
        return transient.result


def _csv_pricer():
    """Values share counts at the latest close. Tickers outside the dataset need a live
    lookup, so one CSV gets the same cap and time budget as the holdings table."""
    started, looked_up = time.monotonic(), set()

    def last_price(sym):
        if sym in snap.prices.columns:
            series = snap.prices[sym].dropna()
            return float(series.iloc[-1]) if len(series) else None
        if sym not in looked_up:
            if len(looked_up) >= LIVE_MAX_UI or time.monotonic() - started > LIVE_BUDGET_UI_S:
                return None
            looked_up.add(sym)
        result = _live(sym)
        return float(result.prices.iloc[-1]) if result.status == "ok" else None

    return last_price


CONFIDENCE_LEVELS = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
CSV_MAX_BYTES = 2 * 1024 * 1024
NON_EQUITY_LABEL = {"Bond": "bonds", "Commodity": "commodities"}
EXAMPLE = {"AAPL": 20000.0, "MSFT": 20000.0, "JPM": 20000.0, "XOM": 20000.0, "TLT": 20000.0}

if "holdings" not in st.session_state:
    _shared = importer.decode_share(st.query_params.get("p", ""))
    st.session_state.holdings = _shared.holdings or dict(EXAMPLE)
    st.session_state.is_example = not _shared.holdings
    st.session_state.editor_v = 0
    st.session_state.import_notes = _shared.notes


def _replace_holdings(parsed):
    st.session_state.holdings = parsed.holdings
    st.session_state.is_example = False
    st.session_state.editor_v += 1          # a fresh key makes the table show the new rows
    st.session_state.import_notes = parsed.notes


# ── SIDEBAR: build the portfolio ──────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:DM Mono;font-size:9px;letter-spacing:0.2em;
                color:#4d9fff;text-transform:uppercase;margin-bottom:4px;'>Portfolio Risk Dashboard</div>
    <div style='font-family:Syne;font-size:22px;font-weight:800;margin-bottom:20px;'>Build Your Portfolio</div>
    """, unsafe_allow_html=True)

    tab_paste, tab_csv, tab_search = st.tabs(["Paste", "Upload CSV", "Search"])
    with tab_paste:
        _txt = st.text_area("One holding per line: ticker, then dollars",
                            placeholder="AAPL 10000\nVOO, $25,000\nmsft 8,500", key="paste_box")
        if st.button("Load pasted holdings", key="load_paste"):
            _replace_holdings(importer.parse_paste(_txt))
    with tab_csv:
        _up = st.file_uploader("Broker export (.csv)", type="csv", key="csv_up")
        if _up is not None and _up.size > CSV_MAX_BYTES:
            st.error(f"That file is {_up.size / 1048576:.1f} MB; the limit is "
                     f"{CSV_MAX_BYTES // 1048576} MB. A holdings export is far smaller, so "
                     f"check it is the right file.")
        elif _up is not None and st.button("Load CSV", key="load_csv"):
            _replace_holdings(importer.parse_csv(
                _up.getvalue().decode("utf-8-sig", errors="replace"), last_price=_csv_pricer()))
    with tab_search:
        _pick = st.selectbox("Find a ticker", sorted(snap.universe), index=None,
                             format_func=lambda t: f"{t} — {snap.universe[t]['name']}",
                             key="search_pick")
        _other = st.text_input("…or type any US-listed ticker", key="search_other")
        _amt = st.number_input("Amount ($)", min_value=0.0, value=10000.0, step=1000.0,
                               key="search_amt")
        if st.button("Add holding", key="add_holding"):
            _sym = (_other or _pick or "").strip()
            if _sym:
                _base = {} if st.session_state.is_example else dict(
                    st.session_state.get("table_holdings", st.session_state.holdings))
                _replace_holdings(importer.from_rows(list(_base.items()) + [(_sym, _amt)]))

    _example_note = st.empty()
    st.caption("Amounts are US dollars (market value), not share counts.")
    _h = st.session_state.holdings
    _edited = st.data_editor(
        pd.DataFrame({"Ticker": pd.Series(list(_h), dtype="object"),
                      "Amount $": pd.Series(list(_h.values()), dtype="float64")}),
        num_rows="dynamic", hide_index=True, width="stretch",
        key=f"amounts_{st.session_state.editor_v}",
        column_config={"Amount $": st.column_config.NumberColumn(
            min_value=0.0, step=1000.0, format="$%d")},
    )
    st.caption("The page address now contains these holdings and amounts — share it only with "
               "people who should see them.")

    conf = CONFIDENCE_LEVELS[st.select_slider(
        "Confidence level", options=list(CONFIDENCE_LEVELS), value="95%")]
    use_shrinkage = st.checkbox(
        "Use Ledoit-Wolf shrinkage (robust covariance)", value=False,
        help="Shrinks the noisy sample covariance toward a stable target so the "
             "optimizer stops chasing estimation error. Standard practice on real desks.")

_parsed = importer.from_rows(list(zip(_edited["Ticker"], _edited["Amount $"])))
st.session_state.table_holdings = _parsed.holdings
if _parsed.holdings != EXAMPLE:
    st.session_state.is_example = False
if st.session_state.is_example:
    _example_note.caption("Showing an example portfolio — paste, upload or search to use your own.")
_notes = st.session_state.import_notes + _parsed.notes
if len(_parsed.holdings) > MAX_HOLDINGS:
    st.error(f"This tool analyses up to {MAX_HOLDINGS} holdings; the table has "
             f"{len(_parsed.holdings)}. Combine or remove some.")
    st.stop()


class _Unsettled(Exception):
    """A report that left out a ticker for a reason a retry could fix; raised so the cache
    never keeps it."""

    def __init__(self, result):
        super().__init__("a lookup failed for now")
        self.result = result


@st.cache_data(ttl=3600, show_spinner=False)
def _settled_report(holdings_items, conf, shrinkage, as_of, source, _snap, _live_fetch):
    """One report per (holdings, settings, data version). The underscore arguments are
    not hashed: the snapshot is identified by as_of + source."""
    inp = inputs.prepare(dict(holdings_items), _snap, live_fetch=_live_fetch,
                         max_live=LIVE_MAX_UI, budget_s=LIVE_BUDGET_UI_S, allow_rejected=True)
    built = report.report_from_inputs(inp, confidence=conf, shrinkage=shrinkage)
    if any(r.retryable for r in inp.res.rejected):
        raise _Unsettled(built)
    return built


def _build_report(holdings_items, conf, shrinkage, snap, live_fetch):
    """A report that left a ticker out because Yahoo failed is rebuilt on the next rerun, so
    the lookup is tried again as soon as `_live` lets it (a minute), not an hour from now."""
    try:
        return _settled_report(holdings_items, conf, shrinkage, snap.as_of, snap.source,
                               snap, live_fetch)
    except _Unsettled as unsettled:
        return unsettled.result


with st.spinner("Looking up tickers and running the numbers..."):
    try:
        rep = _build_report(tuple(_parsed.holdings.items()), conf, use_shrinkage, snap, _live)
    except inputs.PortfolioError as e:
        rep, _error = None, e
_rejected = rep["data"]["rejected"] if rep is not None else _error.rejected

if _rejected or _notes:
    _lists = ""
    if _rejected:
        _lists += "<strong>Not included</strong><ul>" + "".join(
            f"<li><b>{html.escape(r['ticker'])}</b> — {html.escape(r['reason'])} · "
            f"${r['amount']:,.0f} excluded</li>" for r in _rejected) + "</ul>"
    if _notes:
        _lists += "<strong>Import notes</strong><ul>" + "".join(
            f"<li>{html.escape(n)}</li>" for n in _notes) + "</ul>"
    st.sidebar.markdown(f"<div class='insight warn'><div class='insight-text'>{_lists}</div></div>",
                        unsafe_allow_html=True)

if snap.warning:
    st.warning(snap.warning)
if data.is_stale(snap):
    st.warning(f"These prices are from {snap.as_of} — the daily refresh looks behind, so the "
               f"numbers below are not current.")

if rep is None:
    if _error.problems[0]["ticker"] == inputs.HOLDINGS_PROBLEM:
        st.warning("Add at least 2 holdings with an amount above $0 to build a portfolio.")
    else:
        st.error(" ".join(p["reason"] for p in _error.problems))
    st.stop()
if rep["data"]["window_status"] == "short":
    st.warning(rep["data"]["window_message"])

P = rep["portfolio"]
selected = [h["ticker"] for h in P["holdings"]]
weights = np.array([h["weight"] for h in P["holdings"]])
port_val = P["value"]
SECTOR = {h["ticker"]: h["sector"] for h in P["holdings"]}
_link = importer.encode_share({h["ticker"]: h["amount"] for h in P["holdings"]})
if st.query_params.get("p") != _link:
    st.query_params["p"] = _link
R, PROF = rep["risk"], rep["profile"]

# ── HERO ──────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
    <div class='hero-eyebrow'>{html.escape(data.freshness_line(snap))} · {PROF["trading_days"]:,} trading days analysed</div>
    <div class='hero-title'>Portfolio Risk Dashboard</div>
    <div class='hero-desc'>
        How much can this portfolio lose on a bad day — and how much of that risk does a
        standard Gaussian model quietly miss? Every figure below is computed from end-of-day
        market data for your {len(selected)} holdings.
    </div>
</div>""", unsafe_allow_html=True)

# ── HOW TO USE (always visible, even if sidebar is collapsed) ─
st.markdown("""
<div class='insight'>
    <div class='insight-icon'>👈</div>
    <div class='insight-text'>
        <strong>This dashboard is interactive — build your own portfolio.</strong>
        All controls are in the panel on the left: paste, upload or search for holdings, set the
        dollar amount in each, and choose the confidence level. Every number and chart below recomputes instantly.
        <strong>Don't see the panel?</strong> Click the <strong>›</strong> arrow at the very
        top-left of the page to open it.
    </div>
</div>""", unsafe_allow_html=True)

# ── HEADLINE METRICS ──────────────────────────────────────────
st.markdown("<div class='sec'>Daily downside risk · ${:,.0f} portfolio · {}% confidence</div>".format(
    port_val, int(conf * 100)), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Historical CVaR</div>
        <div class='big-stat-num' style='color:#ff3d5a;'>${abs(R["dollars"]["hist_cvar"]):,.0f}</div>
        <div class='big-stat-sub'>average loss on the worst {int(round((1 - conf) * 100))}% of days</div>
        </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Historical VaR</div>
        <div class='big-stat-num' style='color:#ff8a5a;'>${abs(R["dollars"]["hist_var"]):,.0f}</div>
        <div class='big-stat-sub'>the loss threshold · {abs(R["hist_var"])*100:.2f}% of portfolio</div>
        </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Gaussian VaR</div>
        <div class='big-stat-num' style='color:#4d9fff;'>${abs(R["dollars"]["gaussian_var"]):,.0f}</div>
        <div class='big-stat-sub'>what a standard model reports</div>
        </div>""", unsafe_allow_html=True)
gap = R["gap_dollars"]
with c4:
    gap_color = "#ff3d5a" if gap < 0 else "#05d69e"
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Hidden Risk Gap</div>
        <div class='big-stat-num' style='color:{gap_color};'>${abs(gap):,.0f}</div>
        <div class='big-stat-sub'>what Gaussian misses each day</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='insight {"danger" if gap < 0 else "warn"}'>
    <div class='insight-icon'>{"⚠️" if gap < 0 else "💡"}</div>
    <div class='insight-text'>
        <strong>CVaR is the honest headline.</strong> VaR only tells you the threshold
        ("{int(conf*100)}% of days you won't lose more than ${abs(R["dollars"]["hist_var"]):,.0f}"). CVaR answers the
        question that actually matters — <em>when it goes bad, how bad?</em> — at ${abs(R["dollars"]["hist_cvar"]):,.0f}.
        The Gaussian model here {"understates" if gap < 0 else "is close to"} the real loss threshold by
        ${abs(gap):,.0f} a day, because it assumes returns follow a bell curve and the market doesn't.
    </div>
</div>""", unsafe_allow_html=True)

# ── SUPPORTING METRICS ────────────────────────────────────────
st.markdown("<div class='sec'>Portfolio profile (analysis window)</div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Annualized Volatility</div>
        <div class='big-stat-num' style='color:#4d9fff;'>{PROF["annual_vol"]*100:.1f}%</div>
        <div class='big-stat-sub'>standard deviation of returns</div>
        </div>""", unsafe_allow_html=True)
with s2:
    sharpe = PROF["sharpe"]
    sh_color = "#05d69e" if sharpe >= 1 else "#ffb347" if sharpe >= 0 else "#ff3d5a"
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Sharpe Ratio</div>
        <div class='big-stat-num' style='color:{sh_color};'>{sharpe:.2f}</div>
        <div class='big-stat-sub'>return per unit of risk (rf = 0)</div>
        </div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Maximum Drawdown</div>
        <div class='big-stat-num' style='color:#ff3d5a;'>{PROF["max_drawdown"]*100:.1f}%</div>
        <div class='big-stat-sub'>worst peak-to-trough fall</div>
        </div>""", unsafe_allow_html=True)

# ── BENCHMARK vs S&P 500 ──────────────────────────────────────
st.markdown("<div class='sec'>Benchmark — your portfolio vs. the S&P 500</div>",
            unsafe_allow_html=True)
st.caption("Every number above is more meaningful next to the market. The S&P 500 (SPY) is the "
           "yardstick professionals measure against.")


def _vs(p, b, higher_better=True, pct=True, dec=1):
    better = (p > b) if higher_better else (p < b)
    col = "#05d69e" if better else "#ff8a5a"
    fmt = (f"{p*100:.{dec}f}%" if pct else f"{p:.2f}")
    return col, fmt


cmp_rows = [(r["label"], r["portfolio"], r["benchmark"], r["higher_is_better"],
             r["kind"] == "pct", 1 if r["kind"] == "pct" else 2)
            for r in rep["benchmark"]["rows"]]
rows_html = ""
for label, p, b, hb, pct, dec in cmp_rows:
    col, pf = _vs(p, b, hb, pct, dec)
    bf = (f"{b*100:.{dec}f}%" if pct else f"{b:.2f}")
    rows_html += (
        f"<tr>"
        f"<td style='padding:10px 14px;color:#8a9bb8;'>{label}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;font-weight:600;color:{col};'>{pf}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;color:#6a849e;'>{bf}</td>"
        f"</tr>")
st.markdown(f"""
<table style='width:100%;border-collapse:collapse;background:var(--card);
              border:1px solid var(--border);border-radius:10px;overflow:hidden;'>
    <tr style='background:#0c1220;'>
        <th style='padding:10px 14px;text-align:left;font-family:DM Mono;font-size:10px;
                   letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Metric</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;
                   letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Your Portfolio</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;
                   letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>S&amp;P 500</th>
    </tr>
    {rows_html}
</table>""", unsafe_allow_html=True)

# Growth of the actual portfolio value vs the same money in the S&P
beta = rep["benchmark"]["beta"]
_growth = rep["benchmark"]["growth"]
pg = pd.Series(_growth["portfolio"]["values"], index=pd.to_datetime(_growth["portfolio"]["dates"]))
sg = pd.Series(_growth["benchmark"]["values"], index=pd.to_datetime(_growth["benchmark"]["dates"]))
bfig = go.Figure()
bfig.add_trace(go.Scatter(x=pg.index, y=pg.values, mode="lines",
                          line=dict(color="#05d69e", width=2), name="Your portfolio"))
bfig.add_trace(go.Scatter(x=sg.index, y=sg.values, mode="lines",
                          line=dict(color="#6a849e", width=2, dash="dash"), name="S&P 500"))
bfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                   font=dict(color="#dde4f0", family="DM Sans"),
                   xaxis=dict(gridcolor="#1c2d44"),
                   yaxis=dict(title=f"Value of ${port_val:,.0f} invested", gridcolor="#1c2d44"),
                   height=340, margin=dict(l=20, r=20, t=10, b=20),
                   legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(bfig, width="stretch")

beat = "outperformed" if pg.iloc[-1] > sg.iloc[-1] else "trailed"
risk_word = "less" if beta < 1 else "more"
st.markdown(f"""
<div class='insight'>
    <div class='insight-icon'>🏛️</div>
    <div class='insight-text'>
        <strong>Beta {beta:.2f}:</strong> your portfolio moves {abs(beta):.2f}× the market —
        it carries <strong>{risk_word} market risk than the S&amp;P 500</strong>. Over this period
        ${port_val:,.0f} grew to <strong>${pg.iloc[-1]:,.0f}</strong> in your portfolio vs
        <strong>${sg.iloc[-1]:,.0f}</strong> in the index — you {beat} the market.
        The point isn't just return — it's whether you were <em>paid for the risk you took</em>,
        which is what the Sharpe comparison above shows.
    </div>
</div>""", unsafe_allow_html=True)

# ── RETURN DISTRIBUTION: ACTUAL vs GAUSSIAN ───────────────────
st.markdown("<div class='sec'>Return distribution — actual vs what Gaussian assumes</div>",
            unsafe_allow_html=True)
fig = go.Figure()
D = rep["distribution"]
edges = D["edges"]
fig.add_trace(go.Bar(x=[(a + b) / 2 * 100 for a, b in zip(edges, edges[1:])],
                     y=[d / 100 for d in D["density"]],
                     width=[(b - a) * 100 for a, b in zip(edges, edges[1:])],
                     marker_color="#4d9fff", opacity=0.6, name="Actual daily returns"))
x_r = np.linspace(edges[0] * 100, edges[-1] * 100, 400)
fig.add_trace(go.Scatter(x=x_r, y=norm.pdf(x_r, D["mean"] * 100, D["std"] * 100),
                         mode="lines", line=dict(color="#ffb347", width=2.5, dash="dash"),
                         name="Gaussian assumption"))
fig.add_vline(x=R["hist_var"] * 100, line_color="#ff3d5a", line_width=2,
              annotation_text=f"Historical VaR {R['hist_var']*100:.2f}%", annotation_font_color="#ff3d5a")
fig.add_vline(x=R["gaussian_var"] * 100, line_color="#4d9fff", line_width=2, line_dash="dash",
              annotation_text=f"Gaussian VaR {R['gaussian_var']*100:.2f}%", annotation_font_color="#4d9fff",
              annotation_position="bottom right")
fig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                  font=dict(color="#dde4f0", family="DM Sans"),
                  xaxis=dict(title="Daily portfolio return (%)", gridcolor="#1c2d44"),
                  yaxis=dict(title="Density", gridcolor="#1c2d44"),
                  height=400, margin=dict(l=20, r=20, t=10, b=20),
                  legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(fig, width="stretch")
st.markdown("""
<div class='insight'>
    <div class='insight-icon'>📖</div>
    <div class='insight-text'>
        <strong>How to read this:</strong> blue bars are real returns; the orange dashed line is
        the bell curve a standard model assumes. The actual returns reach further into the left
        tail (extreme losses) than Gaussian predicts — that fat tail is exactly the risk a
        normal-distribution model underestimates, and why the red Historical VaR line sits
        further left than the blue Gaussian one.
    </div>
</div>""", unsafe_allow_html=True)

# ── DRAWDOWN ──────────────────────────────────────────────────
st.markdown("<div class='sec'>Drawdown — how far underwater the portfolio went</div>",
            unsafe_allow_html=True)
ddfig = go.Figure()
dd = rep["drawdown"]
ddfig.add_trace(go.Scatter(x=pd.to_datetime(dd["dates"]), y=np.array(dd["values"]) * 100,
                           fill="tozeroy", mode="lines",
                           line=dict(color="#ff3d5a", width=1.2),
                           fillcolor="rgba(255,61,90,0.15)", name="Drawdown"))
ddfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                    font=dict(color="#dde4f0", family="DM Sans"),
                    xaxis=dict(gridcolor="#1c2d44"),
                    yaxis=dict(title="Drawdown (%)", gridcolor="#1c2d44"),
                    height=320, margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
st.plotly_chart(ddfig, width="stretch")

# ── STRESS TESTING: HISTORICAL CRISES ─────────────────────────
st.markdown("<div class='sec'>Stress test — how this exact portfolio would have survived past crises</div>",
            unsafe_allow_html=True)
st.caption("Buy-and-hold your current holdings & weights through real historical crash windows. This is what risk managers actually do — past distributions break, so you pressure-test against the real thing.")


CRISIS_PALETTE = ["#ffb347", "#ff3d5a", "#4d9fff", "#05d69e", "#c792ea", "#7fdbca"]
_crisis_color = {c["label"]: CRISIS_PALETTE[i % len(CRISIS_PALETTE)]
                 for i, c in enumerate(rep["stress"])}
results = [c for c in rep["stress"] if c["status"] == "ok"]
_missed = {c["label"]: c["missing"] for c in rep["stress"] if c["status"] == "not_covered"}
_sparse = [c["label"] for c in rep["stress"] if c["status"] == "too_few_days"]

if results:
    cols = st.columns(len(results))
    for col, c in zip(cols, results):
        with col:
            tcol = "#05d69e" if c["total"] >= 0 else "#ff3d5a"
            col.markdown(f"""<div class='big-stat'>
                <div class='big-stat-label'>{c["label"]}</div>
                <div class='big-stat-num' style='color:{tcol};'>{c["total"]*100:+.1f}%</div>
                <div class='big-stat-sub'>${abs(c["change_dollars"]):,.0f} {"gain" if c["total"]>=0 else "loss"} on ${port_val:,.0f}</div>
                <div class='big-stat-sub' style='margin-top:8px;color:#ff8a5a;'>worst day {c["worst_day"]*100:.1f}% · max drawdown {c["max_dd"]*100:.1f}%</div>
                <div class='big-stat-sub' style='margin-top:6px;font-style:italic;'>{c["context"]}</div>
                </div>""", unsafe_allow_html=True)

    # Rebased value paths through each crisis
    sfig = go.Figure()
    for c in results:
        path = np.array(c["path"]["values"]) * 100
        sfig.add_trace(go.Scatter(
            x=list(range(len(path))), y=path,
            mode="lines", name=c["label"], line=dict(color=_crisis_color[c["label"]], width=2)))
    sfig.add_hline(y=0, line_color="#5a7088", line_width=1)
    sfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                       font=dict(color="#dde4f0", family="DM Sans"),
                       xaxis=dict(title="Trading days into the crisis", gridcolor="#1c2d44"),
                       yaxis=dict(title="Portfolio return (%)", gridcolor="#1c2d44"),
                       height=360, margin=dict(l=20, r=20, t=10, b=20),
                       legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
    st.plotly_chart(sfig, width="stretch")

    worst_crisis = min(results, key=lambda c: c["total"])
    st.markdown(f"""
    <div class='insight danger'>
        <div class='insight-icon'>🔥</div>
        <div class='insight-text'>
            <strong>Your worst case was {worst_crisis["label"]}:</strong> this portfolio would have lost
            <strong>{abs(worst_crisis["total"])*100:.1f}%</strong> (${abs(worst_crisis["change_dollars"]):,.0f} on ${port_val:,.0f}),
            with a worst single day of {worst_crisis["worst_day"]*100:.1f}%. Daily VaR tells you about an
            ordinary bad day — a stress test tells you about the days that actually end careers.
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.info("None of the crisis windows is covered by every holding's price history.")

if _missed:
    st.caption("Not shown, because a holding was not yet trading: " + "; ".join(
        f"{label} ({', '.join(tickers)})" for label, tickers in _missed.items()))
if _sparse:
    st.caption(f"Not shown, because the data has fewer than {stress.MIN_CRISIS_DAYS} clean "
               "trading days in the window: " + "; ".join(_sparse))

# ── CORRELATION MATRIX ────────────────────────────────────────
st.markdown("<div class='sec'>Correlation matrix — how your holdings move together</div>",
            unsafe_allow_html=True)
st.caption("Green = move together · Red = move opposite · high correlations mean less diversification benefit")
C = rep["correlation"]
corr = pd.DataFrame(C["matrix"], index=C["tickers"], columns=C["tickers"]).astype(float)
hm = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale=[[0, "#ff3d5a"], [0.5, "#0c1220"], [1, "#05d69e"]],
    zmid=0, zmin=-1, zmax=1,
    text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=11),
    colorbar=dict(title="Corr", tickfont=dict(color="#dde4f0")),
))
hm.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                 font=dict(color="#dde4f0", family="DM Sans"),
                 height=420, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(hm, width="stretch")

# ── MODEL VALIDATION: DID THE VAR ACTUALLY HOLD? ──────────────
st.markdown("<div class='sec'>Model validation — did the risk numbers actually hold up?</div>",
            unsafe_allow_html=True)
B = rep["backtest"]
st.caption(f"A VaR estimate is only worth anything if it's been backtested. We roll a "
           f"{B['window']}-day window across the analysis window, and each day ask: did the real loss breach "
           f"the VaR? A {int(conf*100)}% model should be breached about {int(round((1-conf)*100))}% "
           f"of the time — no more, and not in clusters. "
           f"Kupiec tests the rate; Christoffersen tests that breaches don't bunch up in crises.")

bt_rows = [{"method": m, **row} for m, row in B["methods"].items()]
bt_graded = all(r["passed"] is not None for r in bt_rows)

rows_html = ""
for row in bt_rows:
    if row["passed"] is None:
        verdict, vcol = "too few days", "#5a7088"
    else:
        verdict = "PASS" if row["passed"] else "FAIL"
        vcol = "#05d69e" if row["passed"] else "#ff3d5a"
    rows_html += (
        f"<tr>"
        f"<td style='padding:10px 14px;color:#8a9bb8;text-transform:capitalize;'>{row['method']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;color:#6a849e;'>{row['observations']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['breaches']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;color:#6a849e;'>{row['expected']}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['kupiec_p']:.3f}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;'>{row['christoffersen_p']:.3f}</td>"
        f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;font-weight:600;color:{vcol};'>{verdict}</td>"
        f"</tr>")
st.markdown(f"""
<table style='width:100%;border-collapse:collapse;background:var(--card);
              border:1px solid var(--border);border-radius:10px;overflow:hidden;'>
    <tr style='background:#0c1220;'>
        <th style='padding:10px 14px;text-align:left;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Method</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Days tested</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Breaches</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Expected</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Kupiec p</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Christoffersen p</th>
        <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Verdict</th>
    </tr>
    {rows_html}
</table>""", unsafe_allow_html=True)

# Breach-timeline chart for the historical method
bt_series = B["series"]
bt_dates = pd.to_datetime(bt_series["dates"])
bt_realized = np.array(bt_series["realized"]) * 100
breach_mask = np.array(bt_series["breach"], dtype=bool)
mvfig = go.Figure()
mvfig.add_trace(go.Scatter(x=bt_dates, y=bt_realized, mode="lines",
                           line=dict(color="#4d9fff", width=1), name="Daily return"))
mvfig.add_trace(go.Scatter(x=bt_dates, y=np.array(bt_series["var"]) * 100, mode="lines",
                           line=dict(color="#ffb347", width=1.5, dash="dash"), name="Historical VaR"))
mvfig.add_trace(go.Scatter(x=bt_dates[breach_mask], y=bt_realized[breach_mask],
                           mode="markers", marker=dict(color="#ff3d5a", size=6, symbol="x"),
                           name="Breach"))
mvfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                    font=dict(color="#dde4f0", family="DM Sans"),
                    xaxis=dict(gridcolor="#1c2d44"),
                    yaxis=dict(title="Daily return (%)", gridcolor="#1c2d44"),
                    height=360, margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(mvfig, width="stretch")

_hist_row = next(r for r in bt_rows if r["method"] == "historical")
_gauss_row = next(r for r in bt_rows if r["method"] == "gaussian")


def _mv_rate(row):
    return (f"gets the breach <em>rate</em> right (Kupiec p={row['kupiec_p']:.2f})"
            if row["kupiec_p"] > 0.05
            else f"breaks the expected rate (Kupiec p={row['kupiec_p']:.2f})")


def _mv_clust(row):
    return ("with breaches staying independent" if row["christoffersen_p"] > 0.05
            else f"but breaches <strong>cluster in crises</strong> "
                 f"(Christoffersen p={row['christoffersen_p']:.3f})")


_both_cluster = (_hist_row["christoffersen_p"] <= 0.05
                 and _gauss_row["christoffersen_p"] <= 0.05)
_closing = (
    "Both models get the frequency about right yet fail the independence test — real losses "
    "bunch together in crises (COVID 2020, the 2022 bear), the exact stretches a static VaR "
    "never sees coming. That gap is the honest limit of any single-number risk measure, and "
    "naming it is the whole job."
    if _both_cluster else
    "A model earns trust only by passing both — the right rate <em>and</em> independent breaches. "
    "That is the test a risk desk runs before it believes any VaR at all.")
if bt_graded:
    st.markdown(f"""
<div class='insight warn'>
    <div class='insight-icon'>🧪</div>
    <div class='insight-text'>
        <strong>The honest scoreboard.</strong> Over the {_hist_row['observations']:,} tested days, historical VaR was breached
        <strong>{_hist_row['breaches']}</strong> times vs <strong>{_hist_row['expected']}</strong> expected,
        and Gaussian <strong>{_gauss_row['breaches']}</strong> times. Historical {_mv_rate(_hist_row)}
        {_mv_clust(_hist_row)}; Gaussian {_mv_rate(_gauss_row)} {_mv_clust(_gauss_row)}.
        Kupiec asks whether the breach <em>rate</em> matches the confidence level; Christoffersen asks
        whether breaches arrive <em>independently</em> or bunch together. {_closing}
    </div>
</div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
<div class='insight warn'>
    <div class='insight-icon'>🧪</div>
    <div class='insight-text'>
        <strong>Too few days to judge.</strong> The first {B['window']} days of this window are used to
        estimate VaR, which leaves only {_hist_row['observations']} days to test it against; at least
        {B['min_observations']} are needed before a pass or fail means anything. Historical VaR was breached
        <strong>{_hist_row['breaches']}</strong> times vs <strong>{_hist_row['expected']}</strong> expected,
        and Gaussian <strong>{_gauss_row['breaches']}</strong> times — shown for reference, not as a verdict.
    </div>
</div>""", unsafe_allow_html=True)

# ── RISK CONTRIBUTION ─────────────────────────────────────────
st.markdown("<div class='sec'>Risk contribution — who really drives your risk, not just your money</div>",
            unsafe_allow_html=True)
st.caption("A holding can be a small slice of your capital but a big slice of your risk — or the "
           "reverse. This splits total portfolio risk into how much each holding actually contributes.")

RC = rep["risk_contribution"]
risk_pct = np.array([r["risk_share"] for r in RC["rows"]])

rcfig = go.Figure()
rcfig.add_trace(go.Bar(x=selected, y=weights * 100, name="Capital %", marker_color="#4d9fff"))
rcfig.add_trace(go.Bar(x=selected, y=risk_pct * 100, name="Risk contribution %", marker_color="#ff3d5a"))
rcfig.update_layout(barmode="group", plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                    font=dict(color="#dde4f0", family="DM Sans"),
                    xaxis=dict(gridcolor="#1c2d44"),
                    yaxis=dict(title="% of portfolio", gridcolor="#1c2d44"),
                    height=360, margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(rcfig, width="stretch")

_hot = selected.index(RC["biggest_driver"])
_hot_txt = f"{selected[_hot]} is {weights[_hot]*100:.0f}% of your money but {risk_pct[_hot]*100:.0f}% of your risk"
_div_txt = ""
if RC["diversifier"] is not None:
    _div = selected.index(RC["diversifier"])
    _div_txt = (f" Meanwhile <strong>{selected[_div]}</strong> pulls the other way — "
                f"{weights[_div]*100:.0f}% of capital but only {risk_pct[_div]*100:.0f}% of the risk, "
                f"so it's working as a diversifier that calms the whole portfolio.")
st.markdown(f"""
<div class='insight warn'>
    <div class='insight-icon'>🎯</div>
    <div class='insight-text'>
        <strong>Concentration hides here:</strong> <strong>{_hot_txt}</strong> — a red bar towering over
        the blue one means that holding drives more danger than its size suggests.{_div_txt}
        This is exactly why equal-dollar weighting is not equal-<em>risk</em> weighting — and it's the gap
        the optimizer below closes.
    </div>
</div>""", unsafe_allow_html=True)

# ── FACTOR EXPOSURE (Fama-French 3-factor) ────────────────────
st.markdown("<div class='sec'>Factor exposure — what bets is this portfolio really making?</div>",
            unsafe_allow_html=True)
st.caption("Every portfolio is a bundle of a few underlying bets. The Fama-French model splits "
           "returns into three: the market, company size (small vs large), and value vs growth. "
           "Regressing this portfolio on them shows the tilts you actually hold — and how much of "
           "your risk is plain market beta vs bets you chose.")
F = rep["factors"]
if F["data_through"]:
    st.caption(f"Factor data through {F['data_through']} — Ken French "
               f"publishes monthly, so it trails prices by about a month.")

if not F["available"]:
    st.info(F["reason"])
else:
    _LABEL = {"Mkt-RF": "Market", "SMB": "Size (small−large)", "HML": "Value (value−growth)"}
    _rows = ""
    for name, b in F["betas"].items():
        tilt = ("—" if abs(b) < 0.05 else
                ("tilts toward " + ("small-cap" if name == "SMB" and b > 0 else
                                    "large-cap" if name == "SMB" else
                                    "value" if name == "HML" and b > 0 else
                                    "growth" if name == "HML" else
                                    "more market risk" if b > 1 else "less market risk")))
        _rows += (f"<tr>"
                  f"<td style='padding:10px 14px;color:#8a9bb8;'>{_LABEL[name]}</td>"
                  f"<td style='padding:10px 14px;text-align:right;font-family:DM Mono;font-weight:600;'>{b:+.2f}</td>"
                  f"<td style='padding:10px 14px;color:#6a849e;'>{tilt}</td>"
                  f"</tr>")
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;background:var(--card);
                  border:1px solid var(--border);border-radius:10px;overflow:hidden;'>
        <tr style='background:#0c1220;'>
            <th style='padding:10px 14px;text-align:left;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Factor</th>
            <th style='padding:10px 14px;text-align:right;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#4d9fff;'>Beta</th>
            <th style='padding:10px 14px;text-align:left;font-family:DM Mono;font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:#5a7088;'>Reading</th>
        </tr>
        {_rows}
    </table>""", unsafe_allow_html=True)

    # Variance decomposition bar
    _names = [*F["betas"], "Idiosyncratic"]
    _vals = [F["variance_split"][n] * 100 for n in _names]
    _colors = ["#4d9fff", "#ffb347", "#05d69e", "#5a7088"]
    ffig = go.Figure(go.Bar(x=[_LABEL.get(n, n) for n in _names], y=_vals,
                            marker_color=_colors, text=[f"{v:.0f}%" for v in _vals],
                            textposition="outside"))
    ffig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                       font=dict(color="#dde4f0", family="DM Sans"),
                       xaxis=dict(gridcolor="#1c2d44"),
                       yaxis=dict(title="% of portfolio variance", gridcolor="#1c2d44"),
                       height=340, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(ffig, width="stretch")

    _mkt_pct = F["variance_split"]["Mkt-RF"] * 100
    _smb, _hml = F["betas"]["SMB"], F["betas"]["HML"]
    _size_word = "small-cap" if _smb > 0.05 else "large-cap" if _smb < -0.05 else "size-neutral"
    _val_word = "value" if _hml > 0.05 else "growth" if _hml < -0.05 else "style-neutral"
    st.markdown(f"""
    <div class='insight'>
        <div class='insight-icon'>🧬</div>
        <div class='insight-text'>
            <strong>Your portfolio in one sentence:</strong> about <strong>{_mkt_pct:.0f}%</strong> of its
            risk is plain market beta (market β = {F['betas']['Mkt-RF']:.2f}), with a
            <strong>{_size_word}</strong> tilt and a <strong>{_val_word}</strong> lean. Annualized alpha —
            the return not explained by these three factors — is <strong>{F['alpha_annual']*100:+.1f}%</strong>,
            and the model explains <strong>{F['r2']*100:.0f}%</strong> of the day-to-day moves (R²).
            Alpha this small is the honest norm: most of what a diversified portfolio does is factor exposure,
            not stock-picking magic. Factor betas here regress log excess returns on the simple Fama-French
            factors — a standard daily-frequency approximation.
        </div>
    </div>""", unsafe_allow_html=True)

# ── PORTFOLIO OPTIMIZER & DIVERSIFICATION ─────────────────────
st.markdown("<div class='sec'>Optimizer — where you are vs. where the math says you should be</div>",
            unsafe_allow_html=True)
st.caption("Markowitz mean-variance optimization on your holdings (long-only, fully invested). "
           "Educational only — historical returns are a noisy guide to the future, never a promise.")

O = rep["optimizer"]
opt_ok = O["available"]

if not opt_ok:
    st.info("Optimizer couldn't converge for this selection — try different holdings. ("
            + O["reason"] + ")")
else:
    cur, ms, mv = O["current"], O["max_sharpe"], O["min_variance"]

    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Your Portfolio</div>
            <div class='big-stat-num' style='color:#4d9fff;'>{cur["sharpe"]:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {cur["ret"]*100:.1f}% return · {cur["vol"]*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)
    with o2:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Best Risk-Adjusted (Max Sharpe)</div>
            <div class='big-stat-num' style='color:#05d69e;'>{ms["sharpe"]:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {ms["ret"]*100:.1f}% return · {ms["vol"]*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)
    with o3:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Lowest Risk (Min Variance)</div>
            <div class='big-stat-num' style='color:#ffb347;'>{mv["sharpe"]:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {mv["ret"]*100:.1f}% return · {mv["vol"]*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)

    # Efficient frontier
    fr_v = [p["vol"] * 100 for p in O["frontier"]]
    fr_r = [p["ret"] * 100 for p in O["frontier"]]

    effig = go.Figure()
    if fr_v:
        effig.add_trace(go.Scatter(x=fr_v, y=fr_r, mode="lines",
                                   line=dict(color="#4d9fff", width=2.5),
                                   name="Efficient frontier"))
    effig.add_trace(go.Scatter(x=[cur["vol"]*100], y=[cur["ret"]*100], mode="markers",
                               marker=dict(color="#4d9fff", size=14, symbol="circle",
                                           line=dict(color="#fff", width=1)),
                               name="Your portfolio"))
    effig.add_trace(go.Scatter(x=[ms["vol"]*100], y=[ms["ret"]*100], mode="markers",
                               marker=dict(color="#05d69e", size=18, symbol="star"),
                               name="Max Sharpe"))
    effig.add_trace(go.Scatter(x=[mv["vol"]*100], y=[mv["ret"]*100], mode="markers",
                               marker=dict(color="#ffb347", size=14, symbol="diamond"),
                               name="Min variance"))
    effig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                        font=dict(color="#dde4f0", family="DM Sans"),
                        xaxis=dict(title="Risk — annualized volatility (%)", gridcolor="#1c2d44"),
                        yaxis=dict(title="Expected return (%)", gridcolor="#1c2d44"),
                        height=400, margin=dict(l=20, r=20, t=10, b=20),
                        legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
    st.plotly_chart(effig, width="stretch")
    st.markdown("""
    <div class='insight'>
        <div class='insight-icon'>📖</div>
        <div class='insight-text'>
            <strong>How to read this:</strong> the blue curve is every "best possible" portfolio —
            the most return for each level of risk. If <strong>your dot sits below the curve</strong>,
            you're taking risk you aren't paid for. The green star is the best risk-adjusted mix;
            the orange diamond is the safest. Moving your dot up toward the curve is the entire game.
        </div>
    </div>""", unsafe_allow_html=True)

    # Suggested reallocation (toward Max Sharpe)
    st.markdown("<div class='sec'>Suggested reallocation — to reach the best risk-adjusted mix</div>",
                unsafe_allow_html=True)
    moves = O["reallocation"]
    realloc = pd.DataFrame({
        "Holding": [m["ticker"] for m in moves],
        "Type": [m["asset_class"] for m in moves],
        "Sector": [m["sector"] for m in moves],
        "Now": [f"${m['current']:,.0f}" for m in moves],
        "Suggested": [f"${m['suggested']:,.0f}" for m in moves],
        "Change": [f"{'+' if m['delta_dollars'] >= 0 else '−'}${abs(m['delta_dollars']):,.0f}"
                   for m in moves],
    })
    st.dataframe(realloc, hide_index=True, width="stretch")

# Diversification / concentration insight. The sector mix needs no optimizer, so it shows either way.
SM = O["sector_mix"]
if SM["sectors"]:
    top_sec_pct = SM["top_sector_pct"] * 100
    conc_txt = (f"you're <strong>{top_sec_pct:.0f}% concentrated in {html.escape(SM['top_sector'])}</strong> "
                f"and {SM['equity']*100:.0f}% in equities overall.")
else:
    conc_txt = f"you're {SM['equity']*100:.0f}% in equities overall." if SM["equity"] else ""
diversified_txt = (f"{SM['diversified']*100:.0f}% is in broad-market or international funds, "
                   "which spread across many sectors." if SM["diversified"] else "")
unplaced_txt = (f"{SM['unplaced']*100:.0f}% is in holdings whose sector isn't known here, so it is "
                "left out of that check." if SM["unplaced"] else "")
non_equity_parts = [f"{share*100:.0f}% is in {NON_EQUITY_LABEL.get(cls, cls.lower())}"
                    for cls, share in SM["non_equity"].items()]
non_equity_txt = (f"{' and '.join(non_equity_parts)}, which are not a stock sector."
                  if non_equity_parts else "")

# Biggest suggested moves, in plain language; there are none to suggest when the solve failed
opt_txt = ""
if opt_ok:
    add_txt = ", ".join(f"{t} ({html.escape(SECTOR[t])})" for t in O["adds"]) or "none"
    trim_txt = ", ".join(O["trims"]) or "none"
    opt_txt = (" To climb toward the best risk-adjusted mix, the optimizer would "
               f"<strong>add to {add_txt}</strong> and <strong>trim {trim_txt}</strong> — "
               f"lifting your Sharpe from <strong>{cur['sharpe']:.2f}</strong> to "
               f"<strong>{ms['sharpe']:.2f}</strong> "
               f"({'more return for the same risk' if ms['sharpe'] > cur['sharpe'] else 'already near optimal'}).")

conc = "danger" if SM["concentrated"] else "warn"
st.markdown(f"""
<div class='insight {conc}'>
    <div class='insight-icon'>{"⚠️" if SM["concentrated"] else "🧭"}</div>
    <div class='insight-text'>
        <strong>Diversification check:</strong> {conc_txt} {non_equity_txt} {diversified_txt} {unplaced_txt}{opt_txt}
        Spreading across sectors and adding bonds is what flattens the crash damage you saw in the stress test above.
    </div>
</div>""", unsafe_allow_html=True)

# ── MONTE CARLO SIMULATION ────────────────────────────────────
st.markdown("<div class='sec'>Monte Carlo — ten thousand possible futures for this portfolio</div>",
            unsafe_allow_html=True)
st.caption("Each future is built by resampling this portfolio's actual historical daily returns "
           "(bootstrap), so it keeps the real fat tails a bell-curve model smooths away. "
           "Educational projection, not a forecast.")

MC = rep["monte_carlo"]
HORIZON_DAYS = {h["label"]: h["days"] for h in MC["horizons"]}
mc_h = HORIZON_DAYS[st.select_slider("Time horizon", options=list(HORIZON_DAYS), value="1 year")]
H = next(h for h in MC["horizons"] if h["days"] == mc_h)
t5, t50 = H["p5"], H["p50"]
prob_loss, prob_up20 = H["prob_loss"], H["prob_up20"]

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Median Outcome</div>
        <div class='big-stat-num' style='color:#05d69e;'>${t50:,.0f}</div>
        <div class='big-stat-sub'>{(t50/port_val-1)*100:+.1f}% on ${port_val:,.0f}</div>
        </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Bad Case (5th %ile)</div>
        <div class='big-stat-num' style='color:#ff3d5a;'>${t5:,.0f}</div>
        <div class='big-stat-sub'>{(t5/port_val-1)*100:+.1f}% · 1-in-20 downside</div>
        </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Probability of Loss</div>
        <div class='big-stat-num' style='color:{"#ff3d5a" if prob_loss>=0.3 else "#ffb347"};'>{prob_loss*100:.0f}%</div>
        <div class='big-stat-sub'>chance of ending below ${port_val:,.0f}</div>
        </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Chance of +20%</div>
        <div class='big-stat-num' style='color:#05d69e;'>{prob_up20*100:.0f}%</div>
        <div class='big-stat-sub'>ending at ${port_val*1.2:,.0f} or more</div>
        </div>""", unsafe_allow_html=True)

# Fan chart — the cone of outcomes over time
k = MC["band_days"].index(mc_h)
_x = MC["band_days"][:k + 1]
bands = [MC["bands"][f"p{p}"][:k + 1] for p in (5, 25, 50, 75, 95)]
mcfig = go.Figure()
mcfig.add_trace(go.Scatter(x=_x + _x[::-1], y=bands[4] + bands[0][::-1],
                           fill="toself", fillcolor="rgba(77,159,255,0.10)",
                           line=dict(width=0), name="5–95% range", hoverinfo="skip"))
mcfig.add_trace(go.Scatter(x=_x + _x[::-1], y=bands[3] + bands[1][::-1],
                           fill="toself", fillcolor="rgba(77,159,255,0.22)",
                           line=dict(width=0), name="25–75% range", hoverinfo="skip"))
mcfig.add_trace(go.Scatter(x=_x, y=bands[2], mode="lines",
                           line=dict(color="#05d69e", width=2.5), name="Median path"))
mcfig.add_hline(y=port_val, line_color="#5a7088", line_dash="dot",
                annotation_text=f"Start ${port_val:,.0f}", annotation_font_color="#5a7088")
mcfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                    font=dict(color="#dde4f0", family="DM Sans"),
                    xaxis=dict(title="Trading days into the future", gridcolor="#1c2d44"),
                    yaxis=dict(title="Portfolio value ($)", gridcolor="#1c2d44"),
                    height=400, margin=dict(l=20, r=20, t=10, b=20),
                    legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
st.plotly_chart(mcfig, width="stretch")

st.markdown(f"""
<div class='insight'>
    <div class='insight-icon'>🔮</div>
    <div class='insight-text'>
        <strong>How to read this:</strong> the green line is the most likely path; the shaded
        bands are where the portfolio lands 50–90% of the time. Across {MC['n_sims']:,} simulated futures,
        the middle outcome is <strong>${t50:,.0f}</strong>, but 1 in 20 times, after {H['label']}, it falls to
        <strong>${t5:,.0f}</strong> or worse. The fan widening over time is the honest truth about
        investing — <strong>the further out you look, the less certain anything is.</strong>
    </div>
</div>""", unsafe_allow_html=True)

# ── HONEST FOOTER: LIMITATIONS ────────────────────────────────
st.markdown("<div class='sec'>What this model assumes — and where it can be wrong</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='insight warn'>
    <div class='insight-icon'>🔍</div>
    <div class='insight-text'>
        <strong>Know the limits — this is the front-office skill.</strong>
        (1) Historical VaR/CVaR assume the future resembles the past distribution — every model
        that failed in 2008 made that assumption. (2) Correlations here are an average over the
        analysis window; in a real crisis they spike toward 1 and diversification collapses exactly
        when you need it. (3) Prices are end-of-day closes, refreshed once each US trading day — not
        live intraday. The honest framing of this tool is "a way to see how downside risk is
        measured," not a guarantee of tomorrow's loss.
    </div>
</div>""", unsafe_allow_html=True)
