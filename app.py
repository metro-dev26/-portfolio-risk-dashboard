"""
Portfolio Risk Dashboard
========================
A focused tool for measuring portfolio downside risk the way professionals do —
with real market data, historical (non-Gaussian) tail risk, and the honest gap
between what standard models assume and what the market actually does.

Every number on this page is computed live from real market data.
Nothing is hardcoded.
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Use the operating system's certificate store for SSL. Fixes local machines
# behind antivirus / corporate-network HTTPS inspection; harmless in the cloud.
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

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
</style>
""", unsafe_allow_html=True)

# ── PAGE TOGGLE: Dashboard vs Beginner's Guide ────────────────
mode = st.radio("view", ["📊 Dashboard", "📖 Beginner's Guide"],
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
        left panel, pick the stocks and bonds you hold (or want to test) and type how much money is in each.</div>
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

# ── UNIVERSE (US large-caps + bond ETFs, priced in USD) ───────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
           "JPM", "GS", "BAC", "MS", "XOM", "CVX", "COP",
           "JNJ", "PFE", "UNH", "ABBV", "TSLA", "WMT", "BA",
           "TLT", "IEF", "AGG", "LQD"]

# Sector + asset-class tags drive the diversification analysis.
SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "NVDA": "Technology", "META": "Technology",
    "AMZN": "Consumer", "TSLA": "Consumer", "WMT": "Consumer",
    "JPM": "Financials", "GS": "Financials", "BAC": "Financials", "MS": "Financials",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "BA": "Industrials",
    "TLT": "Govt Bonds", "IEF": "Govt Bonds", "AGG": "Aggregate Bonds", "LQD": "Corp Bonds",
}
BONDS = {"TLT", "IEF", "AGG", "LQD"}
ASSET_CLASS = {t: ("Bond" if t in BONDS else "Equity") for t in TICKERS}

TRADING_DAYS = 252
SNAPSHOT = os.path.join(os.path.dirname(__file__), "prices.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def load():
    """Return (prices, log_returns, source_label).

    Pulls live daily prices from Yahoo Finance through Python's own HTTP/SSL
    stack — this works behind antivirus / corporate networks that do HTTPS
    inspection and break curl-based clients. Any ticker that fails is dropped
    rather than crashing. If the live pull is unusable, falls back to a frozen
    snapshot (prices.csv) so the dashboard never breaks in front of an audience.
    """
    import urllib.request
    import json
    import datetime

    def fetch(sym):
        p1 = int(datetime.datetime(2018, 1, 1).timestamp())
        p2 = int(datetime.datetime.now().timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
               f"?period1={p1}&period2={p2}&interval=1d")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        d = json.loads(urllib.request.urlopen(req, timeout=15).read())
        r = d["chart"]["result"][0]
        idx = pd.to_datetime([datetime.date.fromtimestamp(t) for t in r["timestamp"]])
        cl = r["indicators"]["adjclose"][0]["adjclose"]
        return pd.Series(cl, index=idx, name=sym)

    prices, source = None, None

    # 1) Live pull (Python SSL — survives HTTPS-inspecting networks)
    try:
        series = {}
        for t in TICKERS + ["SPY"]:   # SPY = S&P 500 benchmark
            try:
                s = fetch(t)
                if s.notna().sum() > 500:
                    series[t] = s
            except Exception:
                pass
        if len(series) >= 2:
            prices = pd.concat(series.values(), axis=1).ffill().dropna()
            source = "Yahoo Finance · live"
    except Exception:
        prices = None

    # 2) Frozen snapshot fallback
    if prices is None or prices.shape[1] < 2:
        if os.path.exists(SNAPSHOT):
            prices = pd.read_csv(SNAPSHOT, index_col=0, parse_dates=True).ffill().dropna()
            source = f"frozen snapshot · {prices.index.max().date()}"
        else:
            return None, None, None

    lr = np.log(prices / prices.shift(1)).dropna()
    return prices, lr, source


with st.spinner("Loading market data..."):
    prices, lr, source = load()

if prices is None:
    st.error("Market data is temporarily unavailable (the data provider is rate-limiting "
             "requests). Please refresh in a minute.")
    st.stop()

AVAILABLE = [c for c in prices.columns if c != "SPY"]
DEFAULTS = [t for t in ["AAPL", "MSFT", "JPM", "XOM", "TLT"] if t in AVAILABLE][:5] \
    or AVAILABLE[:5]

# ── SIDEBAR: build the portfolio ──────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:DM Mono;font-size:9px;letter-spacing:0.2em;
                color:#4d9fff;text-transform:uppercase;margin-bottom:4px;'>Portfolio Risk Dashboard</div>
    <div style='font-family:Syne;font-size:22px;font-weight:800;margin-bottom:20px;'>Build Your Portfolio</div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Holdings — stocks & bonds (pick 2–12)", AVAILABLE, default=DEFAULTS,
        help="Add the stocks and bond ETFs you actually hold.",
    )

    conf = st.select_slider(
        "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    st.markdown("---")
    st.caption("Enter how much money you hold in each (USD). The total is your portfolio value.")

if len(selected) < 2:
    st.warning("Pick at least 2 holdings in the sidebar to build a portfolio.")
    st.stop()

# ── HOLDINGS (real dollar amounts) ────────────────────────────
with st.sidebar:
    default_amt = float(round(100_000 / len(selected)))
    amt_df = pd.DataFrame({
        "Holding": selected,
        "Type": [ASSET_CLASS[t] for t in selected],
        "Amount $": [default_amt] * len(selected),
    })
    edited = st.data_editor(
        amt_df, hide_index=True, width="stretch",
        disabled=["Holding", "Type"],
        column_config={"Amount $": st.column_config.NumberColumn(
            min_value=0.0, step=1000.0, format="$%d")},
        key="amounts",
    )

amounts = edited["Amount $"].to_numpy(dtype=float)
amounts = np.where(np.isfinite(amounts) & (amounts > 0), amounts, 0.0)
if amounts.sum() <= 0:
    amounts = np.ones(len(selected))
port_val = float(amounts.sum())
weights = amounts / amounts.sum()

# ── RISK ENGINE (all computed live) ───────────────────────────
ret_sel = lr[selected].dropna()
pr = (ret_sel * weights).sum(axis=1)          # weighted daily portfolio log-returns
mu, std = pr.mean(), pr.std()

# Historical (non-parametric) — the honest numbers
h_var = np.percentile(pr, (1 - conf) * 100)   # quantile of returns (negative)
h_cvar = pr[pr <= h_var].mean()               # expected shortfall beyond VaR

# Gaussian (what standard models assume)
g_var = mu + std * norm.ppf(1 - conf)
g_cvar = mu - std * norm.pdf(norm.ppf(1 - conf)) / (1 - conf)

gap = (h_var - g_var) * port_val              # dollars of risk Gaussian misses

# Supporting metrics
ann_vol = std * np.sqrt(TRADING_DAYS)
sharpe = (mu / std) * np.sqrt(TRADING_DAYS) if std > 0 else 0.0
wealth = np.exp(pr.cumsum())
drawdown = wealth / wealth.cummax() - 1.0
max_dd = drawdown.min()

# ── HERO ──────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
    <div class='hero-eyebrow'>{source} · {len(ret_sel):,} trading days · end-of-day, refreshed hourly</div>
    <div class='hero-title'>Portfolio Risk Dashboard</div>
    <div class='hero-desc'>
        How much can this portfolio lose on a bad day — and how much of that risk does a
        standard Gaussian model quietly miss? Every figure below is computed live from real
        market data for your {len(selected)} selected holdings.
    </div>
</div>""", unsafe_allow_html=True)

# ── HOW TO USE (always visible, even if sidebar is collapsed) ─
st.markdown("""
<div class='insight'>
    <div class='insight-icon'>👈</div>
    <div class='insight-text'>
        <strong>This dashboard is interactive — build your own portfolio.</strong>
        All controls are in the panel on the left: choose stocks, set their weights, the portfolio
        value, and the confidence level. Every number and chart below recomputes instantly.
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
        <div class='big-stat-num' style='color:#ff3d5a;'>${abs(h_cvar*port_val):,.0f}</div>
        <div class='big-stat-sub'>average loss on the worst {int((1-conf)*100)}% of days</div>
        </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Historical VaR</div>
        <div class='big-stat-num' style='color:#ff8a5a;'>${abs(h_var*port_val):,.0f}</div>
        <div class='big-stat-sub'>the loss threshold · {abs(h_var)*100:.2f}% of portfolio</div>
        </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Gaussian VaR</div>
        <div class='big-stat-num' style='color:#4d9fff;'>${abs(g_var*port_val):,.0f}</div>
        <div class='big-stat-sub'>what a standard model reports</div>
        </div>""", unsafe_allow_html=True)
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
        ("{int(conf*100)}% of days you won't lose more than ${abs(h_var*port_val):,.0f}"). CVaR answers the
        question that actually matters — <em>when it goes bad, how bad?</em> — at ${abs(h_cvar*port_val):,.0f}.
        The Gaussian model here {"understates" if gap < 0 else "is close to"} the real loss threshold by
        ${abs(gap):,.0f} a day, because it assumes returns follow a bell curve and the market doesn't.
    </div>
</div>""", unsafe_allow_html=True)

# ── SUPPORTING METRICS ────────────────────────────────────────
st.markdown("<div class='sec'>Portfolio profile (full history)</div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Annualized Volatility</div>
        <div class='big-stat-num' style='color:#4d9fff;'>{ann_vol*100:.1f}%</div>
        <div class='big-stat-sub'>standard deviation of returns</div>
        </div>""", unsafe_allow_html=True)
with s2:
    sh_color = "#05d69e" if sharpe >= 1 else "#ffb347" if sharpe >= 0 else "#ff3d5a"
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Sharpe Ratio</div>
        <div class='big-stat-num' style='color:{sh_color};'>{sharpe:.2f}</div>
        <div class='big-stat-sub'>return per unit of risk (rf = 0)</div>
        </div>""", unsafe_allow_html=True)
with s3:
    st.markdown(f"""<div class='big-stat'>
        <div class='big-stat-label'>Maximum Drawdown</div>
        <div class='big-stat-num' style='color:#ff3d5a;'>{max_dd*100:.1f}%</div>
        <div class='big-stat-sub'>worst peak-to-trough fall</div>
        </div>""", unsafe_allow_html=True)

# ── BENCHMARK vs S&P 500 ──────────────────────────────────────
if "SPY" in lr.columns:
    st.markdown("<div class='sec'>Benchmark — your portfolio vs. the S&P 500</div>",
                unsafe_allow_html=True)
    st.caption("Every number above is more meaningful next to the market. The S&P 500 (SPY) is the "
               "yardstick professionals measure against.")

    spy_r = lr["SPY"]
    _common = pr.index.intersection(spy_r.index)
    prc, spyc = pr.loc[_common], spy_r.loc[_common]

    b_mu, b_sd = spyc.mean(), spyc.std()
    b_ret, b_vol = b_mu * TRADING_DAYS, b_sd * np.sqrt(TRADING_DAYS)
    b_sharpe = (b_mu / b_sd) * np.sqrt(TRADING_DAYS) if b_sd > 0 else 0.0
    b_wealth = np.exp(spyc.cumsum())
    b_dd = (b_wealth / b_wealth.cummax() - 1.0).min()
    port_ret = mu * TRADING_DAYS
    beta = float(np.cov(prc, spyc)[0, 1] / np.var(spyc)) if np.var(spyc) > 0 else 0.0

    def _vs(p, b, higher_better=True, pct=True, dec=1):
        better = (p > b) if higher_better else (p < b)
        col = "#05d69e" if better else "#ff8a5a"
        fmt = (f"{p*100:.{dec}f}%" if pct else f"{p:.2f}")
        return col, fmt

    cmp_rows = [
        ("Annualized return", port_ret, b_ret, True, True, 1),
        ("Annualized volatility (risk)", ann_vol, b_vol, False, True, 1),
        ("Sharpe ratio", sharpe, b_sharpe, True, False, 2),
        ("Maximum drawdown", max_dd, b_dd, True, True, 1),  # higher (less negative) is better
    ]
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
    pg = port_val * np.exp(prc.cumsum())
    sg = port_val * np.exp(spyc.cumsum())
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
fig.add_trace(go.Histogram(x=pr * 100, nbinsx=80, marker_color="#4d9fff", opacity=0.6,
                           name="Actual daily returns", histnorm="probability density"))
x_r = np.linspace(pr.min() * 100, pr.max() * 100, 400)
fig.add_trace(go.Scatter(x=x_r, y=norm.pdf(x_r, mu * 100, std * 100),
                         mode="lines", line=dict(color="#ffb347", width=2.5, dash="dash"),
                         name="Gaussian assumption"))
fig.add_vline(x=h_var * 100, line_color="#ff3d5a", line_width=2,
              annotation_text=f"Historical VaR {h_var*100:.2f}%", annotation_font_color="#ff3d5a")
fig.add_vline(x=g_var * 100, line_color="#4d9fff", line_width=2, line_dash="dash",
              annotation_text=f"Gaussian VaR {g_var*100:.2f}%", annotation_font_color="#4d9fff",
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
ddfig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100,
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

# Real crisis windows present in the data (2018→). Each: label, start, end, one-line context.
CRISES = [
    ("2018 Q4 Selloff",  "2018-10-01", "2018-12-24", "Fed tightening + trade-war fears"),
    ("COVID-19 Crash",   "2020-02-19", "2020-03-23", "Fastest-ever 30%+ market drop"),
    ("2022 Bear Market", "2022-01-03", "2022-10-12", "Inflation shock + rate hikes"),
]


def stress(window_start, window_end):
    """Buy-and-hold the selected portfolio across a window. Returns (total_return,
    max_drawdown, worst_day, value_path) or None if the window isn't fully covered."""
    wp = prices.loc[window_start:window_end, selected].dropna()
    if len(wp) < 5:
        return None
    norm = wp / wp.iloc[0]
    path = (norm * weights).sum(axis=1)          # portfolio value, starts at 1.0
    total = path.iloc[-1] - 1.0
    dd = (path / path.cummax() - 1.0).min()
    worst = path.pct_change().min()
    return total, dd, worst, path


results = [(label, ctx, stress(s, e)) for label, s, e, ctx in CRISES]
results = [(label, ctx, r) for label, ctx, r in results if r is not None]

if results:
    cols = st.columns(len(results))
    for col, (label, ctx, (total, dd, worst, _)) in zip(cols, results):
        with col:
            tcol = "#05d69e" if total >= 0 else "#ff3d5a"
            col.markdown(f"""<div class='big-stat'>
                <div class='big-stat-label'>{label}</div>
                <div class='big-stat-num' style='color:{tcol};'>{total*100:+.1f}%</div>
                <div class='big-stat-sub'>${abs(total*port_val):,.0f} {"gain" if total>=0 else "loss"} on ${port_val:,.0f}</div>
                <div class='big-stat-sub' style='margin-top:8px;color:#ff8a5a;'>worst day {worst*100:.1f}% · max drawdown {dd*100:.1f}%</div>
                <div class='big-stat-sub' style='margin-top:6px;font-style:italic;'>{ctx}</div>
                </div>""", unsafe_allow_html=True)

    # Rebased value paths through each crisis
    sfig = go.Figure()
    palette = ["#ffb347", "#ff3d5a", "#4d9fff"]
    for (label, _, (_, _, _, path)), color in zip(results, palette):
        sfig.add_trace(go.Scatter(
            x=list(range(len(path))), y=(path.values - 1) * 100,
            mode="lines", name=label, line=dict(color=color, width=2)))
    sfig.add_hline(y=0, line_color="#5a7088", line_width=1)
    sfig.update_layout(plot_bgcolor="#0c1220", paper_bgcolor="#0c1220",
                       font=dict(color="#dde4f0", family="DM Sans"),
                       xaxis=dict(title="Trading days into the crisis", gridcolor="#1c2d44"),
                       yaxis=dict(title="Portfolio return (%)", gridcolor="#1c2d44"),
                       height=360, margin=dict(l=20, r=20, t=10, b=20),
                       legend=dict(bgcolor="#111d2e", bordercolor="#1c2d44"))
    st.plotly_chart(sfig, width="stretch")

    worst_crisis = min(results, key=lambda r: r[2][0])
    st.markdown(f"""
    <div class='insight danger'>
        <div class='insight-icon'>🔥</div>
        <div class='insight-text'>
            <strong>Your worst case was {worst_crisis[0]}:</strong> this portfolio would have lost
            <strong>{abs(worst_crisis[2][0])*100:.1f}%</strong> (${abs(worst_crisis[2][0]*port_val):,.0f} on ${port_val:,.0f}),
            with a worst single day of {worst_crisis[2][2]*100:.1f}%. Daily VaR tells you about an
            ordinary bad day — a stress test tells you about the days that actually end careers.
        </div>
    </div>""", unsafe_allow_html=True)
else:
    st.info("The selected data range doesn't fully cover the crisis windows — try the default tickers.")

# ── CORRELATION MATRIX ────────────────────────────────────────
st.markdown("<div class='sec'>Correlation matrix — how your holdings move together</div>",
            unsafe_allow_html=True)
st.caption("Green = move together · Red = move opposite · high correlations mean less diversification benefit")
corr = ret_sel.corr()
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

# ── RISK CONTRIBUTION ─────────────────────────────────────────
st.markdown("<div class='sec'>Risk contribution — who really drives your risk, not just your money</div>",
            unsafe_allow_html=True)
st.caption("A holding can be a small slice of your capital but a big slice of your risk — or the "
           "reverse. This splits total portfolio risk into how much each holding actually contributes.")

_cov_rc = (lr[selected].cov() * TRADING_DAYS).to_numpy()
_pv = float(np.sqrt(weights @ _cov_rc @ weights))
if _pv > 0:
    _mcr = (_cov_rc @ weights) / _pv      # marginal contribution to risk
    _ccr = weights * _mcr                  # component contribution (sums to portfolio vol)
    risk_pct = _ccr / _ccr.sum()
else:
    risk_pct = weights.copy()

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

_gap = risk_pct - weights
_hot = int(np.argmax(_gap))
_div = int(np.argmin(risk_pct))
_hot_txt = f"{selected[_hot]} is {weights[_hot]*100:.0f}% of your money but {risk_pct[_hot]*100:.0f}% of your risk"
_div_txt = ""
if risk_pct[_div] < weights[_div] - 0.02:
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

# ── PORTFOLIO OPTIMIZER & DIVERSIFICATION ─────────────────────
from scipy.optimize import minimize

st.markdown("<div class='sec'>Optimizer — where you are vs. where the math says you should be</div>",
            unsafe_allow_html=True)
st.caption("Markowitz mean-variance optimization on your holdings (long-only, fully invested). "
           "Educational only — historical returns are a noisy guide to the future, never a promise.")

mu_v = (lr[selected].mean() * TRADING_DAYS).to_numpy()
cov_m = (lr[selected].cov() * TRADING_DAYS).to_numpy()
nA = len(selected)


def _perf(w):
    r = float(w @ mu_v)
    v = float(np.sqrt(w @ cov_m @ w))
    return r, v, (r / v if v > 0 else 0.0)


_cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
_bnds = tuple((0.0, 1.0) for _ in range(nA))
_w0 = np.ones(nA) / nA

try:
    w_ms = minimize(lambda w: -_perf(w)[2], _w0, method="SLSQP",
                    bounds=_bnds, constraints=_cons).x
    w_mv = minimize(lambda w: float(w @ cov_m @ w), _w0, method="SLSQP",
                    bounds=_bnds, constraints=_cons).x
    w_ms = np.clip(w_ms, 0, None); w_ms = w_ms / w_ms.sum()
    w_mv = np.clip(w_mv, 0, None); w_mv = w_mv / w_mv.sum()
    opt_ok = True
except Exception:
    opt_ok = False

if not opt_ok:
    st.info("Optimizer couldn't converge for this selection — try different holdings.")
else:
    cur_r, cur_v, cur_s = _perf(weights)
    ms_r, ms_v, ms_s = _perf(w_ms)
    mv_r, mv_v, mv_s = _perf(w_mv)

    o1, o2, o3 = st.columns(3)
    with o1:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Your Portfolio</div>
            <div class='big-stat-num' style='color:#4d9fff;'>{cur_s:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {cur_r*100:.1f}% return · {cur_v*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)
    with o2:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Best Risk-Adjusted (Max Sharpe)</div>
            <div class='big-stat-num' style='color:#05d69e;'>{ms_s:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {ms_r*100:.1f}% return · {ms_v*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)
    with o3:
        st.markdown(f"""<div class='big-stat'>
            <div class='big-stat-label'>Lowest Risk (Min Variance)</div>
            <div class='big-stat-num' style='color:#ffb347;'>{mv_s:.2f}</div>
            <div class='big-stat-sub'>Sharpe · {mv_r*100:.1f}% return · {mv_v*100:.1f}% risk</div>
            </div>""", unsafe_allow_html=True)

    # Efficient frontier
    fr_v, fr_r = [], []
    for tr in np.linspace(mv_r, float(mu_v.max()), 40):
        c = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},
             {"type": "eq", "fun": lambda w, tr=tr: float(w @ mu_v) - tr})
        res = minimize(lambda w: float(w @ cov_m @ w), _w0, method="SLSQP",
                       bounds=_bnds, constraints=c)
        if res.success:
            fr_v.append(float(np.sqrt(res.fun)) * 100)
            fr_r.append(tr * 100)

    effig = go.Figure()
    if fr_v:
        effig.add_trace(go.Scatter(x=fr_v, y=fr_r, mode="lines",
                                   line=dict(color="#4d9fff", width=2.5),
                                   name="Efficient frontier"))
    effig.add_trace(go.Scatter(x=[cur_v*100], y=[cur_r*100], mode="markers",
                               marker=dict(color="#4d9fff", size=14, symbol="circle",
                                           line=dict(color="#fff", width=1)),
                               name="Your portfolio"))
    effig.add_trace(go.Scatter(x=[ms_v*100], y=[ms_r*100], mode="markers",
                               marker=dict(color="#05d69e", size=18, symbol="star"),
                               name="Max Sharpe"))
    effig.add_trace(go.Scatter(x=[mv_v*100], y=[mv_r*100], mode="markers",
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
    realloc = pd.DataFrame({
        "Holding": selected,
        "Type": [ASSET_CLASS[t] for t in selected],
        "Sector": [SECTOR[t] for t in selected],
        "Now": [f"${a:,.0f}" for a in (weights * port_val)],
        "Suggested": [f"${a:,.0f}" for a in (w_ms * port_val)],
        "Change": [f"{'+' if d >= 0 else '−'}${abs(d):,.0f}" for d in ((w_ms - weights) * port_val)],
    })
    st.dataframe(realloc, hide_index=True, width="stretch")

    # Diversification / concentration insight
    sec_w = {}
    eq_w = 0.0
    for i, t in enumerate(selected):
        sec_w[SECTOR[t]] = sec_w.get(SECTOR[t], 0.0) + weights[i]
        if ASSET_CLASS[t] == "Equity":
            eq_w += weights[i]
    top_sec = max(sec_w, key=sec_w.get)
    top_sec_pct = sec_w[top_sec] * 100

    # Biggest suggested moves, in plain language
    deltas = (w_ms - weights)
    add_idx = [i for i in np.argsort(deltas)[::-1] if deltas[i] > 0.01][:2]
    trim_idx = [i for i in np.argsort(deltas) if deltas[i] < -0.01][:2]
    add_txt = ", ".join(f"{selected[i]} ({SECTOR[selected[i]]})" for i in add_idx) or "none"
    trim_txt = ", ".join(selected[i] for i in trim_idx) or "none"

    conc = "danger" if top_sec_pct >= 50 else "warn"
    st.markdown(f"""
    <div class='insight {conc}'>
        <div class='insight-icon'>{"⚠️" if top_sec_pct >= 50 else "🧭"}</div>
        <div class='insight-text'>
            <strong>Diversification check:</strong> you're <strong>{top_sec_pct:.0f}% concentrated in {top_sec}</strong>
            and {eq_w*100:.0f}% in equities overall. To climb toward the best risk-adjusted mix, the
            optimizer would <strong>add to {add_txt}</strong> and <strong>trim {trim_txt}</strong> —
            lifting your Sharpe from <strong>{cur_s:.2f}</strong> to <strong>{ms_s:.2f}</strong>
            ({"more return for the same risk" if ms_s > cur_s else "already near optimal"}).
            Spreading across sectors and adding bonds is what flattens the crash damage you saw in the stress test above.
        </div>
    </div>""", unsafe_allow_html=True)

# ── MONTE CARLO SIMULATION ────────────────────────────────────
st.markdown("<div class='sec'>Monte Carlo — ten thousand possible futures for this portfolio</div>",
            unsafe_allow_html=True)
st.caption("Each future is built by resampling this portfolio's actual historical daily returns "
           "(bootstrap), so it keeps the real fat tails a bell-curve model smooths away. "
           "Educational projection, not a forecast.")

mc_h = st.select_slider(
    "Time horizon", options=[126, 252, 504, 756], value=252,
    format_func=lambda d: {126: "6 months", 252: "1 year",
                           504: "2 years", 756: "3 years"}[d])

_rng = np.random.default_rng(42)
N_SIMS = 10_000
hist_r = pr.to_numpy()
_draws = _rng.integers(0, len(hist_r), size=(N_SIMS, mc_h))
sim_paths = port_val * np.exp(np.cumsum(hist_r[_draws], axis=1))   # N x H
terminal = sim_paths[:, -1]

bands = np.percentile(sim_paths, [5, 25, 50, 75, 95], axis=0)
t5, t50, t95 = np.percentile(terminal, [5, 50, 95])
prob_loss = float((terminal < port_val).mean())
prob_up20 = float((terminal >= port_val * 1.2).mean())

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
_x = list(range(1, mc_h + 1))
mcfig = go.Figure()
mcfig.add_trace(go.Scatter(x=_x + _x[::-1], y=list(bands[4]) + list(bands[0][::-1]),
                           fill="toself", fillcolor="rgba(77,159,255,0.10)",
                           line=dict(width=0), name="5–95% range", hoverinfo="skip"))
mcfig.add_trace(go.Scatter(x=_x + _x[::-1], y=list(bands[3]) + list(bands[1][::-1]),
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
        bands are where the portfolio lands {{}} of the time. Across 10,000 simulated futures,
        the middle outcome is <strong>${t50:,.0f}</strong>, but 1 year in 20 it falls to
        <strong>${t5:,.0f}</strong> or worse. The fan widening over time is the honest truth about
        investing — <strong>the further out you look, the less certain anything is.</strong>
    </div>
</div>""".replace("{}", "50–90%"), unsafe_allow_html=True)

# ── HONEST FOOTER: LIMITATIONS ────────────────────────────────
st.markdown("<div class='sec'>What this model assumes — and where it can be wrong</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='insight warn'>
    <div class='insight-icon'>🔍</div>
    <div class='insight-text'>
        <strong>Know the limits — this is the front-office skill.</strong>
        (1) Historical VaR/CVaR assume the future resembles the past distribution — every model
        that failed in 2008 made that assumption. (2) Correlations here are a full-history average;
        in a real crisis they spike toward 1 and diversification collapses exactly when you need it.
        (3) Data is end-of-day, refreshed hourly — not live intraday. The honest framing of this tool
        is "a way to see how downside risk is measured," not a guarantee of tomorrow's loss.
    </div>
</div>""", unsafe_allow_html=True)
