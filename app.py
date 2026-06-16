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
import yfinance as yf

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

# ── UNIVERSE (US large-caps, priced in USD) ───────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN",
           "JPM", "GS", "BAC", "MS", "XOM", "CVX", "COP",
           "JNJ", "PFE", "UNH", "ABBV", "TSLA", "WMT", "BA"]

TRADING_DAYS = 252
SNAPSHOT = os.path.join(os.path.dirname(__file__), "prices.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def load():
    """Return (prices, log_returns, source_label).

    Tries live Yahoo Finance first. Any ticker that fails to download is
    dropped rather than crashing the app. If the live pull is unusable,
    falls back to a frozen snapshot (prices.csv) committed alongside the app,
    so the dashboard never breaks in front of an audience.
    """
    prices, source = None, None

    # 1) Live pull
    try:
        raw = yf.download(TICKERS, start="2018-01-01", auto_adjust=True,
                          progress=False)["Close"]
        good = [t for t in TICKERS if t in raw.columns and raw[t].notna().sum() > 500]
        if len(good) >= 2:
            prices = raw[good].ffill().dropna()
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

AVAILABLE = list(prices.columns)
DEFAULTS = [t for t in ["AAPL", "MSFT", "JPM", "NVDA", "XOM"] if t in AVAILABLE][:5] \
    or AVAILABLE[:5]

# ── SIDEBAR: build the portfolio ──────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:DM Mono;font-size:9px;letter-spacing:0.2em;
                color:#4d9fff;text-transform:uppercase;margin-bottom:4px;'>Portfolio Risk Dashboard</div>
    <div style='font-family:Syne;font-size:22px;font-weight:800;margin-bottom:20px;'>Build Your Portfolio</div>
    """, unsafe_allow_html=True)

    selected = st.multiselect(
        "Stocks (pick 2–12)", AVAILABLE, default=DEFAULTS,
        help="Choose the holdings in your portfolio.",
    )

    port_val = st.number_input(
        "Portfolio value ($)", min_value=1_000, max_value=1_000_000_000,
        value=100_000, step=10_000, format="%d",
    )

    conf = st.select_slider(
        "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
        format_func=lambda x: f"{int(x*100)}%",
    )

    st.markdown("---")
    st.caption("Set weights (normalized to 100%). Leave as-is for equal weight.")

if len(selected) < 2:
    st.warning("Pick at least 2 stocks in the sidebar to build a portfolio.")
    st.stop()

# ── WEIGHTS (custom, normalized) ──────────────────────────────
with st.sidebar:
    default_w = round(100.0 / len(selected), 2)
    w_df = pd.DataFrame({"Stock": selected, "Weight %": [default_w] * len(selected)})
    edited = st.data_editor(
        w_df, hide_index=True, use_container_width=True,
        disabled=["Stock"],
        column_config={"Weight %": st.column_config.NumberColumn(
            min_value=0.0, max_value=100.0, step=1.0, format="%.1f")},
        key="weights",
    )

raw_w = edited["Weight %"].to_numpy(dtype=float)
if not np.isfinite(raw_w).any() or raw_w.sum() <= 0:
    raw_w = np.ones(len(selected))
weights = raw_w / raw_w.sum()

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
st.plotly_chart(fig, use_container_width=True)
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
st.plotly_chart(ddfig, use_container_width=True)

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
st.plotly_chart(hm, use_container_width=True)

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
