# API Showcase Section — Design Doc

**Date:** 2026-08-23
**Status:** Design settled (agreed 2026-08-21), grounded against live code 2026-08-23
**Scope:** Purely additive UI section in `app.py`. No engine changes, no new deps, no changes to any existing section.

## Why

The Phase-3 REST API (`https://marketplug-risk-api.onrender.com`) is deployed and live, but nobody visiting the Streamlit app knows it exists. This adds a third view that explains the API is a **sibling door for other programs** — not a webpage — and links to the live Swagger docs.

Explicitly rejected last session (do not revisit):
- **Routing Streamlit's own compute through the API** — the app already calls `risk_engine` in-process; hitting its own deployed API adds cold-start lag + network failure for zero gain.
- **A live in-app "playground"** — either hangs on Render cold-start, or is theater faking a local call as an "API response."

## Placement — mirror the Beginner's Guide exactly

Current toggle (`app.py:104`):
```python
mode = st.radio("view", ["📊 Dashboard", "📖 Beginner's Guide"],
                horizontal=True, label_visibility="collapsed")
```

**Change 1 —** add a third option:
```python
mode = st.radio("view", ["📊 Dashboard", "📖 Beginner's Guide", "🔌 API"],
                horizontal=True, label_visibility="collapsed")
```

**Change 2 —** add an `elif` block immediately after the existing Beginner's Guide block ends (`st.stop()` on `app.py:189`), before the `# ── UNIVERSE` comment on `app.py:191`:
```python
elif mode == "🔌 API":
    ...  # ~40 self-contained lines
    st.stop()
```

Same shape as the guide block: render markdown → `st.stop()`. Pure additive. Zero touch to existing sections.

## Reuse only classes that exist (verified in app.py CSS, lines 62–99)

`.hero` `.hero-eyebrow` `.hero-title` `.hero-desc` · `.sec` · `.big-stat` `.big-stat-num` `.big-stat-label` `.big-stat-sub` · `.insight` `.insight-icon` `.insight-text` `.insight.warn`

No new CSS. No invented "card" class. Copyable request block = native `st.code(..., language="json")` (Streamlit renders a copy button for free). Docs link = native `st.link_button`. Both native to streamlit≥1.40 (already pinned) — no new deps.

## Content — 4 blocks

### 1. Hero
- eyebrow: `For developers · a public REST API`
- title: `This risk engine is also an API`
- desc: same math the dashboard runs, exposed as JSON for **other programs** — a data endpoint, not a webpage. One HTTP call returns risk metrics, factor exposures, and an optimized mix.

### 2. What it does — endpoint list (reuse `.insight` blocks, one per endpoint)
- `POST /analyze` — send holdings in dollars, get back VaR/CVaR, Sharpe, max drawdown, factor betas, and optimizer weights.
- `POST /portfolios` · `GET /portfolios` · `GET /portfolios/{id}` — save and retrieve named portfolios.
- `GET /health` — liveness check.

Then one copyable example request via `st.code(language="json")`:
```json
{"holdings": {"AAPL": 10000, "MSFT": 8000, "TLT": 5000}}
```

### 3. Worked example — REAL numbers, labeled a sample
Uses the actual response captured live 2026-08-21 from `POST /analyze` on `{AAPL 10000, MSFT 8000, TLT 5000}`:

| Field | Value |
|---|---|
| Historical VaR (1-day, 95%) | −2.12% |
| Sharpe | 0.78 |
| Max drawdown | −31.5% |
| Market beta (factor) | 0.88 |
| R² (factor fit) | 0.76 |
| Optimizer → max-Sharpe mix | AAPL 71% / MSFT 29% / TLT ~0%, Sharpe 0.80 |

Rendered as `.big-stat` cards or a compact table. **Hardcoded — NOT a live fetch** → no cold-start, no lie.

**Honesty guardrail (Tam's hard rule):** label it exactly `Sample response — computed 2026-08-21. Live values drift as the data snapshot updates.` This kills any implied guarantee that clicking "Try it live" reproduces these exact numbers. The example explains **what the optimizer computes and how to read it**, never a returns promise.

> Build-time decision: re-pull fresh numbers from the live API at build so the "sample" is current, then date-stamp with today's date. If the API is cold/slow at build time, fall back to the 8/21 numbers above — either way it's date-labeled and honest.

### 4. Docs link + honest cold-start note
- `st.link_button("Try it live → API docs", "https://marketplug-risk-api.onrender.com/docs")`
- small note beneath: `Free-tier API — the first request may take ~40s to wake the server.` (turns cold-start from "broken" into "expected")
- close with an `.insight.warn` block: educational tool, not financial advice; every number is built from past data.

## Non-goals / guardrails
- No live fetch anywhere in this block. No `requests`/`urllib` call to the API.
- No engine imports change. No existing section touched.
- No returns promise — describe computation, not advice.
- Small enough to build inline/one-shot (not subagent-driven).

## Build steps (after this doc is reviewed)
1. writing-plans → short plan (this is ~40 lines, plan can be brief).
2. Build the `elif` block + radio edit.
3. Boot locally, visually confirm all three views render and Dashboard/Guide are unchanged.
4. Commit (no AI co-author trailer). Push → Streamlit Cloud auto-redeploys.
