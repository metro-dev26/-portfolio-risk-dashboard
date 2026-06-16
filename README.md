# Portfolio Risk Dashboard

A focused tool for measuring portfolio downside risk the way professional investors do —
built to understand, and show, how risk is actually measured.

**Live demo:** _(add your Streamlit Cloud URL here once deployed)_

## What it does

You build a portfolio from US large-caps, set weights, a dollar value, and a confidence level.
Every figure is computed **live from real Yahoo Finance data** — nothing is hardcoded:

- **Historical VaR & CVaR** — the honest, non-parametric loss numbers (CVaR is the headline:
  it answers "when it goes bad, how bad?").
- **Gaussian VaR & the Hidden Risk Gap** — how much risk a standard bell-curve model quietly
  misses, because real returns have fat tails.
- **Annualized volatility, Sharpe ratio, maximum drawdown** — the standard portfolio profile.
- **Return distribution** chart — actual returns vs the Gaussian assumption, side by side.
- **Drawdown** chart and a **correlation heatmap** of the holdings.
- An explicit **limitations** section — knowing where a model breaks is the point.

## Why it's built this way

Standard risk models assume returns follow a normal distribution. They don't — crashes are
bigger and more frequent than a bell curve predicts. This dashboard leads with the historical
(real-data) numbers and shows the gap against the Gaussian model, so the risk that standard
tools underestimate is visible.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech

Streamlit · NumPy · pandas · SciPy · Plotly · yfinance. Data: Yahoo Finance, end-of-day,
cached and refreshed hourly.
