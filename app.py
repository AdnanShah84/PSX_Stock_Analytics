"""
PSX Stock Analytics Dashboard — Streamlit app
Deploy for free on Streamlit Community Cloud (streamlit.io/cloud).

Reuses psx_utils.py (same logic as the companion Colab notebook) so the analysis
shown here is consistent with what's documented/demonstrated in the notebook.
"""

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from psx_utils import (
    FALLBACK_TICKERS,
    get_kse100_tickers,
    fetch_stock_data,
    get_stock_data,
    get_live_quote,
    add_technical_indicators,
    build_features_and_labels,
    train_model,
    generate_signal,
    backtest_signals,
)
import psx_utils as _psx_utils_module

st.set_page_config(page_title="PSX Stock Analytics", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
# Caching — avoid re-scraping PSX on every interaction. Data refreshes every
# 15 minutes automatically, or immediately via the manual "Refresh now" button.
# --------------------------------------------------------------------------- #

CACHE_TTL_SECONDS = 15 * 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_ticker_universe():
    tickers = get_kse100_tickers(limit=20)
    return tickers if tickers else FALLBACK_TICKERS


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_and_analyze(ticker: str, years_back: int):
    raw, source, as_of = get_stock_data(ticker, years_back=years_back)
    if raw.empty:
        return {"error": _psx_utils_module.LAST_FETCH_ERROR or "Unknown fetch failure. "
                          "No bundled data/ snapshot found either."}
    if len(raw) < 210:
        return {"error": f"Only {len(raw)} rows available for {ticker} — need at least 210 "
                          f"(~200 trading days) to compute SMA200. Try a longer history window."}
    df = add_technical_indicators(raw)
    X, y, feat_df = build_features_and_labels(df)
    if len(X) < 100:
        return {"error": f"Only {len(X)} usable feature rows after indicator calculation "
                          f"for {ticker} — not enough for a reliable train/test split."}
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    test_df = feat_df.loc[X_test.index]
    test_accuracy = model.score(X_test, y_test)
    signal = generate_signal(model, feat_df.iloc[-1])
    equity_df, trades_df, backtest_summary = backtest_signals(model, test_df)
    return {
        "raw": raw,
        "source": source,
        "as_of": as_of,
        "df": df,
        "feat_df": feat_df,
        "model": model,
        "test_accuracy": test_accuracy,
        "signal": signal,
        "equity_df": equity_df,
        "trades_df": trades_df,
        "backtest_summary": backtest_summary,
    }


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def scan_watchlist(tickers: list, years_back: int):
    rows = []
    for tkr in tickers:
        result = load_and_analyze(tkr, years_back)
        if result is None or "error" in result:
            continue
        sig = result["signal"]
        rows.append({
            "Ticker": tkr,
            "Close (PKR)": round(result["raw"]["Close"].iloc[-1], 2),
            "Decision": sig["decision"],
            "Confidence": sig["confidence"],
            "Trend": sig["trend"],
            "RSI": sig["rsi"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.title("📈 PSX Analytics")
st.sidebar.caption("Data: PSX Data Portal (EOD) · Not a licensed real-time feed")

TICKERS = get_ticker_universe()

ticker = st.sidebar.selectbox("Select stock", TICKERS, index=0)
years_back = st.sidebar.slider("History window (years)", 1, 5, 3)

if st.sidebar.button("🔄 Refresh data now"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disclaimer:** Educational decision-support tool. Signals are based on "
    "historical technical + ML patterns and are **not financial advice**. "
    "Markets carry risk — verify independently before acting on real capital."
)
st.sidebar.caption(f"Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"(cache refreshes every {CACHE_TTL_SECONDS // 60} min)")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

st.title("Pakistan Stock Exchange — Analytics & Signal Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Stock Detail", "🧭 Watchlist Scan", "ℹ️ About"])

with tab1:
    with st.spinner(f"Fetching {ticker} from PSX and running model..."):
        result = load_and_analyze(ticker, years_back)

    if result is None or "error" in result:
        error_msg = result.get("error") if result else "Unknown error."
        st.error(f"**Data fetch failed for {ticker}:**\n\n{error_msg}")
        st.caption("This is the actual reason from the server — no need to check logs separately.")
    else:
        sig = result["signal"]
        raw = result["raw"]
        df = result["df"]

        if result.get("source") == "bundled":
            st.info(f"⚠️ Live PSX fetch unavailable from this server — showing a snapshot "
                     f"as of **{result['as_of']}** (bundled with the app). "
                     f"Click 'Refresh data now' to retry live fetch.")

        decision_color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[sig["decision"]]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close (PKR)", f"{raw['Close'].iloc[-1]:.2f}",
                   f"{raw['Close'].pct_change().iloc[-1]*100:+.2f}%")
        c2.metric("Signal", f"{decision_color} {sig['decision']}", f"{sig['confidence']:.0%} confidence")
        c3.metric("Trend", sig["trend"])
        c4.metric("RSI (14)", f"{sig['rsi']:.1f}")

        with st.expander("Why this signal? (reasoning)"):
            for r in sig["reasons"]:
                st.write("•", r)
            st.caption(f"Model test-set accuracy on held-out data: {result['test_accuracy']:.1%}")

        with st.expander("Live screener snapshot (refreshes ~every 15 min)"):
            live = get_live_quote(ticker)
            if live:
                st.json(live)
            else:
                st.caption("Live snapshot unavailable right now — showing EOD data only.")

        # Price + indicators chart
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                             subplot_titles=("Price + Moving Averages", "RSI (14)", "MACD"))
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"), row=3, col=1)
        fig.update_layout(height=700, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # Backtest
        st.subheader("Backtest — Strategy vs Buy & Hold")
        bt = result["backtest_summary"]
        b1, b2, b3 = st.columns(3)
        b1.metric("Strategy return", f"{bt['strategy_return_pct']:+.2f}%")
        b2.metric("Buy & Hold return", f"{bt['buy_and_hold_return_pct']:+.2f}%")
        b3.metric("Number of trades", bt["num_trades"])

        eq = result["equity_df"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="Strategy"))
        fig2.add_trace(go.Scatter(x=eq.index, y=eq["buy_and_hold"], name="Buy & Hold"))
        fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Portfolio Value (PKR)")
        st.plotly_chart(fig2, use_container_width=True)

        if not result["trades_df"].empty:
            with st.expander("Trade log (backtest period)"):
                st.dataframe(result["trades_df"], use_container_width=True)

with tab2:
    st.subheader("Watchlist signal scan")
    selected = st.multiselect("Tickers to scan", TICKERS, default=TICKERS[:10])
    if st.button("Run scan"):
        with st.spinner("Scanning tickers — this may take a minute..."):
            scan_df = scan_watchlist(selected, years_back)
        if scan_df.empty:
            st.warning("No results — try different tickers.")
        else:
            def highlight(row):
                color = {"BUY": "background-color: #d4f7d4",
                         "SELL": "background-color: #f7d4d4",
                         "HOLD": "background-color: #f7f3d4"}.get(row["Decision"], "")
                return [color] * len(row)
            st.dataframe(scan_df.style.apply(highlight, axis=1), use_container_width=True)

with tab3:
    st.markdown("""
### About this project
This dashboard combines **technical analysis** (SMA, RSI, MACD) with a **Random Forest**
machine learning model trained on historical PSX price action to generate Buy / Hold / Sell
signals for KSE-100 stocks, backed by a backtest against a buy-and-hold benchmark.

**Data source:** Official PSX Data Portal (`dps.psx.com.pk`) end-of-day historical data,
via the open-source `psx-data-reader` package.

**Important limitations:**
- This is **end-of-day** data, not a licensed real-time/intraday feed (PSX's live feed is a
  paid commercial product).
- The ML model predicts *direction*, not magnitude, and does not account for transaction
  costs, slippage, or position sizing.
- This is an **educational decision-support tool**, not financial advice. Always apply your
  own judgement before acting on real capital.

**Tech stack:** Python · scikit-learn · Streamlit · Plotly · psx-data-reader
""")
