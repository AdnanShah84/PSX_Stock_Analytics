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
    get_bundled_tickers,
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
# Custom styling — Streamlit's defaults are intentionally plain; this layers a
# cleaner look on top (card-style KPIs, colored signal badges, tighter type)
# without touching any of the underlying logic above.
# --------------------------------------------------------------------------- #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

/* Header */
.psx-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 2px;
}
.psx-header .logo { font-size: 1.9rem; line-height: 1; }
.psx-header h1 {
    margin: 0;
    font-size: 1.85rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
}
.psx-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.4rem;
}
.psx-sidebar-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 2px;
}
.psx-sidebar-header .logo { font-size: 1.5rem; }
.psx-sidebar-header .name { font-size: 1.15rem; font-weight: 700; color: #0f172a; }

/* KPI card grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 0.9rem 0 1.3rem;
}
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
}
.kpi-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
    font-weight: 600;
    margin-bottom: 8px;
}
.kpi-value { font-size: 1.5rem; font-weight: 700; color: #0f172a; line-height: 1.15; }
.kpi-value.up { color: #16a34a; }
.kpi-value.down { color: #dc2626; }
.kpi-sub { font-size: 0.8rem; font-weight: 600; margin-top: 6px; }
.kpi-sub.up { color: #16a34a; }
.kpi-sub.down { color: #dc2626; }
.kpi-sub.neutral { color: #64748b; }

/* Signal badge */
.signal-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 3px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.15rem;
}
.signal-badge.buy  { background: #dcfce7; color: #15803d; }
.signal-badge.hold { background: #fef3c7; color: #b45309; }
.signal-badge.sell { background: #fee2e2; color: #b91c1c; }

/* Sidebar */
[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] .stCaption { color: #94a3b8; }

/* Section headers */
h2, h3 { color: #0f172a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Caching — avoid re-scraping PSX on every interaction. Data refreshes every
# 15 minutes automatically, or immediately via the manual "Refresh now" button.
# --------------------------------------------------------------------------- #

CACHE_TTL_SECONDS = 15 * 60


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_ticker_universe():
    """
    Build the selectable ticker list. Always includes every ticker that has a
    bundled CSV snapshot (guaranteed to work even if PSX is unreachable live),
    plus anything from a live KSE-100 fetch if that happens to succeed — so the
    dropdown never offers a stock that then fails when selected.
    """
    bundled = get_bundled_tickers()
    live = get_kse100_tickers(limit=None)
    combined = sorted(set(bundled) | set(live)) if live else bundled
    return combined if combined else FALLBACK_TICKERS


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

st.sidebar.markdown("""
<div class="psx-sidebar-header">
  <span class="logo">📈</span>
  <span class="name">PSX Analytics</span>
</div>
""", unsafe_allow_html=True)
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

st.markdown("""
<div class="psx-header">
  <span class="logo">🇵🇰</span>
  <h1>Pakistan Stock Exchange — Analytics &amp; Signal Dashboard</h1>
</div>
<div class="psx-subtitle">Technical indicators + machine learning signals for KSE-100 stocks</div>
""", unsafe_allow_html=True)

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

        decision_class = sig["decision"].lower()
        decision_icon = {"buy": "▲", "hold": "●", "sell": "▼"}[decision_class]
        pct_change = raw["Close"].pct_change().iloc[-1] * 100
        pct_class = "up" if pct_change >= 0 else "down"
        pct_arrow = "▲" if pct_change >= 0 else "▼"

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-label">Last close (PKR)</div>
            <div class="kpi-value">{raw['Close'].iloc[-1]:.2f}</div>
            <div class="kpi-sub {pct_class}">{pct_arrow} {pct_change:+.2f}%</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Signal</div>
            <div class="signal-badge {decision_class}">{decision_icon} {sig['decision']}</div>
            <div class="kpi-sub neutral">{sig['confidence']:.0%} confidence</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Trend</div>
            <div class="kpi-value" style="font-size:1.25rem">{sig['trend']}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">RSI (14)</div>
            <div class="kpi-value">{sig['rsi']:.1f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

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
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close",
                                  line=dict(color="#2563eb", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50",
                                  line=dict(color="#f59e0b", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200",
                                  line=dict(color="#94a3b8", width=1.3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                  line=dict(color="#7c3aed", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#dc2626", row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#16a34a", row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                  line=dict(color="#2563eb", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal",
                                  line=dict(color="#f59e0b", width=1.5)), row=3, col=1)
        fig.update_layout(height=700, showlegend=True, plot_bgcolor="white", paper_bgcolor="white",
                           font=dict(family="Inter, sans-serif", color="#334155"),
                           margin=dict(l=10, r=10, t=40, b=10))
        fig.update_xaxes(gridcolor="#f1f5f9")
        fig.update_yaxes(gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        # Backtest
        st.subheader("Backtest — Strategy vs Buy & Hold")
        bt = result["backtest_summary"]
        strat_class = "up" if bt["strategy_return_pct"] >= 0 else "down"
        bh_class = "up" if bt["buy_and_hold_return_pct"] >= 0 else "down"
        st.markdown(f"""
        <div class="kpi-grid" style="grid-template-columns: repeat(3, 1fr);">
          <div class="kpi-card">
            <div class="kpi-label">Strategy return</div>
            <div class="kpi-value {strat_class}">{bt['strategy_return_pct']:+.2f}%</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Buy &amp; hold return</div>
            <div class="kpi-value {bh_class}">{bt['buy_and_hold_return_pct']:+.2f}%</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Number of trades</div>
            <div class="kpi-value">{bt['num_trades']}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        eq = result["equity_df"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=eq.index, y=eq["equity"], name="Strategy",
                                   line=dict(color="#2563eb", width=2)))
        fig2.add_trace(go.Scatter(x=eq.index, y=eq["buy_and_hold"], name="Buy & Hold",
                                   line=dict(color="#94a3b8", width=2, dash="dot")))
        fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Portfolio Value (PKR)",
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter, sans-serif", color="#334155"),
                            margin=dict(l=10, r=10, t=20, b=10))
        fig2.update_xaxes(gridcolor="#f1f5f9")
        fig2.update_yaxes(gridcolor="#f1f5f9")
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
                color = {"BUY": "background-color: #dcfce7; color: #15803d;",
                         "SELL": "background-color: #fee2e2; color: #b91c1c;",
                         "HOLD": "background-color: #fef3c7; color: #b45309;"}.get(row["Decision"], "")
                return [color] * len(row)
            st.dataframe(scan_df.style.apply(highlight, axis=1), use_container_width=True)

with tab3:
    st.markdown("""
### About this project
This dashboard combines **technical analysis** (SMA, RSI, MACD) with a **Random Forest**
machine learning model trained on historical PSX price action to generate Buy / Hold / Sell
signals for KSE-100 stocks, backed by a backtest against a buy-and-hold benchmark.

**Data source:** Official PSX Data Portal (`dps.psx.com.pk`) end-of-day historical data,
via the actively-maintained `psxdata` package.

**Important limitations:**
- This is **end-of-day** data, not a licensed real-time/intraday feed (PSX's live feed is a
  paid commercial product).
- The ML model predicts *direction*, not magnitude, and does not account for transaction
  costs, slippage, or position sizing.
- This is an **educational decision-support tool**, not financial advice. Always apply your
  own judgement before acting on real capital.

**Tech stack:** Python · scikit-learn · Streamlit · Plotly · psxdata
""")
