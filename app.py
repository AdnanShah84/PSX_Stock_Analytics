"""
PSX Stock Analytics Dashboard — Streamlit app
Deploy for free on Streamlit Community Cloud (streamlit.io/cloud).

Reuses psx_utils.py (same logic as the companion Colab notebook) so the analysis
shown here is consistent with what's documented/demonstrated in the notebook.
"""

from datetime import datetime, date, timedelta

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

st.set_page_config(page_title="PSX Stock Analytics", page_icon="▪", layout="wide")

# --------------------------------------------------------------------------- #
# Design tokens (kept in sync with .streamlit/config.toml, which sets
# Streamlit's own native widget theme to the same dark palette so selectboxes,
# tabs, sliders, buttons, and alerts all match this custom CSS automatically).
# --------------------------------------------------------------------------- #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600;700&display=swap');

:root {
    --bg-card: #161b22;
    --border: #263041;
    --border-strong: #334155;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #5b6472;
    --accent: #3b82f6;
    --accent-soft: rgba(59, 130, 246, 0.12);
    --up: #4ade80;
    --down: #f87171;
    --hold: #fbbf24;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

/* Extra top clearance so the header never sits under Streamlit Cloud's
   floating Share/star/edit/GitHub toolbar in the top-right corner. */
.block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: none; }

/* Top bar */
.psx-topbar { display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem; }
.psx-mark {
    width: 38px; height: 38px; border-radius: 8px;
    background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.8rem;
    letter-spacing: 0.02em; flex-shrink: 0;
}
.psx-title { font-size: 1.35rem; font-weight: 600; color: var(--text-primary); letter-spacing: -0.01em; line-height: 1.3; }
.psx-eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.09em; color: var(--text-muted); margin-top: 2px; }

.psx-sidebar-mark { display: flex; align-items: center; gap: 9px; margin-bottom: 2px; }
.psx-sidebar-mark .psx-mark { width: 30px; height: 30px; font-size: 0.65rem; }
.psx-sidebar-mark .name { font-size: 1.02rem; font-weight: 600; color: var(--text-primary); letter-spacing: -0.01em; }

.psx-quick-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); font-weight: 600; margin: 10px 0 4px; }

/* Sidebar quick-select + refresh buttons: compact, mono, pill-like */
[data-testid="stSidebar"] .stButton button {
    font-size: 0.76rem; font-family: 'JetBrains Mono', monospace;
    padding: 4px 6px; border-radius: 6px;
}

/* KPI cards — responsive grid that stretches to fill the row at any width */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 0.9rem 0 1.4rem; }
.kpi-card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; }
.kpi-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-muted); font-weight: 600; margin-bottom: 9px; }
.kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 1.5rem; font-weight: 600; color: var(--text-primary); line-height: 1.15; }
.kpi-value.up { color: var(--up); }
.kpi-value.down { color: var(--down); }
.kpi-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; font-weight: 600; margin-top: 7px; }
.kpi-sub.up { color: var(--up); }
.kpi-sub.down { color: var(--down); }
.kpi-sub.neutral { color: var(--text-muted); }

/* Signal badge */
.signal-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 3px 12px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 1.05rem;
}
.signal-badge.buy  { background: rgba(74, 222, 128, 0.12); color: var(--up); }
.signal-badge.hold { background: rgba(251, 191, 36, 0.12); color: var(--hold); }
.signal-badge.sell { background: rgba(248, 113, 113, 0.12); color: var(--down); }

/* Chart explanation panel (sits beside each chart) */
.chart-note {
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 18px; height: 100%;
}
.chart-note .t { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-muted); font-weight: 600; margin-bottom: 10px; }
.chart-note p { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; margin: 0 0 10px; }
.chart-note p:last-child { margin-bottom: 0; }
.chart-note b { color: var(--text-primary); }

/* Sidebar disclaimer — smaller, muted */
.psx-disclaimer { font-size: 0.76rem; line-height: 1.55; color: var(--text-secondary); }
.psx-disclaimer b { color: var(--text-primary); }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--border); }
[data-testid="stSidebar"] hr { border-color: var(--border); }

/* Section headers */
h2, h3 { color: var(--text-primary); font-weight: 600; letter-spacing: -0.005em; }

/* Dataframe corners */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Caching — avoid re-scraping PSX on every interaction. Data refreshes every
# 15 minutes automatically, or immediately via the manual "Refresh now" button.
# --------------------------------------------------------------------------- #

CACHE_TTL_SECONDS = 15 * 60

# Curated liquid names for the sidebar quick-select row — filtered against
# whatever tickers actually exist (bundled/live) so it never offers a dead pick.
POPULAR_CANDIDATES = ["OGDC", "PPL", "LUCK", "ENGRO", "HBL", "MCB", "UBL", "PSO"]


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


# Shared dark-theme layout for every Plotly chart in the app
CHART_LAYOUT = dict(
    plot_bgcolor="#161b22",
    paper_bgcolor="#161b22",
    font=dict(family="Inter, sans-serif", color="#8b949e", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
)
CHART_GRID = "#263041"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

st.sidebar.markdown("""
<div class="psx-sidebar-mark">
  <div class="psx-mark">PSX</div>
  <span class="name">Analytics</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("PSX Data Portal (EOD) · Not a licensed real-time feed")

TICKERS = get_ticker_universe()
POPULAR_TICKERS = [t for t in POPULAR_CANDIDATES if t in TICKERS]

if "ticker" not in st.session_state:
    st.session_state.ticker = POPULAR_TICKERS[0] if POPULAR_TICKERS else TICKERS[0]

if POPULAR_TICKERS:
    st.sidebar.markdown('<div class="psx-quick-label">Most active</div>', unsafe_allow_html=True)
    for row_start in range(0, len(POPULAR_TICKERS), 4):
        row_tickers = POPULAR_TICKERS[row_start:row_start + 4]
        cols = st.sidebar.columns(4)
        for i, t in enumerate(row_tickers):
            if cols[i].button(t, key=f"quick_{t}", use_container_width=True):
                st.session_state.ticker = t
                st.rerun()

st.sidebar.markdown('<div class="psx-quick-label">All stocks</div>', unsafe_allow_html=True)
ticker = st.sidebar.selectbox(
    "Select stock", TICKERS,
    index=TICKERS.index(st.session_state.ticker) if st.session_state.ticker in TICKERS else 0,
    label_visibility="collapsed",
)
st.session_state.ticker = ticker

st.sidebar.markdown('<div class="psx-quick-label">History window</div>', unsafe_allow_html=True)
default_end = date.today()
default_start = default_end - timedelta(days=365 * 3)
dcol1, dcol2 = st.sidebar.columns(2)
start_date = dcol1.date_input("From", value=default_start, max_value=default_end)
end_date = dcol2.date_input("To", value=default_end, max_value=default_end)
# Underlying fetch always pulls enough history for SMA200 + model training;
# the date range above controls what the price chart displays, not the fetch.
years_back = max(3, ((default_end - start_date).days // 365) + 1)

if st.sidebar.button("Refresh data now", use_container_width=True):
    st.cache_data.clear()

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="psx-disclaimer"><b>Disclaimer</b><br>'
    "Educational decision-support tool. Signals are based on historical "
    "technical + ML patterns and are <b>not financial advice</b>. Markets carry "
    "risk — verify independently before acting on real capital.</div>",
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"· cache refreshes every {CACHE_TTL_SECONDS // 60} min")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

st.markdown("""
<div class="psx-topbar">
  <div class="psx-mark">PSX</div>
  <div>
    <div class="psx-title">Analytics &amp; signal dashboard</div>
    <div class="psx-eyebrow">Pakistan Stock Exchange · KSE-100</div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Stock detail", "Watchlist scan", "About"])

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

        decision_class = sig["decision"].lower()
        decision_icon = {"buy": "▲", "hold": "●", "sell": "▼"}[decision_class]
        pct_change = raw["Close"].pct_change().iloc[-1] * 100
        pct_class = "up" if pct_change >= 0 else "down"
        pct_arrow = "▲" if pct_change >= 0 else "▼"
        latest_macd = df["MACD"].iloc[-1]
        macd_class = "up" if latest_macd >= 0 else "down"
        latest_volume = raw["Volume"].iloc[-1]

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
            <div class="kpi-value" style="font-size:1.2rem">{sig['trend']}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">RSI (14)</div>
            <div class="kpi-value">{sig['rsi']:.1f}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Volume</div>
            <div class="kpi-value" style="font-size:1.2rem">{latest_volume:,.0f}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">MACD</div>
            <div class="kpi-value {macd_class}">{latest_macd:+.2f}</div>
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

        # Price + indicators chart, sliced to the selected From/To range,
        # with an explanation panel filling the space beside it.
        chart_df = df.loc[str(start_date):str(end_date)]
        if chart_df.empty:
            chart_df = df

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
                             subplot_titles=("Price + moving averages", "RSI (14)", "MACD"))
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name="Close",
                                  line=dict(color="#3b82f6", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA50"], name="SMA50",
                                  line=dict(color="#fbbf24", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA200"], name="SMA200",
                                  line=dict(color="#5b6472", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], name="RSI",
                                  line=dict(color="#a78bfa", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#f87171", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#4ade80", line_width=1, row=2, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], name="MACD",
                                  line=dict(color="#3b82f6", width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_signal"], name="Signal",
                                  line=dict(color="#fbbf24", width=1.5)), row=3, col=1)
        fig.update_layout(height=700, showlegend=True, **CHART_LAYOUT)
        fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
        fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
        fig.update_annotations(font=dict(color="#8b949e", size=12))

        chart_col, note_col = st.columns([2.2, 1])
        with chart_col:
            st.plotly_chart(fig, use_container_width=True)
        with note_col:
            st.markdown("""
            <div class="chart-note">
              <div class="t">What this shows</div>
              <p><b>Price panel</b> — closing price vs 50-day and 200-day moving averages.
              SMA50 above SMA200 signals an uptrend (golden cross); below signals a downtrend.</p>
              <p><b>RSI panel</b> — momentum, 0–100. Above 70 = overbought (possible pullback),
              below 30 = oversold (possible bounce).</p>
              <p><b>MACD panel</b> — trend momentum. MACD crossing above its signal line
              often precedes upward moves, and vice versa.</p>
            </div>
            """, unsafe_allow_html=True)

        # Backtest
        st.subheader("Backtest — strategy vs buy & hold")
        bt = result["backtest_summary"]
        strat_class = "up" if bt["strategy_return_pct"] >= 0 else "down"
        bh_class = "up" if bt["buy_and_hold_return_pct"] >= 0 else "down"
        st.markdown(f"""
        <div class="kpi-grid">
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
                                   line=dict(color="#3b82f6", width=2)))
        fig2.add_trace(go.Scatter(x=eq.index, y=eq["buy_and_hold"], name="Buy & hold",
                                   line=dict(color="#5b6472", width=2, dash="dot")))
        fig2.update_layout(height=340, xaxis_title="Date", yaxis_title="Portfolio value (PKR)",
                            **CHART_LAYOUT)
        fig2.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)
        fig2.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_GRID)

        eqchart_col, eqnote_col = st.columns([2.2, 1])
        with eqchart_col:
            st.plotly_chart(fig2, use_container_width=True)
        with eqnote_col:
            outperform_html = ("The strategy <b>outperformed</b> passive holding"
                                if bt['strategy_return_pct'] > bt['buy_and_hold_return_pct']
                                else "Passive holding <b>outperformed</b> the strategy")
            st.markdown(f"""
            <div class="chart-note">
              <div class="t">What this shows</div>
              <p>Portfolio value if you'd mechanically followed this model's Buy/Sell
              signals (<b>blue</b>) vs simply buying and holding the stock the whole
              period (<b>gray, dashed</b>).</p>
              <p>{outperform_html} over this backtest window — {bt['num_trades']} trades were made.</p>
            </div>
            """, unsafe_allow_html=True)

        if not result["trades_df"].empty:
            with st.expander("Trade log (backtest period)"):
                st.dataframe(result["trades_df"], use_container_width=True)

        # Data-source notice goes last — supplementary info, not the headline.
        if result.get("source") == "bundled":
            st.info(f"Live PSX fetch unavailable from this server — the analysis above uses "
                     f"a snapshot as of **{result['as_of']}** (bundled with the app). "
                     f"Click 'Refresh data now' in the sidebar to retry live fetch.")

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
                color = {"BUY": "background-color: rgba(74,222,128,0.14); color: #4ade80;",
                         "SELL": "background-color: rgba(248,113,113,0.14); color: #f87171;",
                         "HOLD": "background-color: rgba(251,191,36,0.14); color: #fbbf24;"}.get(row["Decision"], "")
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
