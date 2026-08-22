"""
psx_utils.py
Core logic for PSX Stock Analytics Dashboard.
Shared between the Colab notebook (research/backtesting) and the Streamlit app (deployment).

Data source: `psxdata` package (https://github.com/mtauha/psxdata) — actively maintained,
built specifically to survive PSX's Data Portal HTML changes (dynamic column extraction,
retries with backoff, disk caching). Returns official historical EOD OHLCV data plus a
15-minute-cached live screener snapshot (the closest thing to "live" price data available
without a paid PSX data license).

IMPORTANT: The historical series is EOD (end-of-day) data. `get_live_quote()` gives the
latest screener snapshot (refreshed every 15 min while cached) — useful as a "current
price" reference, but still not a licensed real-time tick feed. PSX's true real-time/
delayed live feed is a paid, licensed commercial product (see
psx.com.pk/product-and-services/data-services-vending). This tool is for educational/
decision-support purposes only and is not financial advice.
"""

from datetime import date, timedelta
import warnings

import numpy as np
import pandas as pd
import psxdata
from psxdata import exceptions as psx_exceptions

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# Liquid, well-known fallback tickers, used only if the live KSE-100 constituent
# fetch fails for some reason.
FALLBACK_TICKERS = [
    "OGDC", "PPL", "LUCK", "ENGRO", "HBL", "MCB", "UBL", "FFC",
    "PSO", "MARI", "HUBC", "MEBL", "SYS", "TRG", "DGKC", "FCCL",
    "NBP", "BAHL", "EPCL", "KOHC",
]


# --------------------------------------------------------------------------- #
# Ticker universe
# --------------------------------------------------------------------------- #

def get_kse100_tickers(limit: int = 20) -> list:
    """
    Fetch the current live KSE-100 index constituents from PSX.
    Falls back to a hardcoded liquid-stock list if the live fetch fails.
    """
    try:
        idx_df = psxdata.indices("KSE100")
        if idx_df is not None and not idx_df.empty and "symbol" in idx_df.columns:
            tickers = idx_df["symbol"].dropna().astype(str).str.upper().tolist()
            if tickers:
                return tickers[:limit] if limit else tickers
    except Exception as exc:  # noqa: BLE001
        print(f"[get_kse100_tickers] Live fetch failed ({exc}); using fallback list.")
    return FALLBACK_TICKERS[:limit] if limit else FALLBACK_TICKERS


DEFAULT_TICKERS = FALLBACK_TICKERS  # safe default at import time; call get_kse100_tickers() for live list


# --------------------------------------------------------------------------- #
# Data fetching
# --------------------------------------------------------------------------- #

def fetch_stock_data(ticker: str, years_back: int = 3) -> pd.DataFrame:
    """
    Fetch historical EOD OHLCV data for a single PSX ticker via psxdata.

    Returns a DataFrame indexed by Date with columns: Open, High, Low, Close, Volume.
    Returns an empty DataFrame if the fetch fails (bad ticker, network issue, etc.)
    """
    end = date.today()
    start = end - timedelta(days=365 * years_back)
    try:
        df = psxdata.stocks(ticker, start=start, end=end)
        if df is None or df.empty:
            print(f"[fetch_stock_data] {ticker}: no rows returned for this range.")
            return pd.DataFrame()

        df = df.rename(columns={
            "date": "Date", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[keep_cols]

    except psx_exceptions.InvalidSymbolError:
        print(f"[fetch_stock_data] '{ticker}' is not a valid/recognized PSX symbol.")
    except psx_exceptions.DelistedSymbolError:
        print(f"[fetch_stock_data] '{ticker}' appears to be delisted.")
    except psx_exceptions.PSXAuthError as exc:
        print(f"[fetch_stock_data] {ticker}: PSX rejected the request (auth/403): {exc}")
        print("  -> This can happen from data-center IPs (e.g. some cloud sandboxes). "
              "Try again from Colab directly — it usually has a normal residential/cloud IP "
              "that PSX doesn't block.")
    except psx_exceptions.PSXConnectionError as exc:
        print(f"[fetch_stock_data] {ticker}: network/connection error after retries: {exc}")
    except psx_exceptions.PSXServerError as exc:
        print(f"[fetch_stock_data] {ticker}: PSX server error (5xx) after retries: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface any other failure
        print(f"[fetch_stock_data] Unexpected failure for {ticker}: {type(exc).__name__}: {exc}")

    return pd.DataFrame()


def get_live_quote(ticker: str) -> dict:
    """
    Fetch the latest screener snapshot for a ticker (cached ~15 min by psxdata).
    This is the closest available "current price" reference without a paid feed.
    Returns {} if unavailable.
    """
    try:
        q = psxdata.quote(ticker)
        if q is None or q.empty:
            return {}
        return q.iloc[0].to_dict()
    except Exception as exc:  # noqa: BLE001
        print(f"[get_live_quote] Failed for {ticker}: {type(exc).__name__}: {exc}")
        return {}


def diagnose_connection(ticker: str = "OGDC") -> None:
    """
    Run this if fetch_stock_data() keeps returning empty results.
    Reports exactly which stage failed and why, using psxdata's typed exceptions.
    """
    print(f"Testing psxdata connectivity for '{ticker}'...\n")

    print("1. Fetching symbol list...")
    try:
        all_syms = psxdata.tickers()
        print(f"   OK — {len(all_syms)} symbols returned.")
        if ticker.upper() not in [s.upper() for s in all_syms]:
            print(f"   WARNING: '{ticker}' not found in symbol list. Check spelling.")
        else:
            print(f"   '{ticker}' confirmed valid.")
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {type(exc).__name__}: {exc}")

    print("\n2. Fetching a short historical window (last 30 days)...")
    try:
        df = psxdata.stocks(ticker, start=date.today() - timedelta(days=30), end=date.today())
        print(f"   OK — {len(df)} rows returned.")
        if not df.empty:
            print(df.tail(3))
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {type(exc).__name__}: {exc}")

    print("\n3. Fetching live quote/screener snapshot...")
    try:
        q = psxdata.quote(ticker)
        print(f"   OK — {'found' if not q.empty else 'symbol not in screener'}")
        if not q.empty:
            print(q.iloc[0])
    except Exception as exc:  # noqa: BLE001
        print(f"   FAILED: {type(exc).__name__}: {exc}")

    print(
        "\nIf all three steps failed with a connection/auth error, the network you're "
        "running from (not the code) is likely the issue — retry from a different "
        "network/runtime. If only specific steps failed, that narrows down whether "
        "it's a symbol problem vs. a PSX-side outage."
    )


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA50, SMA200, RSI(14), and MACD(12,26,9) columns to a price DataFrame."""
    df = df.copy()

    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA200"] = df["Close"].rolling(window=200, min_periods=1).mean()

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)  # neutral default when undefined

    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["Return_1d"] = df["Close"].pct_change()
    df["Volume_change"] = df["Volume"].pct_change()

    return df


FEATURE_COLUMNS = [
    "SMA50", "SMA200", "RSI", "MACD", "MACD_signal", "MACD_hist",
    "Return_1d", "Volume_change",
]


# --------------------------------------------------------------------------- #
# ML model
# --------------------------------------------------------------------------- #

def build_features_and_labels(df: pd.DataFrame):
    """
    Build the ML feature matrix X and next-day-direction labels y.
    Label = 1 if next day's close is higher than today's close, else 0.
    """
    data = df.copy()
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna(subset=FEATURE_COLUMNS + ["Target"])

    X = data[FEATURE_COLUMNS]
    y = data["Target"]
    return X, y, data


def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Time-based train/test split (no shuffling — this is a time series) and train a
    RandomForestClassifier. Returns (model, X_train, X_test, y_train, y_test).
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Signal generation
# --------------------------------------------------------------------------- #

def generate_signal(model, latest_row: pd.Series) -> dict:
    """
    Combine the ML model's probability with technical trend/RSI context into a
    single Buy / Hold / Sell recommendation with a confidence score and plain-
    English reasoning. This is a decision-support heuristic, not financial advice.
    """
    features = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)
    ml_prob_up = model.predict_proba(features)[0][1]  # P(next day up)

    trend_bullish = latest_row["SMA50"] > latest_row["SMA200"]
    rsi = latest_row["RSI"]

    # Weighted composite score: 0 = strongly bearish, 1 = strongly bullish
    trend_score = 1.0 if trend_bullish else 0.0
    if rsi >= 70:
        rsi_score = 0.1   # overbought -> caution
    elif rsi <= 30:
        rsi_score = 0.9   # oversold -> potential buy zone
    else:
        rsi_score = 0.5   # neutral

    composite = 0.5 * ml_prob_up + 0.3 * trend_score + 0.2 * rsi_score

    if composite >= 0.62:
        decision = "BUY"
    elif composite <= 0.42:
        decision = "SELL"
    else:
        decision = "HOLD"

    reasons = []
    reasons.append(f"ML model: {ml_prob_up:.0%} probability of next-day price increase")
    reasons.append(f"Trend: {'Bullish (SMA50 > SMA200)' if trend_bullish else 'Bearish (SMA50 < SMA200)'}")
    if rsi >= 70:
        reasons.append(f"RSI {rsi:.1f} — overbought zone")
    elif rsi <= 30:
        reasons.append(f"RSI {rsi:.1f} — oversold zone")
    else:
        reasons.append(f"RSI {rsi:.1f} — neutral zone")

    return {
        "decision": decision,
        "confidence": round(float(composite), 3),
        "ml_prob_up": round(float(ml_prob_up), 3),
        "trend": "Bullish" if trend_bullish else "Bearish",
        "rsi": round(float(rsi), 1),
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def backtest_signals(model, test_df: pd.DataFrame, buy_threshold: float = 0.62,
                      sell_threshold: float = 0.42, initial_capital: float = 100_000.0):
    """
    Simple long-only backtest over the held-out test period:
    - When flat and composite score >= buy_threshold -> buy (enter position)
    - When holding and composite score <= sell_threshold -> sell (exit position)
    Compares strategy equity curve against a buy-and-hold benchmark.
    """
    cash = initial_capital
    shares = 0.0
    holding = False
    trades = []
    equity_curve = []

    for date_idx, row in test_df.iterrows():
        sig = generate_signal(model, row)
        price = row["Close"]

        if not holding and sig["decision"] == "BUY":
            shares = cash / price
            cash = 0.0
            holding = True
            trades.append({"date": date_idx, "action": "BUY", "price": price})
        elif holding and sig["decision"] == "SELL":
            cash = shares * price
            shares = 0.0
            holding = False
            trades.append({"date": date_idx, "action": "SELL", "price": price})

        equity = cash + shares * price
        equity_curve.append({"date": date_idx, "equity": equity, "price": price})

    equity_df = pd.DataFrame(equity_curve).set_index("date")

    # Buy-and-hold benchmark
    first_price = test_df["Close"].iloc[0]
    equity_df["buy_and_hold"] = (initial_capital / first_price) * test_df["Close"]

    strategy_return = (equity_df["equity"].iloc[-1] / initial_capital - 1) * 100
    bh_return = (equity_df["buy_and_hold"].iloc[-1] / initial_capital - 1) * 100

    summary = {
        "strategy_return_pct": round(float(strategy_return), 2),
        "buy_and_hold_return_pct": round(float(bh_return), 2),
        "num_trades": len(trades),
        "final_equity": round(float(equity_df["equity"].iloc[-1]), 2),
    }

    return equity_df, pd.DataFrame(trades), summary
