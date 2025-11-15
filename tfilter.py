# utils.py
import yfinance as yf
import pandas as pd
import ta
import numpy as np

# ===============================
# CONFIG (Used by advanced filter)
# ===============================

CONFIG = {
    "TICKER": "^NSEI",

    "INTERVAL": "15m",
    "LOOKBACK_DAYS": 30,

    # Adaptive RSI
    "RSI_PERIOD": 14,
    "RSI_ROLLING_WINDOW": 100,
    "RSI_STD_MULTIPLIER": 1.6,

    # Bollinger / Keltner Squeeze
    "BB_PERIOD": 20,
    "BB_DEV": 2.0,
    "KC_PERIOD": 20,
    "KC_ATR_PERIOD": 10,
    "KC_ATR_MULTIPLIER": 1.5,
}


# -----------------------
# Helper utilities
# -----------------------
def _ensure_1d_series(x, index):
    """Return a pandas Series flattened to 1D with the provided index."""
    if isinstance(x, pd.Series):
        s = x.copy()
        s.index = index
        return s
    if isinstance(x, np.ndarray):
        arr = np.asarray(x)
        arr = arr.reshape(-1)  # flatten (N,1) -> (N,)
        return pd.Series(arr, index=index)
    # fallback: try to convert
    return pd.Series(x, index=index)


def _safe_assign(df, colname, values):
    """Assign values to df[colname] ensuring a 1D Series with matching index length."""
    series = _ensure_1d_series(values, df.index)
    # if lengths mismatch, truncate/expand safely
    if len(series) != len(df):
        series = series.reindex(df.index)
    df[colname] = series
    return df


# ========================================================
# 1️⃣ Market Data Fetch (THIS WILL BE USED BY THE FILTER)
# ========================================================
def get_market_data(ticker, interval="15m", lookback_days=30):
    """
    Fetch independent fresh data ONLY for technical analysis.
    Returns a DataFrame with a datetime 'date' column.
    """
    print(f"\n=== Fetching TECHNICAL analysis data: {ticker}, {interval}, {lookback_days}d ===")

    try:
        df = yf.download(
            ticker,
            period=f"{lookback_days}d",
            interval=interval,
            progress=False
        )

        if df is None or df.empty:
            print("⚠ ERROR: No technical data received")
            return pd.DataFrame()

        # Flatten multiindex columns if any
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # ensure index name
        df.index.name = "date"
        df = df.reset_index()

        # Normalize column names to Title case like your app expects
        df.columns = [c if c == "date" else c for c in df.columns]

        # Ensure numeric types for core OHLCV columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # drop rows where price is missing
        df.dropna(subset=["Close", "Open", "High", "Low"], inplace=True)

        return df

    except Exception as e:
        print("⚠ ERROR fetching technical data:", e)
        return pd.DataFrame()


# ========================================================
# 2️⃣ Adaptive RSI Calculation
# ========================================================
def calculate_adaptive_rsi(data, config):
    """
    Expects data indexed by datetime and a 'RSI' column present.
    Produces RSI_MEAN, RSI_STD, ADAPTIVE_OB, ADAPTIVE_OS and ADAPTIVE_SIGNAL.
    """
    rsi = data["RSI"]
    window = int(config.get("RSI_ROLLING_WINDOW", 100))
    multiplier = float(config.get("RSI_STD_MULTIPLIER", 1.6))

    # safe rolling with min_periods=1 so we don't get all NaN early
    data["RSI_MEAN"] = rsi.rolling(window=window, min_periods=1).mean()
    data["RSI_STD"] = rsi.rolling(window=window, min_periods=1).std().fillna(0.0)

    data["ADAPTIVE_OB"] = data["RSI_MEAN"] + data["RSI_STD"] * multiplier
    data["ADAPTIVE_OS"] = data["RSI_MEAN"] - data["RSI_STD"] * multiplier

    # ADAPTIVE_SIGNAL: OVERBOUGHT / OVERSOLD / NEUTRAL
    data["ADAPTIVE_SIGNAL"] = pd.Series(
        np.where(
            rsi >= data["ADAPTIVE_OB"],
            "OVERBOUGHT",
            np.where(rsi <= data["ADAPTIVE_OS"], "OVERSOLD", "NEUTRAL")
        ),
        index=data.index
    )

    return data


# ========================================================
# 3️⃣ ADVANCED TECHNICAL FILTER (SELF-FETCHES 15m DATA)
# ========================================================
def advanced_technical_filter(config):
    print("\n=== Running Advanced Technical Filter (FLEX++ v2.2 Slightly Loose) ===")

    lookback = int(config.get("LOOKBACK_DAYS", 30))
    interval = config.get("INTERVAL", "15m")
    ticker = config.get("TICKER", "^NSEI")

    df = get_market_data(ticker=ticker, interval=interval, lookback_days=lookback)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    def force(x):
        return x if isinstance(x, pd.Series) else pd.Series(np.array(x).squeeze(), index=df.index)

    # ------------------------------------------------
    # INDICATORS
    # ------------------------------------------------
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = force(macd.macd())
    df["MACD_SIGNAL"] = force(macd.macd_signal())
    df["MACD_HIST"] = force(macd.macd_diff())

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA_DIFF"] = (df["EMA20"] - df["EMA50"]) / df["Close"]

    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_UP"] = force(bb.bollinger_hband())
    df["BB_LOW"] = force(bb.bollinger_lband())
    df["BB_MID"] = force(bb.bollinger_mavg())
    df["BB_WIDTH"] = df["BB_UP"] - df["BB_LOW"]
    df["BB_EXPAND"] = df["BB_WIDTH"].diff() > 0

    df["BodyPct"] = (df["Close"] - df["Open"]).abs() / df["Open"]
    df["VolumeAvg"] = df["Volume"].rolling(20).mean()
    df["HighVol"] = df["Volume"] > 1.3 * df["VolumeAvg"]

    st = ta.trend.STCIndicator(close=df["Close"], fillna=True)
    df["SuperTrend"] = st.stc()
    df["ST_BULL"] = df["SuperTrend"] > df["SuperTrend"].shift(1)
    df["ST_BEAR"] = df["SuperTrend"] < df["SuperTrend"].shift(1)

    df["CCI"] = ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20
    ).cci()
    df["CCI_BULL"] = df["CCI"] > 25
    df["CCI_BEAR"] = df["CCI"] < -25

    df["Pivot"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["R1"] = 2 * df["Pivot"] - df["Low"]
    df["S1"] = 2 * df["Pivot"] - df["High"]

    df["BREAK_R1"] = (df["Close"] > df["R1"]) & (df["BodyPct"] > 0.0011)
    df["BREAK_S1"] = (df["Close"] < df["S1"]) & (df["BodyPct"] > 0.0011)

    # Kalman
    def kalman_smooth(series, q=0.00001, r=0.01):
        x, p = 0, 1
        result = []
        for z in series:
            x_p, p_p = x, p + q
            k = p_p / (p_p + r)
            x = x_p + k * (z - x_p)
            p = (1 - k) * p_p
            result.append(x)
        return pd.Series(result, index=series.index)

    df["Kalman"] = kalman_smooth(df["Close"])
    df["KALMAN_UP"] = df["Kalman"] > df["Kalman"].shift(1)
    df["KALMAN_DOWN"] = df["Kalman"] < df["Kalman"].shift(1)

    # Volume Delta
    df["CVD"] = (df["Close"] - df["Open"]) * df["Volume"]
    df["CVD_TREND"] = df["CVD"].rolling(5).mean()
    df["CVD_UP"] = df["CVD_TREND"] > df["CVD_TREND"].shift(1) * 1.015
    df["CVD_DOWN"] = df["CVD_TREND"] < df["CVD_TREND"].shift(1) * 0.985

    # ------------------------------------------------
    # SCORING v2.2 (slightly looser)
    # ------------------------------------------------
    df["BullScore"] = 0
    df["BearScore"] = 0

    # Trend
    df.loc[df["EMA_DIFF"] > 0.00042, "BullScore"] += 1
    df.loc[df["EMA_DIFF"] < -0.00042, "BearScore"] += 1
    df.loc[df["KALMAN_UP"], "BullScore"] += 1
    df.loc[df["KALMAN_DOWN"], "BearScore"] += 1

    # RSI
    df.loc[df["RSI"] > 59, "BullScore"] += 1
    df.loc[df["RSI"] < 41, "BearScore"] += 1

    # MACD slightly easier
    df.loc[(df["MACD"] > df["MACD_SIGNAL"]) & (df["MACD_HIST"] > -0.01), "BullScore"] += 1
    df.loc[(df["MACD"] < df["MACD_SIGNAL"]) & (df["MACD_HIST"] < 0.01), "BearScore"] += 1

    # SuperTrend
    df.loc[df["ST_BULL"], "BullScore"] += 1
    df.loc[df["ST_BEAR"], "BearScore"] += 1

    # CCI slightly looser
    df.loc[df["CCI_BULL"], "BullScore"] += 1
    df.loc[df["CCI_BEAR"], "BearScore"] += 1

    # Pivot breakout
    df.loc[df["BREAK_R1"], "BullScore"] += 1
    df.loc[df["BREAK_S1"], "BearScore"] += 1

    # Volume Delta
    df.loc[df["CVD_UP"], "BullScore"] += 1
    df.loc[df["CVD_DOWN"], "BearScore"] += 1

    # Strong candle
    df.loc[df["HighVol"] & (df["Close"] > df["Open"]), "BullScore"] += 1
    df.loc[df["HighVol"] & (df["Close"] < df["Open"]), "BearScore"] += 1

    # ------------------------------------------------
    # FINAL SIGNALS (softened slightly)
    # ------------------------------------------------
    df["Signal"] = None
    df.loc[df["BullScore"] >= 6, "Signal"] = "BULLISH_PREDICTED_MOVE"
    df.loc[df["BearScore"] >= 6, "Signal"] = "BEARISH_PREDICTED_MOVE"

    print("\n=== FINAL FLEX++ STRICT v2.2 SIGNALS ===")
    res = df[df["Signal"].notna()].tail(15)
    print(res[["Open", "Close", "BullScore", "BearScore", "Signal"]]
          if not res.empty else "-> No FLEX++ strict signals")

    return df.reset_index()
