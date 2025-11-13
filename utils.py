import yfinance as yf
import pandas as pd
import ta
import numpy as np

def get_nifty_data(period="1y", interval="1d"):
    """Download NIFTY 50 data via yfinance and flatten columns if MultiIndex."""
    df = yf.download("^NSEI", period=period, interval=interval, progress=False)
    df.dropna(inplace=True)

    # If data has multi-level columns, flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Force Close to be a flat Series
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()

    return df


def add_indicators(df: pd.DataFrame):
    # Ensure numeric and 1D
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Close"] = df["Close"].astype(float)
    df["Close"] = df["Close"].values.reshape(-1)  # Flatten ndarray

    # Simple moving averages
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # RSI (Relative Strength Index)
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()

    return df