# =============================================================
#  tradingservices.py
#  Alpaca Paper Trading Services + Live Streaming + Signal Overlay
# =============================================================

import os
import datetime as dt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import asyncio
import threading

# ---------------------- Alpaca Trading APIs ----------------------
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# ---------------------- Historical Data --------------------------
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ---------------------- WebSocket Streaming ----------------------
from alpaca.data.live import StockDataStream

# ---------------------- For Chart Rendering ----------------------
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, Arrow, NormalHead

load_dotenv()

# =============================================================
# CONFIG — Alpaca API KEYS
# =============================================================
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Trading Client
trading_client = TradingClient(API_KEY, SECRET_KEY)

# Historical Data Client
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

# GLOBALS
LIVE_PRICES = {}
LIVE_STREAM = None
STREAM_THREAD = None
STREAM_LOOP = None  # Keep reference to the loop for stopping

# =============================================================
# 1️⃣ HISTORICAL DATA
# =============================================================
def get_alpaca_data(symbol="AAPL", timeframe="15m", bars=200):
    tf_map = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame.Minute,
        "15m": TimeFrame.Minute,
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day
    }

    if timeframe not in tf_map:
        raise ValueError("Invalid timeframe")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf_map[timeframe],
        start=dt.datetime.now() - dt.timedelta(days=7),
        limit=bars
    )

    bars = data_client.get_stock_bars(request).df
    bars.reset_index(inplace=True)
    bars.rename(columns={"timestamp": "date"}, inplace=True)
    bars["date"] = pd.to_datetime(bars["date"])
    return bars[["date", "open", "high", "low", "close", "volume"]]

# =============================================================
# 2️⃣ ALPACA CHART WITH SIGNALS
# =============================================================
def create_alpaca_chart(df, signals=None, title="Alpaca Chart"):
    df = df.copy()
    df["inc"] = df["close"] > df["open"]
    df["dec"] = df["close"] <= df["open"]
    df["x"] = list(range(len(df)))  # numeric index for arrows

    # Ensure 'date' column in df is tz-naive to match signals
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert(None)

    p = figure(
        x_range=(df["x"].min() - 1, df["x"].max() + 1),
        width=1200, height=400, title=title
    )

    inc = ColumnDataSource(df[df["inc"]])
    dec = ColumnDataSource(df[df["dec"]])

    # Candlestick segments
    p.segment("x", "high", "x", "low", color="black", source=inc)
    p.vbar("x", width=0.8, top="close", bottom="open",
           fill_color="#00FF00", line_color="black", source=inc)

    p.segment("x", "high", "x", "low", color="black", source=dec)
    p.vbar("x", width=0.8, top="open", bottom="close",
           fill_color="#FF3333", line_color="black", source=dec)

    # ------------------------------
    # Overlay Signals (align by nearest timestamp)
    # ------------------------------
    if signals is not None and not signals.empty:
        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"])

        # Make signals tz-naive as well (if somehow tz-aware)
        if signals["date"].dt.tz is not None:
            signals["date"] = signals["date"].dt.tz_convert(None)

        for sig_type, color, offset in [
            ("BULLISH_PREDICTED_MOVE", "green", -0.01),
            ("BEARISH_PREDICTED_MOVE", "red", 0.01),
            ("MOVE", "yellow", 0)
        ]:
            sig_df = signals[signals["Signal"] == sig_type]
            for _, r in sig_df.iterrows():
                nearest_idx = df["date"].sub(r["date"]).abs().idxmin()
                x = df.loc[nearest_idx, "x"]
                y_low = df.loc[nearest_idx, "low"]
                y_high = df.loc[nearest_idx, "high"]
                y_close = df.loc[nearest_idx, "close"]

                if sig_type == "BULLISH_PREDICTED_MOVE":
                    p.add_layout(
                        Arrow(
                            end=NormalHead(fill_color=color, size=14),
                            x_start=x, y_start=y_low + offset * y_low,
                            x_end=x, y_end=y_low
                        )
                    )
                elif sig_type == "BEARISH_PREDICTED_MOVE":
                    p.add_layout(
                        Arrow(
                            end=NormalHead(fill_color=color, size=14),
                            x_start=x, y_start=y_high + offset * y_high,
                            x_end=x, y_end=y_high
                        )
                    )
                elif sig_type == "MOVE":
                    p.scatter(
                        x=[x], y=[y_close],
                        size=9, color=color, marker="circle", legend_label="MOVE"
                    )

    # Hover Tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F %H:%M}"),
            ("Open", "@open"), ("High", "@high"),
            ("Low", "@low"), ("Close", "@close"),
        ],
        formatters={"@date": "datetime"}
    )
    p.add_tools(hover)

    script, div = components(p)
    return script, div


# =============================================================
# 3️⃣ PLACE ORDERS
# =============================================================
def place_market_buy(symbol, qty=1):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    return trading_client.submit_order(order)


def place_market_sell(symbol, qty=1):
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    return trading_client.submit_order(order)

# =============================================================
# 4️⃣ ACCOUNT + POSITION + ORDERS
# =============================================================
def get_account():
    return trading_client.get_account()


def get_positions():
    return trading_client.get_all_positions()


def get_orders():
    return trading_client.get_orders()

# =============================================================
# 5️⃣ REAL-TIME LIVE PRICE STREAMING
# =============================================================
async def _on_bar(bar):
    global LIVE_PRICES
    LIVE_PRICES[bar.symbol] = {"price": bar.close, "time": bar.timestamp}
    print(f"LIVE UPDATE → {bar.symbol}: {bar.close} @ {bar.timestamp}")


async def _run_stream(symbol):
    global LIVE_STREAM
    LIVE_STREAM = StockDataStream(API_KEY, SECRET_KEY)
    LIVE_STREAM.subscribe_bars(_on_bar, symbol)
    print(f"Subscribed to live bars for {symbol}")
    await LIVE_STREAM.run()


def start_live_stream(symbol):
    global STREAM_THREAD, STREAM_LOOP, LIVE_STREAM
    if STREAM_THREAD and STREAM_THREAD.is_alive():
        print("Stream already running")
        return

    def run_loop():
        global STREAM_LOOP, LIVE_STREAM
        STREAM_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(STREAM_LOOP)
        LIVE_STREAM = StockDataStream(API_KEY, SECRET_KEY)
        LIVE_STREAM.subscribe_bars(_on_bar, symbol)
        print(f"Subscribed to live bars for {symbol}")
        try:
            STREAM_LOOP.run_until_complete(LIVE_STREAM.run())
        except Exception as e:
            print("Stream error:", e)
        finally:
            STREAM_LOOP.close()
            print("Stream loop closed")

    STREAM_THREAD = threading.Thread(target=run_loop, daemon=True)
    STREAM_THREAD.start()


def stop_live_stream():
    global LIVE_STREAM, STREAM_LOOP
    if LIVE_STREAM and STREAM_LOOP:
        asyncio.run_coroutine_threadsafe(LIVE_STREAM.stop(), STREAM_LOOP)
        LIVE_STREAM = None
        print("Live stream stop requested")


def get_live_price(symbol):
    return LIVE_PRICES.get(symbol, None)
