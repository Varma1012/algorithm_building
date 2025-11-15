from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, Arrow, NormalHead
from bokeh.resources import CDN
import ta

# ==== Import advanced 15m technical filter ====
from tfilter import advanced_technical_filter

# ====Import alpaca paper trade ====
from tradingservices import get_alpaca_data, create_alpaca_chart

#====Live forcast====
import asyncio
import threading
from tradingservices import start_live_stream, get_live_price

app = Flask(__name__)

# ---------------- CONFIG FOR ADVANCED FILTER ----------------
CONFIG = {
    "TICKER": "^NSEI",
    "INTERVAL": "15m",

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
    "SQUEEZE_LOOKBACK": 5,

    # Volume Trend
    "OBV_TREND_PERIOD": 20
}

TIMEFRAME_MAP = {
    "1 Minute": ("7d", "1m"),
    "5 Minutes": ("60d", "5m"),
    "15 Minutes": ("60d", "15m"),
    "30 Minutes": ("60d", "30m"),
    "1 Hour": ("60d", "1h"),
    "1 Day": ("6mo", "1d"),
}

CANDLE_WIDTH_MS = {
    "1 Day": 43_200_000,
    "1 Hour": 1_800_000,
    "30 Minutes": 900_000,
    "15 Minutes": 450_000,
    "5 Minutes": 150_000,
    "1 Minute": 30_000
}


# ===================================================================
#                          DATA FETCH
# ===================================================================
def get_nifty_data(period="6mo", interval="1d"):
    print(f"\n=== Fetching NIFTY data: period={period}, interval={interval} ===")

    df = yf.download(
        "^NSEI",
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,
        timeout=60
    )

    if df.empty:
        raise ValueError("No data returned from Yahoo Finance!")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df.rename(columns={"Date": "date", "Datetime": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["Signal"] = None

    # Basic Indicators
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]

    # Detect move bars
    df, _ = detect_large_moves(df)

    return df


# ===================================================================
#                     LARGE MOVE DETECTION
# ===================================================================
def detect_large_moves(df):
    print("\n--- Running detect_large_moves() UPDATED TRUE FIRST-15M VERSION ---")

    df_ind = df.copy()

    # Ensure datetime
    df_ind["date"] = pd.to_datetime(df_ind["date"])
    df_ind.set_index("date", inplace=True)

    # -------------------------------------------------
    # INDICATORS
    # -------------------------------------------------
    rsi = ta.momentum.RSIIndicator(df_ind["Close"], window=14).rsi()
    df_ind["RSI"] = rsi
    df_ind["RSI_MEAN"] = rsi.rolling(14).mean()
    df_ind["RSI_DIFF"] = df_ind["RSI"] - df_ind["RSI_MEAN"]

    bb = ta.volatility.BollingerBands(df_ind["Close"], window=20, window_dev=2)
    df_ind["BB_MID"] = bb.bollinger_mavg()
    df_ind["BB_UP"] = bb.bollinger_hband()
    df_ind["BB_LOW"] = bb.bollinger_lband()

    kc = ta.volatility.KeltnerChannel(
        high=df_ind["High"], low=df_ind["Low"], close=df_ind["Close"],
        window=20, window_atr=10
    )
    df_ind["KC_UP"] = kc.keltner_channel_hband()
    df_ind["KC_LOW"] = kc.keltner_channel_lband()

    obv = ta.volume.OnBalanceVolumeIndicator(
        close=df_ind["Close"], volume=df_ind["Volume"]
    ).on_balance_volume()

    df_ind["OBV"] = obv
    df_ind["OBV_SLOPE"] = df_ind["OBV"].diff()
    df_ind["OBV_TREND_UP"] = df_ind["OBV_SLOPE"] > 0

    df_ind["SOFT_SQ"] = (df_ind["BB_LOW"] > df_ind["KC_LOW"]) & (df_ind["BB_UP"] < df_ind["KC_UP"])
    df_ind["BREAKOUT_SOFT"] = df_ind["SOFT_SQ"].shift(1) & (~df_ind["SOFT_SQ"])

    # -------------------------------------------------
    # DETECT 1-HOUR MOVES ≥ 0.35%
    # -------------------------------------------------
    df_1h = df_ind.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }).dropna()

    df_1h["pct_change"] = (df_1h["Close"] - df_1h["Open"]) / df_1h["Open"] * 100
    moves = df_1h[abs(df_1h["pct_change"]) >= 0.35]

    print(f"Detected {len(moves)} moves ≥ 0.35%")

    # -------------------------------------------------
    # GET TRUE FIRST 15-MIN CANDLE OF THAT HOUR
    # -------------------------------------------------
    move_records = []

    for move_time in moves.index:

        hour_start = move_time
        hour_end = move_time + pd.Timedelta(hours=1)

        candles_in_hour = df_ind.loc[hour_start:hour_end]

        if candles_in_hour.empty:
            print("No 15m candles inside hour block → skipping", move_time)
            continue

        # REAL TIMESTAMP OF FIRST 15m CANDLE
        first_15m = candles_in_hour.index.min()

        # Fallback if missing
        if first_15m not in df_ind.index:
            try:
                idx = df_ind.index.searchsorted(first_15m)
                first_15m = df_ind.index[idx]
            except:
                print("Skipping missing timestamp:", first_15m)
                continue

        row = df_ind.loc[first_15m]

        # Store record
        record = {"Time": first_15m}
        for col in df_ind.columns:
            record[col] = row[col]

        record["Pct_Change_1h"] = moves.loc[move_time, "pct_change"]
        move_records.append(record)

        # Mark on main df
        nearest_df_idx = df["date"].sub(first_15m).abs().idxmin()

        # SAFE SIGNAL HANDLING (DON’T OVERWRITE)
        current_signal = df.loc[nearest_df_idx, "Signal"]

        if pd.isna(current_signal) or current_signal == "":
            df.loc[nearest_df_idx, "Signal"] = "MOVE"
        else:
            df.loc[nearest_df_idx, "Signal"] = str(current_signal) + "|MOVE"


    # -------------------------------------------------
    # EXPORT EXCEL (FIXED TIMEZONE ERROR)
    # -------------------------------------------------
    if move_records:
        move_df = pd.DataFrame(move_records)
        

        # REMOVE TIMEZONE FROM ALL DATETIME COLUMNS
        for col in move_df.columns:
            if pd.api.types.is_datetime64_any_dtype(move_df[col]):
                try:
                    move_df[col] = move_df[col].dt.tz_convert(None)
                except:
                    try:
                        move_df[col] = move_df[col].dt.tz_localize(None)
                    except:
                        pass

        filename = f"detected_moves_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        move_df.to_excel(filename, index=False)
        print(f"💾 Exported {len(move_df)} rows → {filename}")

    else:
        print("⚠ No moves → Excel not generated")

    return df, moves






# ===================================================================
#                      BOKEH CHART
# ===================================================================
def create_bokeh_chart(df, timeframe_label):
    df = df.copy()
    df = df.tail(300)

    df["inc"] = df["Close"] > df["Open"]
    df["dec"] = df["Close"] <= df["Open"]

    # ------------------------------
    # X-AXIS HANDLING
    # ------------------------------
    intraday = timeframe_label in ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"]

    if intraday:
        df["x"] = list(range(len(df)))     # numeric index → REQUIRED for arrows
        x_axis = "x"
        width = 0.8
        p = figure(x_range=(df["x"].min() - 1, df["x"].max() + 1),
                   width=1200, height=600)
    else:
        x_axis = "date"
        width = CANDLE_WIDTH_MS.get(timeframe_label, 60_000)
        p = figure(x_axis_type="datetime", width=1200, height=600)

    # ------------------------------
    # CANDLESTICKS
    # ------------------------------
    inc = ColumnDataSource(df[df["inc"]])
    dec = ColumnDataSource(df[df["dec"]])

    p.segment(x_axis, "High", x_axis, "Low", color="black", source=inc)
    p.vbar(x_axis, width=width, top="Close", bottom="Open",
           fill_color="#00FF00", line_color="black", source=inc)

    p.segment(x_axis, "High", x_axis, "Low", color="black", source=dec)
    p.vbar(x_axis, width=width, top="Open", bottom="Close",
           fill_color="#FF3333", line_color="black", source=dec)

    # ==========================================================
    #      ARROWS FOR SIGNALS  (NOW WILL SHOW CORRECTLY)
    # ==========================================================
    bull = df[df["Signal"] == "BULLISH_PREDICTED_MOVE"]
    bear = df[df["Signal"] == "BEARISH_PREDICTED_MOVE"]

    # Green UP arrow → placed BELOW candle
    for _, r in bull.iterrows():
        p.add_layout(
            Arrow(
                end=NormalHead(fill_color="green", size=14),
                x_start=r[x_axis], y_start=r["Low"] - (abs(r["Low"] * 0.01)),
                x_end=r[x_axis],   y_end=r["Low"]
            )
        )

    # Red DOWN arrow → placed ABOVE candle
    for _, r in bear.iterrows():
        p.add_layout(
            Arrow(
                end=NormalHead(fill_color="red", size=14),
                x_start=r[x_axis], y_start=r["High"] + (abs(r["High"] * 0.01)),
                x_end=r[x_axis],   y_end=r["High"]
            )
        )

    print("Signal counts ->", df["Signal"].value_counts().to_dict())

    # MOVE SIGNAL — Yellow circle (using scatter, not deprecated circle)
    moves = df[df["Signal"] == "MOVE"]
    if len(moves):
        p.scatter(
            x=moves[x_axis],
            y=moves["Close"],
            size=9,
            color="yellow",
            marker="circle",
            legend_label="MOVE"
        )

    # Hover Tool
    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F %H:%M}"),
            ("Open", "@Open"), ("High", "@High"),
            ("Low", "@Low"), ("Close", "@Close"),
            ("Signal", "@Signal"),
        ],
        formatters={"@date": "datetime"}
    )
    p.add_tools(hover)

    script, div = components(p)
    return script, div


#-----Live price---------
@app.route("/liveprice/<symbol>")
def live_price(symbol):
    return get_live_price(symbol) or {"error": "no price yet"}

# ===================================================================
#                         HOME ROUTE
@app.route("/")
def home():
    try:
        # ----------------------------------------------------
        # 1️⃣ Read timeframe selection from user
        # ----------------------------------------------------
        timeframe_label = request.args.get("timeframe", "1 Day")
        period, interval = TIMEFRAME_MAP.get(timeframe_label, ("6mo", "1d"))

        # ----------------------------------------------------
        # 2️⃣ Fetch main chart data (1m/5m/15m/hour/day)
        # ----------------------------------------------------
        df = get_nifty_data(period=period, interval=interval)

        # ----------------------------------------------------
        # 3️⃣ Run 15-minute advanced technical filter
        #     (Runs ALWAYS, regardless of chart timeframe)
        # ----------------------------------------------------
        print("\n=== Running 15m Technical Filter (called inside home()) ===")
        df_15m = advanced_technical_filter(CONFIG)

        # ----------------------------------------------------
        # 4️⃣ Prepare technical signals for the sidebar table
        # ----------------------------------------------------
        if isinstance(df_15m, pd.DataFrame) and "Signal" in df_15m.columns:

            # Fix timezone → required before merging
            if "date" in df_15m.columns:
                if pd.api.types.is_datetime64_any_dtype(df_15m["date"]):
                    try:
                        df_15m["date"] = df_15m["date"].dt.tz_convert(None)
                    except:
                        df_15m["date"] = df_15m["date"].dt.tz_localize(None)

            # Sidebar signal list
            tech_signals = (
                df_15m[df_15m["Signal"].notna()]
                .tail(10)
                .reset_index()
                .rename(columns={df_15m.index.name: "date"})
                .to_dict(orient="records")
            )
        else:
            tech_signals = []

        # ----------------------------------------------------
        # 5️⃣ Merge 15m technical signals INTO MAIN DF
        #     (FULL FIX → Force both sides to timezone-naive)
        # ----------------------------------------------------
        if isinstance(df_15m, pd.DataFrame) and "Signal" in df_15m.columns:

            # 🔥 Force df main timestamps to tz-naive
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

            # 🔥 Force df_15m timestamps to tz-naive
            if "date" in df_15m.columns:
                df_15m["date"] = (
                    pd.to_datetime(df_15m["date"], errors="coerce")
                    .dt.tz_localize(None)
                )

            # Now merge — GUARANTEED SAFE (both datetime64[ns])
            df_merged = df.merge(
                df_15m[["date", "Signal"]],
                on="date",
                how="left",
                suffixes=("", "_15m")
            )

            # Prefer main signal
            df_merged["Signal"] = df_merged.apply(
                lambda r: r["Signal"] if pd.notna(r["Signal"]) else r["Signal_15m"],
                axis=1
            )

            df = df_merged.drop(columns=["Signal_15m"], errors="ignore")


        # ----------------------------------------------------
        # 6️⃣ Get recent final signals for the table under chart
        # ----------------------------------------------------
        recent_signals = df[df["Signal"].notna()].tail(10).reset_index()

        print("\nFINAL SIDEBAR tech_signals →", tech_signals)
        print("FINAL MERGED SIGNAL COUNTS →", df["Signal"].value_counts().to_dict())

        # ----------------------------------------------------
        # 7️⃣ Build Bokeh chart
        # ----------------------------------------------------
        script, div = create_bokeh_chart(df, timeframe_label)

        #-----------------------------------------------------
        # 7.1 Alpaca call
        #-----------------------------------------------------
        # ---- Alpaca Paper Trading Chart below main chart ----
        alpaca_df = get_alpaca_data("AAPL", timeframe="15m")
        alpaca_script, alpaca_div = create_alpaca_chart(alpaca_df,signals=df_15m)


        # ----------------------------------------------------
        # 8️⃣ Render HTML template
        # ----------------------------------------------------
        return render_template(
            "index.html",
            script=script,
            div=div,
            cdn=CDN.render(),
            current_timeframe=timeframe_label,
            timeframes=list(TIMEFRAME_MAP.keys()),
            signals=recent_signals,
            tech_signals=tech_signals,
            alpaca_script=alpaca_script,
            alpaca_div=alpaca_div
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3 style='color:red'>Error: {e}</h3>"


if __name__ == "__main__":
    start_live_stream("AAPL")
    app.run(debug=True, use_reloader=False)
