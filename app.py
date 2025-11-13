from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import os
from openpyxl import load_workbook
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, Span, Range1d, DatetimeTickFormatter
from bokeh.resources import CDN
import ta

app = Flask(__name__)

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

# ------------------ DATA FETCH ------------------
def get_nifty_data(period="6mo", interval="1d"):
    print(f"\n=== Fetching NIFTY data: period={period}, interval={interval} ===")
    df = yf.download("^NSEI", period=period, interval=interval,
                     progress=False, auto_adjust=False, timeout=60)

    print(f"Raw data fetched: {len(df)} rows")
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance!")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)

    df.reset_index(inplace=True)
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'date'}, inplace=True)
    elif 'Datetime' in df.columns:
        df.rename(columns={'Datetime': 'date'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_convert('UTC').dt.tz_localize(None)

    print("Data after cleaning:")
    print(df.head(3))

    # Indicators
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["BB_MID"] = df["Close"].rolling(20).mean()
    df["BB_STD"] = df["Close"].rolling(20).std()
    df["BB_UPPER"] = df["BB_MID"] + 2 * df["BB_STD"]
    df["BB_LOWER"] = df["BB_MID"] - 2 * df["BB_STD"]

    df, moves = detect_large_moves(df)

    print(f"✅ Final DF ready: {len(df)} rows, {len(moves)} detected large moves.\n")
    return df


def detect_large_moves(df):
    print("\n--- Running detect_large_moves() ---")
    df_ind = df.copy()
    df_ind.set_index("date", inplace=True)

    # === Technical Indicators ===
    # RSI
    rsi = ta.momentum.RSIIndicator(df_ind["Close"], window=14)
    df_ind["RSI"] = rsi.rsi()
    df_ind["RSI_MA"] = df_ind["RSI"].rolling(14).mean()
    df_ind["RSI_DIFF"] = df_ind["RSI"] - df_ind["RSI_MA"]

    # SuperTrend (ATR 10, multiplier 3)
    atr = ta.volatility.AverageTrueRange(df_ind["High"], df_ind["Low"], df_ind["Close"], window=10)
    hl2 = (df_ind["High"] + df_ind["Low"]) / 2
    mult = 3
    upperband = hl2 + mult * atr.average_true_range()
    lowerband = hl2 - mult * atr.average_true_range()
    df_ind["SuperTrend"] = np.nan
    df_ind["Trend_Up"] = True
    for i in range(1, len(df_ind)):
        prev_st = df_ind["SuperTrend"].iloc[i - 1] if i > 0 else hl2.iloc[i]
        if df_ind["Close"].iloc[i] > upperband.iloc[i - 1]:
            df_ind.loc[df_ind.index[i], "SuperTrend"] = lowerband.iloc[i]
            df_ind.loc[df_ind.index[i], "Trend_Up"] = True
        elif df_ind["Close"].iloc[i] < lowerband.iloc[i - 1]:
            df_ind.loc[df_ind.index[i], "SuperTrend"] = upperband.iloc[i]
            df_ind.loc[df_ind.index[i], "Trend_Up"] = False
        else:
            df_ind.loc[df_ind.index[i], "SuperTrend"] = prev_st
            df_ind.loc[df_ind.index[i], "Trend_Up"] = df_ind["Trend_Up"].iloc[i - 1]

    # Minutes since last SuperTrend change
    df_ind["Mins_Since_Trend_Change"] = 0
    last_change = df_ind.index[0]
    for i in range(1, len(df_ind)):
        if df_ind["Trend_Up"].iloc[i] != df_ind["Trend_Up"].iloc[i - 1]:
            last_change = df_ind.index[i]
        df_ind.loc[df_ind.index[i], "Mins_Since_Trend_Change"] = (
            df_ind.index[i] - last_change
        ).total_seconds() / 60

    # DMI
    dmi = ta.trend.ADXIndicator(df_ind["High"], df_ind["Low"], df_ind["Close"], window=14)
    df_ind["+DI"] = dmi.adx_pos()
    df_ind["-DI"] = dmi.adx_neg()

    # MACD
    macd = ta.trend.MACD(df_ind["Close"], window_slow=26, window_fast=12, window_sign=9)
    df_ind["MACD"] = macd.macd()
    df_ind["MACD_SIGNAL"] = macd.macd_signal()

    # 1-hour resample to find move bars
    df_1h = df_ind.resample("1h").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    }).dropna()

    df_1h["pct_change"] = ((df_1h["Close"] - df_1h["Open"]) / df_1h["Open"]) * 100
    moves = df_1h[abs(df_1h["pct_change"]) >= 0.35]  # threshold
    print(f"Detected moves (>=0.35%): {len(moves)} rows")

    move_records = []
    for move_time in moves.index:
        first_min = move_time.replace(minute=0, second=0)
        nearest_idx = df_ind.index.get_indexer([first_min], method='nearest')[0]
        row = df_ind.iloc[nearest_idx]

        move_records.append({
            "Time": first_min,
            "Open": row["Open"],
            "High": row["High"],
            "Low": row["Low"],
            "Close": row["Close"],
            "RSI": row.get("RSI", None),
            "RSI_MA": row.get("RSI_MA", None),
            "RSI_DIFF": row.get("RSI_DIFF", None),
            "SuperTrend": row.get("SuperTrend", None),
            "Trend_Up": row.get("Trend_Up", None),
            "Mins_Since_Trend_Change": row.get("Mins_Since_Trend_Change", None),
            "+DI": row.get("+DI", None),
            "-DI": row.get("-DI", None),
            "MACD": row.get("MACD", None),
            "MACD_SIGNAL": row.get("MACD_SIGNAL", None),
            "Volume": row.get("Volume", None),
            "Pct_Change": moves.loc[move_time, "pct_change"]
        })

        nearest_idx_df = df["date"].sub(first_min).abs().idxmin()
        df.loc[nearest_idx_df, "Signal"] = "MOVE"

    # Save to Excel
    if move_records:
        move_df = pd.DataFrame(move_records)
        output_path = "detected_moves.xlsx"
        if os.path.exists(output_path):
            with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                sheet = writer.sheets.get("Sheet1")
                startrow = sheet.max_row if sheet else 0
                move_df.to_excel(writer, sheet_name="Sheet1", startrow=startrow, header=False, index=False)
        else:
            move_df.to_excel(output_path, index=False)
            print(f"💾 Stored {len(move_df)} move(s) with indicators → {output_path}")
    else:
        print("⚠️ No moves detected to save.")

    return df, moves


# ------------------ BOKEH CHART ------------------
def create_bokeh_chart(df, timeframe_label):
    print(f"\n--- Creating chart for {timeframe_label} ---")
    df = df.copy()
    tail_sizes = {"1 Minute": 300, "5 Minutes": 300, "15 Minutes": 300, "30 Minutes": 300, "1 Hour": 200, "1 Day": 180}
    df = df.tail(tail_sizes.get(timeframe_label, 200))

    df["inc"] = df["Close"] > df["Open"]
    df["dec"] = df["Close"] <= df["Open"]

    intraday = timeframe_label in ["1 Minute", "5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"]

    if intraday:
        df["x_str"] = df["date"].dt.strftime("%Y-%m-%d %H:%M")
        x_axis = "x_str"
        width = 0.8
        p = figure(
            x_range=df[x_axis].tolist(),
            width=1200,
            height=600,
            title=f"NIFTY 50 — {timeframe_label}",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair"
        )
        p.xaxis.major_label_orientation = 1.0
    else:
        x_axis = "date"
        width = CANDLE_WIDTH_MS.get(timeframe_label, 60_000)
        p = figure(
            x_axis_type="datetime",
            width=1200,
            height=600,
            title=f"NIFTY 50 — {timeframe_label}",
            tools="pan,wheel_zoom,box_zoom,reset,save,crosshair"
        )
        p.x_range = Range1d(start=df['date'].min(), end=df['date'].max())
        p.xaxis.formatter = DatetimeTickFormatter(days="%d %b", months="%b %Y",
                                                  hours="%H:%M", minutes="%H:%M")

    move_points = df[df["Signal"] == "MOVE"]
    print(f"Chart data: total={len(df)}, MOVE points={len(move_points)}")

    inc_source = ColumnDataSource(df[df["inc"]])
    dec_source = ColumnDataSource(df[df["dec"]])

    # Candles
    p.segment(x_axis, "High", x_axis, "Low", color="black", source=inc_source)
    p.vbar(x_axis, width=width, top="Close", bottom="Open",
           fill_color="#00FF00", line_color="black", source=inc_source)
    p.segment(x_axis, "High", x_axis, "Low", color="black", source=dec_source)
    p.vbar(x_axis, width=width, top="Open", bottom="Close",
           fill_color="#FF3333", line_color="black", source=dec_source)

    x_vals = df[x_axis] if intraday else df["date"]
    p.line(x_vals, df["SMA50"], color="cyan", legend_label="SMA 50", line_width=2)
    p.line(x_vals, df["SMA200"], color="orange", legend_label="SMA 200", line_width=2)
    p.line(x_vals, df["BB_UPPER"], color="gray", line_dash="dashed", legend_label="BB Upper")
    p.line(x_vals, df["BB_LOWER"], color="gray", line_dash="dashed", legend_label="BB Lower")

    # Highlight ≥0.35% moves
    if not move_points.empty:
        p.circle(
            x=move_points[x_axis],
            y=move_points["Close"],
            size=10,
            color="yellow",
            legend_label="≥0.35% Move",
        )
        print("✅ Yellow move points plotted successfully.")
    else:
        print("⚠️ No move points to plot on chart!")

    avg_price = df["Close"].mean()
    hline = Span(location=avg_price, dimension="width",
                 line_color="yellow", line_dash="dashed", line_width=2)
    p.add_layout(hline)

    hover = HoverTool(
        tooltips=[
            ("Date", "@date{%F %H:%M}"),
            ("Open", "@Open{0.2f}"),
            ("High", "@High{0.2f}"),
            ("Low", "@Low{0.2f}"),
            ("Close", "@Close{0.2f}"),
            ("SMA 50", "@SMA50{0.2f}"),
            ("SMA 200", "@SMA200{0.2f}"),
            ("BB Upper", "@BB_UPPER{0.2f}"),
            ("BB Lower", "@BB_LOWER{0.2f}"),
            ("Signal", "@Signal"),
        ],
        formatters={"@date": "datetime"} if not intraday else {},
        mode="vline"
    )
    p.add_tools(hover)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Date / Time"
    p.yaxis.axis_label = "Price (INR)"

    script, div = components(p)
    return script, div


# ------------------ FLASK ROUTE ------------------
@app.route("/")
def home():
    try:
        timeframe_label = request.args.get("timeframe", "1 Day")
        period, interval = TIMEFRAME_MAP.get(timeframe_label, ("6mo", "1d"))
        df = get_nifty_data(period=period, interval=interval)
        script, div = create_bokeh_chart(df, timeframe_label)
        return render_template("index.html",
                               script=script,
                               div=div,
                               cdn=CDN.render(),
                               current_timeframe=timeframe_label,
                               timeframes=list(TIMEFRAME_MAP.keys()))
    except Exception as e:
        import traceback
        print("❌ ERROR:", e)
        traceback.print_exc()
        return f"<h3 style='color:red'>Error: {e}</h3>"


if __name__ == "__main__":
    app.run(debug=True)
