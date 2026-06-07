import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta


# =========================
# 工具函數
# =========================

def get_preset_dates(option: str):
    today = date.today()

    if option == "近一月":
        return today - timedelta(days=30), today
    elif option == "近三月":
        return today - timedelta(days=90), today
    elif option == "近半年":
        return today - timedelta(days=182), today
    elif option == "近一年":
        return today - timedelta(days=365), today
    elif option == "近三年":
        return today - timedelta(days=365 * 3), today
    elif option == "近五年":
        return today - timedelta(days=365 * 5), today
    else:
        return today - timedelta(days=365 * 5), today


def download_data(symbol: str, start_date, end_date):
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()
    return df


def backtest_ma_cross(df, short_ma=5, long_ma=20, initial_capital=1_000_000):
    df = df.copy()

    df["Short_MA"] = df["Close"].rolling(short_ma).mean()
    df["Long_MA"] = df["Close"].rolling(long_ma).mean()

    df["Signal"] = 0
    df.loc[df["Short_MA"] > df["Long_MA"], "Signal"] = 1

    df["Golden_Cross"] = (df["Signal"] == 1) & (df["Signal"].shift(1) == 0)
    df["Death_Cross"] = (df["Signal"] == 0) & (df["Signal"].shift(1) == 1)

    cash = initial_capital
    shares = 0
    position = 0
    entry_price = 0
    entry_date = None

    equity_curve = []
    trades = []

    for current_date, row in df.iterrows():
        close_price = row["Close"]

        if row["Golden_Cross"] and position == 0:
            shares = cash / close_price
            entry_price = close_price
            entry_date = current_date
            cash = 0
            position = 1

        elif row["Death_Cross"] and position == 1:
            exit_price = close_price
            cash = shares * exit_price
            pnl = cash - initial_capital if len(trades) == 0 else cash - trades[-1]["Equity_After"]

            trade_return = (exit_price / entry_price - 1) * 100

            trades.append({
                "Entry_Date": entry_date,
                "Exit_Date": current_date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Return_%": trade_return,
                "PnL": shares * (exit_price - entry_price),
                "Equity_After": cash,
                "Holding_Days": (current_date - entry_date).days
            })

            shares = 0
            position = 0
            entry_price = 0
            entry_date = None

        current_equity = cash if position == 0 else shares * close_price
        equity_curve.append(current_equity)

    df["Equity"] = equity_curve
    df["Buy_Hold_Equity"] = initial_capital * (df["Close"] / df["Close"].iloc[0])

    trades_df = pd.DataFrame(trades)

    return df, trades_df


def calculate_metrics(df, trades_df, initial_capital):
    final_equity = df["Equity"].iloc[-1]
    total_return = (final_equity / initial_capital - 1) * 100
    total_profit = final_equity - initial_capital

    if trades_df.empty:
        return {
            "期末資金": final_equity,
            "總報酬率": total_return,
            "總損益金額": total_profit,
            "交易次數": 0,
            "勝率": 0,
            "獲利因子": 0,
            "平均每筆報酬": 0,
            "最大回撤": 0,
            "買進持有報酬率": (df["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1) * 100
        }

    wins = trades_df[trades_df["PnL"] > 0]
    losses = trades_df[trades_df["PnL"] < 0]

    gross_profit = wins["PnL"].sum()
    gross_loss = abs(losses["PnL"].sum())

    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    win_rate = len(wins) / len(trades_df) * 100

    equity = df["Equity"]
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1) * 100
    max_drawdown = drawdown.min()

    daily_return = equity.pct_change().dropna()
    sharpe = (daily_return.mean() / daily_return.std()) * np.sqrt(252) if daily_return.std() != 0 else 0

    buy_hold_return = (df["Buy_Hold_Equity"].iloc[-1] / initial_capital - 1) * 100

    return {
        "期末資金": final_equity,
        "總報酬率": total_return,
        "總損益金額": total_profit,
        "交易次數": len(trades_df),
        "勝率": win_rate,
        "獲利因子": profit_factor,
        "平均每筆報酬": trades_df["Return_%"].mean(),
        "平均獲利": wins["Return_%"].mean() if len(wins) > 0 else 0,
        "平均虧損": losses["Return_%"].mean() if len(losses) > 0 else 0,
        "最大回撤": max_drawdown,
        "Sharpe Ratio": sharpe,
        "買進持有報酬率": buy_hold_return
    }


def filter_display_range(df, trades_df, option, custom_start, custom_end):
    today = df.index.max().date()

    if option == "近一月":
        start = today - timedelta(days=30)
        end = today
    elif option == "近三月":
        start = today - timedelta(days=90)
        end = today
    elif option == "近半年":
        start = today - timedelta(days=182)
        end = today
    elif option == "近一年":
        start = today - timedelta(days=365)
        end = today
    elif option == "近三年":
        start = today - timedelta(days=365 * 3)
        end = today
    elif option == "近五年":
        start = today - timedelta(days=365 * 5)
        end = today
    else:
        start = custom_start
        end = custom_end

    display_df = df[(df.index.date >= start) & (df.index.date <= end)]

    if not trades_df.empty:
        display_trades = trades_df[
            (trades_df["Entry_Date"].dt.date >= start) &
            (trades_df["Exit_Date"].dt.date <= end)
        ]
    else:
        display_trades = trades_df

    return display_df, display_trades


# =========================
# Streamlit 介面
# =========================

st.set_page_config(page_title="均線金叉死叉回測系統", layout="wide")

st.title("均線金叉 / 死叉回測系統")
st.write("資料來源：Yahoo Finance，日K。策略：短均線金叉長均線進場，短均線死叉長均線出場。")

with st.sidebar:
    st.header("回測設定")

    symbol = st.text_input("YFinance 標的代號", value="^N225")
    st.caption("例：^N225 日經225、^GSPC S&P500、2330.TW 台積電、0050.TW 元大台灣50")

    short_ma = st.number_input("短均線天數", min_value=1, max_value=300, value=5, step=1)
    long_ma = st.number_input("長均線天數", min_value=2, max_value=500, value=20, step=1)

    initial_capital = st.number_input(
        "投入資金",
        min_value=1000,
        value=1_000_000,
        step=10000
    )

    st.subheader("資料下載區間")
    backtest_start = st.date_input("回測開始日期", value=date.today() - timedelta(days=365 * 10))
    backtest_end = st.date_input("回測結束日期", value=date.today())

    st.subheader("圖表顯示區間")
    display_option = st.selectbox(
        "快速選項",
        ["近一月", "近三月", "近半年", "近一年", "近三年", "近五年", "自訂義日期"],
        index=4
    )

    custom_display_start = st.date_input("自訂顯示開始日期", value=date.today() - timedelta(days=365))
    custom_display_end = st.date_input("自訂顯示結束日期", value=date.today())

    run_backtest = st.button("開始回測")


if run_backtest:
    if short_ma >= long_ma:
        st.error("短均線必須小於長均線。")
        st.stop()

    df = download_data(symbol, backtest_start, backtest_end)

    if df.empty:
        st.error("抓不到資料，請確認 YFinance 代號是否正確。")
        st.stop()

    result_df, trades_df = backtest_ma_cross(
        df,
        short_ma=short_ma,
        long_ma=long_ma,
        initial_capital=initial_capital
    )

    metrics = calculate_metrics(result_df, trades_df, initial_capital)

    display_df, display_trades = filter_display_range(
        result_df,
        trades_df,
        display_option,
        custom_display_start,
        custom_display_end
    )

    st.subheader("回測績效總覽")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("期末資金", f"{metrics['期末資金']:,.0f}")
    col2.metric("總損益金額", f"{metrics['總損益金額']:,.0f}")
    col3.metric("總報酬率", f"{metrics['總報酬率']:.2f}%")
    col4.metric("買進持有報酬率", f"{metrics['買進持有報酬率']:.2f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("交易次數", f"{metrics['交易次數']}")
    col6.metric("勝率", f"{metrics['勝率']:.2f}%")
    col7.metric("獲利因子", "∞" if metrics["獲利因子"] == np.inf else f"{metrics['獲利因子']:.2f}")
    col8.metric("最大回撤", f"{metrics['最大回撤']:.2f}%")

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("平均每筆報酬", f"{metrics['平均每筆報酬']:.2f}%")
    col10.metric("平均獲利", f"{metrics.get('平均獲利', 0):.2f}%")
    col11.metric("平均虧損", f"{metrics.get('平均虧損', 0):.2f}%")
    col12.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")

    st.subheader("價格、均線與買賣訊號")

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(display_df.index, display_df["Close"], label="Close")
    ax.plot(display_df.index, display_df["Short_MA"], label=f"MA{short_ma}")
    ax.plot(display_df.index, display_df["Long_MA"], label=f"MA{long_ma}")

    buy_signals = display_df[display_df["Golden_Cross"]]
    sell_signals = display_df[display_df["Death_Cross"]]

    ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", s=80, label="Buy")
    ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", s=80, label="Sell")

    ax.set_title(f"{symbol} MA{short_ma} / MA{long_ma} Strategy")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("資金曲線")

    fig2, ax2 = plt.subplots(figsize=(16, 6))

    ax2.plot(display_df.index, display_df["Equity"], label="Strategy Equity")
    ax2.plot(display_df.index, display_df["Buy_Hold_Equity"], label="Buy & Hold Equity")

    ax2.set_title("Equity Curve")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Equity")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

    st.subheader("所有交易紀錄")

    if trades_df.empty:
        st.warning("此區間沒有完成交易。")
    else:
        trades_show = trades_df.copy()
        trades_show["Entry_Date"] = trades_show["Entry_Date"].dt.strftime("%Y-%m-%d")
        trades_show["Exit_Date"] = trades_show["Exit_Date"].dt.strftime("%Y-%m-%d")
        trades_show["Entry_Price"] = trades_show["Entry_Price"].round(2)
        trades_show["Exit_Price"] = trades_show["Exit_Price"].round(2)
        trades_show["Return_%"] = trades_show["Return_%"].round(2)
        trades_show["PnL"] = trades_show["PnL"].round(0)
        trades_show["Equity_After"] = trades_show["Equity_After"].round(0)

        st.dataframe(trades_show, use_container_width=True)

        csv = trades_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="下載交易紀錄 CSV",
            data=csv,
            file_name=f"{symbol}_ma_cross_trades.csv",
            mime="text/csv"
        )