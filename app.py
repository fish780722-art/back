from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import isinf
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


BASE_DIR = Path(__file__).resolve().parent
YFINANCE_CACHE_DIR = BASE_DIR / ".cache" / "yfinance"
YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(YFINANCE_CACHE_DIR))


@dataclass
class BacktestResult:
    data: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int | str]


QUICK_RANGES: dict[str, int | None] = {
    "近一月": 31,
    "近三月": 93,
    "近半年": 186,
    "近一年": 365,
    "近三年": 365 * 3,
    "近五年": 365 * 5,
    "自訂日期": None,
}

DISPLAY_RANGES: dict[str, int | None] = {
    "全部區間": None,
    "近一月": 31,
    "近三月": 93,
    "近半年": 186,
    "近一年": 365,
    "近三年": 365 * 3,
    "近五年": 365 * 5,
    "自訂日期": None,
}


METRIC_FORMATS = {
    "期末資產": "currency",
    "投入資金": "currency",
    "報酬金額": "currency",
    "總報酬率": "percent",
    "年化報酬率": "percent",
    "買入持有報酬率": "percent",
    "最大回撤": "percent",
    "交易次數": "integer",
    "勝率": "percent",
    "獲利因子": "ratio",
    "平均單筆報酬率": "percent",
    "最佳單筆報酬率": "percent",
    "最差單筆報酬率": "percent",
    "平均持有天數": "decimal",
    "回測天數": "integer",
    "獲利交易數": "integer",
    "虧損交易數": "integer",
    "總獲利": "currency",
    "總虧損": "currency",
    "平均獲利金額": "currency",
    "平均虧損金額": "currency",
    "最大單筆獲利": "currency",
    "最大單筆虧損": "currency",
    "曝險比例": "percent",
}


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_price_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return normalize_downloaded_data(raw)


def normalize_downloaded_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    columns = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in data.columns]
    data = data[columns].dropna(subset=["Close"])
    data.index = pd.to_datetime(data.index)
    return data.sort_index()


def calculate_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def run_backtest(
    price_data: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float,
) -> BacktestResult:
    data = price_data.copy()
    data["ShortMA"] = data["Close"].rolling(short_window).mean()
    data["LongMA"] = data["Close"].rolling(long_window).mean()
    data["Signal"] = (data["ShortMA"] > data["LongMA"]).astype(int)
    data["PositionChange"] = data["Signal"].diff().fillna(0).astype(int)

    cash = float(initial_capital)
    shares = 0.0
    entry_date: pd.Timestamp | None = None
    entry_price = 0.0
    entry_value = 0.0
    equity_values: list[float] = []
    position_values: list[float] = []
    trades: list[dict[str, float | int | str | pd.Timestamp]] = []

    for current_date, row in data.iterrows():
        close_price = float(row["Close"])
        position_change = int(row["PositionChange"])

        if position_change == 1 and shares == 0 and cash > 0:
            shares = cash / close_price
            entry_date = current_date
            entry_price = close_price
            entry_value = cash
            cash = 0.0

        elif position_change == -1 and shares > 0:
            cash, trade = close_position(
                shares=shares,
                exit_price=close_price,
                exit_date=current_date,
                entry_date=entry_date,
                entry_price=entry_price,
                entry_value=entry_value,
                exit_reason="死叉出場",
            )
            trades.append(trade)
            shares = 0.0
            entry_date = None
            entry_price = 0.0
            entry_value = 0.0

        position_value = shares * close_price
        equity_values.append(cash + position_value)
        position_values.append(position_value)

    if shares > 0 and not data.empty:
        last_date = data.index[-1]
        last_price = float(data.iloc[-1]["Close"])
        final_value, trade = close_position(
            shares=shares,
            exit_price=last_price,
            exit_date=last_date,
            entry_date=entry_date,
            entry_price=entry_price,
            entry_value=entry_value,
            exit_reason="區間結束結算",
        )
        trades.append(trade)
        equity_values[-1] = final_value
        position_values[-1] = 0.0

    data["Equity"] = equity_values
    data["PositionValue"] = position_values

    trades_df = pd.DataFrame(trades)
    metrics = calculate_metrics(
        data=data,
        trades=trades_df,
        initial_capital=initial_capital,
    )
    return BacktestResult(data=data, trades=trades_df, metrics=metrics)


def close_position(
    shares: float,
    exit_price: float,
    exit_date: pd.Timestamp,
    entry_date: pd.Timestamp | None,
    entry_price: float,
    entry_value: float,
    exit_reason: str,
) -> tuple[float, dict[str, float | int | str | pd.Timestamp]]:
    exit_value = shares * exit_price
    profit = exit_value - entry_value
    return_pct = profit / entry_value if entry_value else 0.0
    holding_days = (exit_date - entry_date).days if entry_date is not None else 0

    return exit_value, {
        "進場日期": entry_date,
        "出場日期": exit_date,
        "進場價": entry_price,
        "出場價": exit_price,
        "股數": shares,
        "進場金額": entry_value,
        "出場金額": exit_value,
        "損益金額": profit,
        "損益率": return_pct,
        "持有天數": holding_days,
        "出場原因": exit_reason,
    }


def calculate_metrics(
    data: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
) -> dict[str, float | int | str]:
    final_equity = float(data["Equity"].iloc[-1]) if not data.empty else float(initial_capital)
    total_profit = final_equity - initial_capital
    total_return = total_profit / initial_capital if initial_capital else 0.0
    backtest_days = max((data.index[-1] - data.index[0]).days, 1) if len(data) > 1 else 1
    annualized_return = (final_equity / initial_capital) ** (365 / backtest_days) - 1 if initial_capital else 0.0
    buy_hold_return = float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1) if len(data) > 1 else 0.0
    exposure_ratio = float((data["PositionValue"] > 0).mean()) if not data.empty else 0.0

    if trades.empty:
        return {
            "期末資產": final_equity,
            "投入資金": initial_capital,
            "報酬金額": total_profit,
            "總報酬率": total_return,
            "年化報酬率": annualized_return,
            "買入持有報酬率": buy_hold_return,
            "最大回撤": calculate_max_drawdown(data["Equity"]),
            "交易次數": 0,
            "勝率": 0.0,
            "獲利因子": 0.0,
            "平均單筆報酬率": 0.0,
            "最佳單筆報酬率": 0.0,
            "最差單筆報酬率": 0.0,
            "平均持有天數": 0.0,
            "回測天數": backtest_days,
            "獲利交易數": 0,
            "虧損交易數": 0,
            "總獲利": 0.0,
            "總虧損": 0.0,
            "平均獲利金額": 0.0,
            "平均虧損金額": 0.0,
            "最大單筆獲利": 0.0,
            "最大單筆虧損": 0.0,
            "曝險比例": exposure_ratio,
        }

    profits = trades["損益金額"]
    returns = trades["損益率"]
    winners = trades[profits > 0]
    losers = trades[profits < 0]
    gross_profit = float(winners["損益金額"].sum())
    gross_loss = abs(float(losers["損益金額"].sum()))
    profit_factor = np.inf if gross_loss == 0 and gross_profit > 0 else gross_profit / gross_loss if gross_loss else 0.0

    return {
        "期末資產": final_equity,
        "投入資金": initial_capital,
        "報酬金額": total_profit,
        "總報酬率": total_return,
        "年化報酬率": annualized_return,
        "買入持有報酬率": buy_hold_return,
        "最大回撤": calculate_max_drawdown(data["Equity"]),
        "交易次數": int(len(trades)),
        "勝率": float(len(winners) / len(trades)),
        "獲利因子": float(profit_factor),
        "平均單筆報酬率": float(returns.mean()),
        "最佳單筆報酬率": float(returns.max()),
        "最差單筆報酬率": float(returns.min()),
        "平均持有天數": float(trades["持有天數"].mean()),
        "回測天數": backtest_days,
        "獲利交易數": int(len(winners)),
        "虧損交易數": int(len(losers)),
        "總獲利": gross_profit,
        "總虧損": -gross_loss,
        "平均獲利金額": float(winners["損益金額"].mean()) if not winners.empty else 0.0,
        "平均虧損金額": float(losers["損益金額"].mean()) if not losers.empty else 0.0,
        "最大單筆獲利": float(profits.max()),
        "最大單筆虧損": float(profits.min()),
        "曝險比例": exposure_ratio,
    }


def format_metric_value(key: str, value: float | int | str) -> str:
    metric_format = METRIC_FORMATS.get(key, "text")
    if metric_format == "currency":
        return f"{float(value):,.0f}"
    if metric_format == "percent":
        return f"{float(value):.2%}"
    if metric_format == "ratio":
        numeric_value = float(value)
        return "無限大" if isinf(numeric_value) else f"{numeric_value:.2f}"
    if metric_format == "integer":
        return f"{int(value):,}"
    if metric_format == "decimal":
        return f"{float(value):,.1f}"
    return str(value)


def build_price_chart(
    data: pd.DataFrame,
    symbol: str,
    short_window: int,
    long_window: int,
    marker_size: int,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="日 K",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["ShortMA"], name=f"短均線 {short_window}", line=dict(width=1.8)))
    fig.add_trace(go.Scatter(x=data.index, y=data["LongMA"], name=f"長均線 {long_window}", line=dict(width=1.8)))

    buy_points = data[data["PositionChange"] == 1]
    sell_points = data[data["PositionChange"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=marker_size, color="#15803d"),
            hovertemplate="進場<br>日期：%{x|%Y-%m-%d}<br>價格：%{y:,.2f}<extra></extra>",
            name="金叉進場",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=marker_size, color="#b91c1c"),
            hovertemplate="出場<br>日期：%{x|%Y-%m-%d}<br>價格：%{y:,.2f}<extra></extra>",
            name="死叉出場",
        )
    )
    fig.update_layout(
        title=f"{symbol} 日 K、均線與買賣點",
        height=560,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(range=calculate_price_y_range(data), fixedrange=False)
    return fig


def build_equity_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Equity"], name="策略資產曲線", line=dict(width=2)))
    fig.update_layout(
        title="策略資產曲線",
        height=360,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_tickformat=",",
    )
    fig.update_yaxes(range=calculate_series_y_range(data["Equity"]), fixedrange=False)
    return fig


def build_drawdown_chart(data: pd.DataFrame) -> go.Figure:
    running_max = data["Equity"].cummax()
    drawdown = data["Equity"] / running_max - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=drawdown, name="回撤", fill="tozeroy", line=dict(width=1.5)))
    fig.update_layout(
        title="回撤曲線",
        height=320,
        margin=dict(l=20, r=20, t=55, b=20),
        yaxis_tickformat=".0%",
    )
    fig.update_yaxes(range=calculate_series_y_range(drawdown), fixedrange=False)
    return fig


def calculate_price_y_range(data: pd.DataFrame) -> list[float]:
    price_values = pd.concat(
        [
            data["Low"],
            data["High"],
            data["ShortMA"],
            data["LongMA"],
        ]
    ).dropna()
    return calculate_series_y_range(price_values)


def calculate_series_y_range(series: pd.Series) -> list[float]:
    clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_series.empty:
        return [0.0, 1.0]

    min_value = float(clean_series.min())
    max_value = float(clean_series.max())
    if min_value == max_value:
        padding = abs(min_value) * 0.05 or 1.0
    else:
        padding = (max_value - min_value) * 0.08
    return [min_value - padding, max_value + padding]


def filter_display_data(data: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)
    return data.loc[(data.index >= start_timestamp) & (data.index <= end_timestamp)].copy()


def get_display_date_range(data: pd.DataFrame, range_label: str) -> tuple[date, date]:
    data_start = data.index.min().date()
    data_end = data.index.max().date()

    if range_label == "全部區間":
        return data_start, data_end

    days = DISPLAY_RANGES[range_label]
    if days is None:
        return data_start, data_end

    return max(data_start, data_end - timedelta(days=days)), data_end


def get_default_date_range(range_label: str) -> tuple[date, date]:
    today = date.today()
    days = QUICK_RANGES[range_label]
    if days is None:
        return today - timedelta(days=365), today
    return today - timedelta(days=days), today


def build_metrics_table(metrics: dict[str, float | int | str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"分析數據": key, "數值": format_metric_value(key, value)} for key, value in metrics.items()]
    )


def build_display_trades(trades: pd.DataFrame) -> pd.DataFrame:
    display_trades = trades.copy()
    for column in ["進場日期", "出場日期"]:
        display_trades[column] = pd.to_datetime(display_trades[column]).dt.strftime("%Y-%m-%d")
    for column in ["進場價", "出場價", "股數", "進場金額", "出場金額", "損益金額"]:
        display_trades[column] = display_trades[column].map(lambda value: f"{value:,.2f}")
    display_trades["損益率"] = display_trades["損益率"].map(lambda value: f"{value:.2%}")
    return display_trades


def main() -> None:
    st.set_page_config(page_title="金叉死叉回測 APP", layout="wide")
    st.title("金叉進場、死叉出場回測 APP")

    with st.sidebar:
        st.header("回測設定")
        symbol = st.text_input("yfinance 標的代號", value="2330.TW").strip().upper()
        initial_capital = st.number_input("投入資金", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
        short_window = st.number_input("短均線天數", min_value=2, value=5, step=1)
        long_window = st.number_input("長均線天數", min_value=3, value=20, step=1)
        marker_size = st.number_input("進出場標記大小", min_value=4, max_value=30, value=11, step=1)
        range_label = st.selectbox("交易區間", options=list(QUICK_RANGES.keys()), index=3)

        default_start, default_end = get_default_date_range(range_label)
        if range_label == "自訂日期":
            start_date = st.date_input("開始日期", value=default_start)
            end_date = st.date_input("結束日期", value=default_end)
        else:
            start_date = default_start
            end_date = default_end
            st.caption(f"目前區間：{start_date} 到 {end_date}")

        run_button = st.button("執行回測", type="primary", use_container_width=True)

    if short_window >= long_window:
        st.warning("短均線天數必須小於長均線天數。")
        return

    if start_date >= end_date:
        st.warning("開始日期必須早於結束日期。")
        return

    if not symbol:
        st.warning("請輸入 yfinance 標的代號。")
        return

    if run_button:
        with st.spinner("正在下載日 K 資料並計算回測..."):
            price_data = load_price_data(symbol, start_date, end_date)

        if price_data.empty:
            st.error("沒有抓到資料。請確認 yfinance 代號是否正確，或換一個日期區間。")
            return

        if len(price_data) < long_window + 2:
            st.error("資料筆數不足，請拉長日期區間或降低長均線天數。")
            return

        st.session_state["backtest_result"] = run_backtest(
            price_data,
            int(short_window),
            int(long_window),
            float(initial_capital),
        )
        st.session_state["backtest_symbol"] = symbol
        st.session_state["backtest_short_window"] = int(short_window)
        st.session_state["backtest_long_window"] = int(long_window)
        st.session_state["backtest_start_date"] = start_date
        st.session_state["backtest_end_date"] = end_date

    if "backtest_result" not in st.session_state:
        st.info("設定參數後按下「執行回測」。預設範例標的是台積電 2330.TW。")
        return

    result = st.session_state["backtest_result"]
    result_symbol = st.session_state["backtest_symbol"]
    result_short_window = st.session_state["backtest_short_window"]
    result_long_window = st.session_state["backtest_long_window"]

    st.subheader("核心結果")
    primary_metrics = ["期末資產", "報酬金額", "總報酬率", "勝率", "交易次數", "獲利因子", "最大回撤", "買入持有報酬率"]
    metric_columns = st.columns(4)
    for index, key in enumerate(primary_metrics):
        with metric_columns[index % 4]:
            st.metric(key, format_metric_value(key, result.metrics[key]))

    st.subheader("圖表顯示區間")
    chart_range_label = st.selectbox(
        "選擇圖表要看的日期範圍",
        options=list(DISPLAY_RANGES.keys()),
        index=0,
        key="chart_range_label",
    )
    default_display_start, default_display_end = get_display_date_range(result.data, chart_range_label)

    if chart_range_label == "自訂日期":
        date_columns = st.columns(2)
        with date_columns[0]:
            display_start_date = st.date_input(
                "圖表開始日期",
                value=default_display_start,
                min_value=result.data.index.min().date(),
                max_value=result.data.index.max().date(),
                key="display_start_date",
            )
        with date_columns[1]:
            display_end_date = st.date_input(
                "圖表結束日期",
                value=default_display_end,
                min_value=result.data.index.min().date(),
                max_value=result.data.index.max().date(),
                key="display_end_date",
            )
    else:
        display_start_date = default_display_start
        display_end_date = default_display_end
        st.caption(f"目前圖表顯示：{display_start_date} 到 {display_end_date}")

    if display_start_date >= display_end_date:
        st.warning("圖表開始日期必須早於圖表結束日期。")
        return

    display_data = filter_display_data(result.data, display_start_date, display_end_date)
    if display_data.empty:
        st.warning("這個圖表區間沒有資料。")
        return

    st.plotly_chart(
        build_price_chart(display_data, result_symbol, result_short_window, result_long_window, int(marker_size)),
        use_container_width=True,
    )
    st.plotly_chart(build_equity_chart(display_data), use_container_width=True)
    st.plotly_chart(build_drawdown_chart(display_data), use_container_width=True)

    st.subheader("所有分析數據")
    st.dataframe(build_metrics_table(result.metrics), use_container_width=True, hide_index=True)

    st.subheader("交易明細")
    if result.trades.empty:
        st.info("這段期間沒有產生完整交易。")
    else:
        display_trades = build_display_trades(result.trades)
        st.dataframe(display_trades, use_container_width=True, hide_index=True)
        st.download_button(
            "下載交易明細 CSV",
            data=result.trades.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{result_symbol}_ma_cross_trades.csv",
            mime="text/csv",
        )

    with st.expander("原始日 K 與策略資料"):
        st.dataframe(result.data, use_container_width=True)


if __name__ == "__main__":
    main()
