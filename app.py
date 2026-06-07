from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from html import escape, unescape
from math import isinf
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


class PriceDataUnavailableError(Exception):
    pass


PRICE_ADJUSTMENT_MODES = [
    "使用調整價回測（建議）",
    "使用原始價格回測",
]
SPLIT_GAP_THRESHOLD = 4.0


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
HIDDEN_METRICS = {
    "買入持有報酬率",
}


@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_price_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    data = download_price_data(symbol, start_date, end_date)
    if data.empty:
        data = download_price_data(symbol, start_date - timedelta(days=7), end_date)
        data = filter_display_data(data, start_date, end_date) if not data.empty else data

    if data.empty:
        raise PriceDataUnavailableError("No price data returned from yfinance.")

    return data


def download_price_data(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=start_date,
        end=end_date + timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return normalize_downloaded_data(raw)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_symbol_profile(symbol: str) -> dict[str, str]:
    try:
        info = yf.Ticker(symbol).get_info()
    except Exception:
        return {}

    return {
        "short_name": str(info.get("shortName") or "").strip(),
        "long_name": str(info.get("longName") or "").strip(),
    }


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_yahoo_search_name(symbol: str, lang: str, region: str) -> str:
    query = urlencode(
        {
            "q": symbol,
            "quotes_count": 8,
            "news_count": 0,
            "lang": lang,
            "region": region,
        }
    )
    request = Request(
        f"https://query1.finance.yahoo.com/v1/finance/search?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""

    quotes = payload.get("quotes", [])
    if not isinstance(quotes, list):
        return ""

    normalized_symbol = symbol.upper()
    for quote in quotes:
        quote_symbol = str(quote.get("symbol") or "").upper()
        if quote_symbol == normalized_symbol:
            return str(quote.get("longname") or quote.get("shortname") or "").strip()

    for quote in quotes:
        name = str(quote.get("longname") or quote.get("shortname") or "").strip()
        if name:
            return name

    return ""


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_symbol_search_results(search_text: str) -> list[dict[str, str]]:
    query_text = search_text.strip()
    if not query_text:
        return []

    results: dict[str, dict[str, str]] = {}
    for item in load_twse_search_results(query_text):
        results[item["symbol"]] = item

    for lang, region in [("zh-TW", "TW"), ("en-US", "US")]:
        query = urlencode(
            {
                "q": query_text,
                "quotes_count": 12,
                "news_count": 0,
                "lang": lang,
                "region": region,
            }
        )
        request = Request(
            f"https://query1.finance.yahoo.com/v1/finance/search?{query}",
            headers={"User-Agent": "Mozilla/5.0"},
        )

        try:
            with urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue

        quotes = payload.get("quotes", [])
        if not isinstance(quotes, list):
            continue

        for quote in quotes:
            symbol = str(quote.get("symbol") or "").strip().upper()
            if not symbol or symbol in results:
                continue

            name = str(quote.get("longname") or quote.get("shortname") or "").strip()
            exchange = str(quote.get("exchange") or quote.get("exchDisp") or "").strip()
            quote_type = str(quote.get("quoteType") or "").strip()
            results[symbol] = {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "quote_type": quote_type,
            }

    normalized_query = query_text.upper()
    symbol_prefix_query = bool(re.fullmatch(r"[A-Z0-9.^-]+", normalized_query))
    return sorted(
        results.values(),
        key=lambda item: (
            0 if symbol_prefix_query and item["symbol"].startswith(normalized_query) else 1,
            item["symbol"],
        ),
    )


@st.cache_data(ttl=60 * 15, show_spinner=False)
def load_twse_search_results(search_text: str) -> list[dict[str, str]]:
    query_text = search_text.strip()
    if not query_text:
        return []

    query = urlencode({"query": query_text})
    request = Request(
        f"https://www.twse.com.tw/rwd/zh/api/codeQuery?{query}",
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    suggestions = payload.get("suggestions", [])
    if not isinstance(suggestions, list):
        return []

    results: list[dict[str, str]] = []
    for suggestion in suggestions:
        fields = str(suggestion).split("\t", 1)
        if len(fields) != 2:
            continue

        code = fields[0].strip().upper()
        name = fields[1].strip()
        is_common_stock_or_etf = len(code) in {4, 5} or (len(code) == 6 and code.startswith("00"))
        if not code.isdigit() or not is_common_stock_or_etf:
            continue

        results.append(
            {
                "symbol": f"{code}.TW",
                "name": name,
                "exchange": "TWSE",
                "quote_type": "EQUITY",
            }
        )

        if len(results) >= 12:
            break

    return results


def format_symbol_option(item: dict[str, str]) -> str:
    parts = [item["symbol"]]
    if item.get("name"):
        parts.append(item["name"])
    if item.get("exchange"):
        parts.append(item["exchange"])
    return " - ".join(parts)


def is_symbol_like(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9.^=-]+(?:\.[A-Za-z0-9]+)?", value.strip()))


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_market_page_name(symbol: str) -> str:
    normalized_symbol = symbol.upper()
    if normalized_symbol.endswith((".TW", ".TWO")):
        url = f"https://tw.stock.yahoo.com/quote/{normalized_symbol}"
        title_pattern = r"<title>(.*?)</title>"
        name_pattern = r"^\s*([^(\s]+)"
    elif normalized_symbol.endswith(".T"):
        url = f"https://finance.yahoo.co.jp/quote/{normalized_symbol}"
        title_pattern = r"<title>(.*?)</title>"
        name_pattern = r"^\s*(.*?)【"
    else:
        return ""

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=8) as response:
            html = response.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

    title_match = re.search(title_pattern, html, re.IGNORECASE | re.DOTALL)
    if not title_match:
        return ""

    title = unescape(title_match.group(1)).strip()
    name_match = re.search(name_pattern, title)
    if not name_match:
        return ""

    return name_match.group(1).strip()


def resolve_symbol_names(symbol: str) -> tuple[str, str]:
    profile = load_symbol_profile(symbol)
    chinese_name = load_market_page_name(symbol) or load_yahoo_search_name(symbol, "zh-TW", "TW")
    english_name = load_yahoo_search_name(symbol, "en-US", "US")
    resolved_chinese_name = chinese_name or profile.get("short_name") or symbol
    resolved_english_name = english_name or profile.get("long_name") or profile.get("short_name") or ""
    return resolved_chinese_name, resolved_english_name


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


def prepare_backtest_prices(data: pd.DataFrame, use_adjusted_price: bool) -> pd.DataFrame:
    prepared = data.copy()
    if not use_adjusted_price or "Adj Close" not in prepared.columns:
        return prepared

    close = prepared["Close"].replace(0, np.nan)
    adjustment_factor = (prepared["Adj Close"] / close).replace([np.inf, -np.inf], np.nan)
    adjustment_factor = adjustment_factor.ffill().bfill().fillna(1.0)

    for column in ["Open", "High", "Low", "Close", "Adj Close"]:
        if column in prepared.columns:
            raw_column = f"Raw{column.replace(' ', '')}"
            prepared[raw_column] = prepared[column]

    for column in ["Open", "High", "Low"]:
        if column in prepared.columns:
            prepared[column] = prepared[column] * adjustment_factor

    prepared["Close"] = prepared["Adj Close"]
    prepared["AdjustmentFactor"] = adjustment_factor
    prepared = repair_split_price_gaps(prepared)
    return prepared


def repair_split_price_gaps(data: pd.DataFrame) -> pd.DataFrame:
    repaired = data.copy()
    close = repaired["Close"].replace(0, np.nan)
    price_ratio = close / close.shift(1)
    valid_ratio = price_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    split_points = valid_ratio[(valid_ratio >= SPLIT_GAP_THRESHOLD) | (valid_ratio <= 1 / SPLIT_GAP_THRESHOLD)]

    repaired["SplitRepairFactor"] = 1.0
    if split_points.empty:
        return repaired

    cumulative_factor = 1.0
    repair_factor = pd.Series(1.0, index=repaired.index)
    for split_date, ratio in split_points.sort_index(ascending=False).items():
        cumulative_factor *= float(ratio)
        repair_factor.loc[repair_factor.index < split_date] = cumulative_factor

    for column in ["Open", "High", "Low", "Close"]:
        if column in repaired.columns:
            repaired[column] = repaired[column] * repair_factor

    repaired["SplitRepairFactor"] = repair_factor
    return repaired


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
    take_profit_pct: float = 0.0,
    stop_loss_pct: float = 0.0,
) -> BacktestResult:
    data = price_data.copy()
    data["ShortMA"] = data["Close"].rolling(short_window).mean()
    data["LongMA"] = data["Close"].rolling(long_window).mean()
    data["Signal"] = (data["ShortMA"] > data["LongMA"]).astype(int)
    data["PositionChange"] = data["Signal"].diff().fillna(0).astype(int)
    data["TradeAction"] = data["PositionChange"].shift(1).fillna(0).astype(int)
    data["TradeMarker"] = 0

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
        trade_action = int(row["TradeAction"])

        exit_reason = ""
        if shares > 0 and close_price > 0 and entry_price > 0:
            current_return = close_price / entry_price - 1
            if take_profit_pct > 0 and current_return >= take_profit_pct:
                exit_reason = "停利出場"
            elif stop_loss_pct > 0 and current_return <= -stop_loss_pct:
                exit_reason = "停損出場"
            elif trade_action == -1:
                exit_reason = "死叉出場"

        if exit_reason:
            cash, trade = close_position(
                shares=shares,
                exit_price=close_price,
                exit_date=current_date,
                entry_date=entry_date,
                entry_price=entry_price,
                entry_value=entry_value,
                exit_reason=exit_reason,
            )
            trades.append(trade)
            data.at[current_date, "TradeMarker"] = -1
            shares = 0.0
            entry_date = None
            entry_price = 0.0
            entry_value = 0.0

        if trade_action == 1 and shares == 0 and cash > 0 and close_price > 0:
            shares = cash / close_price
            entry_date = current_date
            entry_price = close_price
            entry_value = cash
            cash = 0.0
            data.at[current_date, "TradeMarker"] = 1

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
        data.at[last_date, "TradeMarker"] = -1

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
    up_color: str,
    down_color: str,
    short_ma_color: str,
    long_ma_color: str,
    buy_marker_color: str,
    sell_marker_color: str,
    legend_text_color: str,
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
            increasing_line_color=up_color,
            decreasing_line_color=down_color,
            increasing_fillcolor=up_color,
            decreasing_fillcolor=down_color,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["ShortMA"],
            name=f"短均線 {short_window}",
            line=dict(width=1.8, color=short_ma_color),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["LongMA"],
            name=f"長均線 {long_window}",
            line=dict(width=1.8, color=long_ma_color),
        )
    )

    buy_points = data[data["TradeMarker"] == 1]
    sell_points = data[data["TradeMarker"] == -1]
    fig.add_trace(
        go.Scatter(
            x=buy_points.index,
            y=buy_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=marker_size, color=buy_marker_color),
            hovertemplate="進場<br>日期：%{x|%Y-%m-%d}<br>收盤價：%{y:,.2f}<extra></extra>",
            name="金叉進場",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points.index,
            y=sell_points["Close"],
            mode="markers",
            marker=dict(symbol="triangle-down", size=marker_size, color=sell_marker_color),
            hovertemplate="出場<br>日期：%{x|%Y-%m-%d}<br>收盤價：%{y:,.2f}<extra></extra>",
            name="死叉出場",
        )
    )
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=80, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=legend_text_color),
        ),
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
        [
            {"分析數據": key, "數值": format_metric_value(key, value)}
            for key, value in metrics.items()
            if key not in HIDDEN_METRICS
        ]
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
        symbol_query = st.text_input("搜尋標的", value="2330.TW").strip()
        symbol_options = load_symbol_search_results(symbol_query)
        if symbol_options:
            option_labels = [format_symbol_option(option) for option in symbol_options]
            selected_option_label = st.selectbox("選擇標的", options=option_labels)
            selected_option = symbol_options[option_labels.index(selected_option_label)]
            symbol = selected_option["symbol"]
        elif is_symbol_like(symbol_query):
            symbol = symbol_query.upper()
            st.caption("找不到候選清單，將直接使用輸入的代號。")
        else:
            symbol = ""
            st.caption("請輸入更多關鍵字，或改輸入完整 yfinance 代號。")
        initial_capital = st.number_input("投入資金", min_value=1_000.0, value=1_000_000.0, step=10_000.0)
        short_window = st.number_input("短均線天數", min_value=2, value=5, step=1)
        long_window = st.number_input("長均線天數", min_value=3, value=20, step=1)
        price_adjustment_mode = st.selectbox("價格調整方式", options=PRICE_ADJUSTMENT_MODES, index=0)
        marker_size = st.number_input("進出場標記大小", min_value=4, max_value=30, value=11, step=1)
        use_take_profit = st.checkbox("啟用停利出場", value=False)
        take_profit_pct = (
            st.number_input("停利百分比 (%)", min_value=0.1, value=10.0, step=0.5) / 100
            if use_take_profit
            else 0.0
        )
        use_stop_loss = st.checkbox("啟用停損出場", value=False)
        stop_loss_pct = (
            st.number_input("停損百分比 (%)", min_value=0.1, value=5.0, step=0.5) / 100
            if use_stop_loss
            else 0.0
        )
        with st.expander("圖表顏色", expanded=False):
            chart_title_color = st.color_picker("圖表標題顏色", value="#111827")
            legend_text_color = st.color_picker("圖例文字顏色", value="#111827")
            up_color = st.color_picker("上漲 K 顏色", value="#16a34a")
            down_color = st.color_picker("下跌 K 顏色", value="#dc2626")
            short_ma_color = st.color_picker("短均線顏色", value="#60a5fa")
            long_ma_color = st.color_picker("長均線顏色", value="#ff3333")
            buy_marker_color = st.color_picker("金叉進場顏色", value="#15803d")
            sell_marker_color = st.color_picker("死叉出場顏色", value="#b91c1c")
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
            try:
                price_data = load_price_data(symbol, start_date, end_date)
            except PriceDataUnavailableError:
                price_data = pd.DataFrame()

        if price_data.empty:
            st.error("沒有抓到資料。請確認 yfinance 代號是否正確，或換一個日期區間。")
            return

        price_data = prepare_backtest_prices(
            price_data,
            price_adjustment_mode == PRICE_ADJUSTMENT_MODES[0],
        )

        if len(price_data) < long_window + 2:
            st.error("資料筆數不足，請拉長日期區間或降低長均線天數。")
            return

        st.session_state["backtest_result"] = run_backtest(
            price_data,
            int(short_window),
            int(long_window),
            float(initial_capital),
            float(take_profit_pct),
            float(stop_loss_pct),
        )
        st.session_state["backtest_symbol"] = symbol
        st.session_state["backtest_short_window"] = int(short_window)
        st.session_state["backtest_long_window"] = int(long_window)
        st.session_state["backtest_start_date"] = start_date
        st.session_state["backtest_end_date"] = end_date
        st.session_state["backtest_price_adjustment_mode"] = price_adjustment_mode
        st.session_state["backtest_take_profit_pct"] = float(take_profit_pct)
        st.session_state["backtest_stop_loss_pct"] = float(stop_loss_pct)
        resolved_chinese_name, resolved_english_name = resolve_symbol_names(symbol)
        st.session_state["backtest_symbol_chinese_name"] = resolved_chinese_name
        st.session_state["backtest_symbol_english_name"] = resolved_english_name

    if "backtest_result" not in st.session_state:
        st.info("設定參數後按下「執行回測」。預設範例標的是台積電 2330.TW。")
        return

    result = st.session_state["backtest_result"]
    result_symbol = st.session_state["backtest_symbol"]
    result_short_window = st.session_state["backtest_short_window"]
    result_long_window = st.session_state["backtest_long_window"]
    result_price_adjustment_mode = st.session_state.get("backtest_price_adjustment_mode", PRICE_ADJUSTMENT_MODES[0])
    result_take_profit_pct = st.session_state.get("backtest_take_profit_pct", 0.0)
    result_stop_loss_pct = st.session_state.get("backtest_stop_loss_pct", 0.0)
    result_symbol_chinese_name = st.session_state.get("backtest_symbol_chinese_name", result_symbol)
    result_symbol_english_name = st.session_state.get("backtest_symbol_english_name", "")
    escaped_symbol_chinese_name = escape(result_symbol_chinese_name)
    escaped_symbol_english_name = escape(result_symbol_english_name)
    symbol_english_html = ""
    if escaped_symbol_english_name:
        symbol_english_html = (
            '<div style="font-size: 1.15rem; font-weight: 500; line-height: 1.35; '
            f'color: #6b7280; margin-top: 6px;">{escaped_symbol_english_name}</div>'
        )

    st.markdown(
        (
            '<div style="margin: 8px 0 28px 0;">'
            '<div style="font-size: 2.75rem; font-weight: 800; line-height: 1.15; color: #111827;">'
            f"{escaped_symbol_chinese_name}</div>"
            f"{symbol_english_html}</div>"
        ),
        unsafe_allow_html=True,
    )

    st.subheader("核心結果")
    st.caption(f"價格調整方式：{result_price_adjustment_mode}")
    take_profit_label = f"停利：{result_take_profit_pct:.2%}" if result_take_profit_pct > 0 else "停利：未啟用"
    stop_loss_label = f"停損：{result_stop_loss_pct:.2%}" if result_stop_loss_pct > 0 else "停損：未啟用"
    st.caption(f"{take_profit_label}；{stop_loss_label}")
    primary_metrics = ["期末資產", "報酬金額", "總報酬率", "勝率", "交易次數", "獲利因子", "最大回撤"]
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

    st.markdown(
        f'<div style="color:{chart_title_color}; font-weight:700; margin: 0 0 12px 0;">'
        f"{result_symbol} 日 K、均線與買賣點</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_price_chart(
            display_data,
            result_symbol,
            result_short_window,
            result_long_window,
            int(marker_size),
            up_color,
            down_color,
            short_ma_color,
            long_ma_color,
            buy_marker_color,
            sell_marker_color,
            legend_text_color,
        ),
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
