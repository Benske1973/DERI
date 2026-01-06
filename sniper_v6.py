#!/usr/bin/env python3
"""
V6: The Final Sniper
MACD histogram volatility breakout + optional RSI confirmation.

Example:
  python sniper_v6.py --symbol PRCL/USDT --timeframe 4h --exchange kucoin --limit 500

Notes:
  - Uses ccxt for OHLCV data.
  - By default shows a 3-panel matplotlib chart (price, MACD hist + threshold, RSI).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Settings:
    symbol: str = "PRCL/USDT"
    timeframe: str = "4h"
    exchange: str = "kucoin"
    limit: int = 500

    # MACD
    fast_len: int = 12
    slow_len: int = 26
    sig_len: int = 9

    # V5 lagged breakout
    lookback_period: int = 20
    sensitivity: float = 3.0

    # V6 RSI filter
    use_rsi_filter: bool = True
    rsi_length: int = 14
    rsi_threshold: float = 50.0

    # Output
    plot: bool = True
    save_plot: str | None = None


def _calc_ema(source: pd.Series, length: int) -> pd.Series:
    return source.ewm(span=length, adjust=False).mean()


def _calc_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    # Handmatige RSI berekening (EMA smoothing) zoals in je snippet
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / length, adjust=False).mean()

    # Vermijd deling door 0: loss==0 => RSI = 100 (pure gains)
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100.0).clip(0, 100)


def _load_ohlcv(exchange_id: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    import ccxt  # local import so script can be imported without ccxt installed

    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise ValueError(f"Unknown ccxt exchange: {exchange_id}")

    exchange = ex_class()
    # Some exchanges require markets loaded for symbol formatting
    try:
        exchange.load_markets()
    except Exception:
        pass

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned.")

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    return df


def compute_indicators(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    out = df.copy()

    # A. MACD
    out["ma_fast"] = _calc_ema(out["close"], s.fast_len)
    out["ma_slow"] = _calc_ema(out["close"], s.slow_len)
    out["macd"] = out["ma_fast"] - out["ma_slow"]
    out["signal"] = _calc_ema(out["macd"], s.sig_len)
    out["hist"] = out["macd"] - out["signal"]

    # B. Lagged volatility baseline (shift(1) is crucial)
    out["abs_hist"] = out["hist"].abs()
    out["baseline_noise"] = out["abs_hist"].rolling(window=s.lookback_period).mean().shift(1)
    out["baseline_noise"] = out["baseline_noise"].replace(0, 1e-8)

    # C. RSI
    out["rsi"] = _calc_rsi(out["close"], length=s.rsi_length)

    return out


def generate_signals(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    out = df.copy()
    buy_signals = [np.nan] * len(out)
    sell_signals = [np.nan] * len(out)

    start_index = s.lookback_period + 2

    for i in range(start_index, len(out)):
        hist_curr = float(out["hist"].iloc[i])
        hist_prev = float(out["hist"].iloc[i - 1])
        baseline = out["baseline_noise"].iloc[i]
        rsi_curr = float(out["rsi"].iloc[i])

        if pd.isna(baseline):
            continue

        # Filters
        is_explosion = hist_curr > (float(baseline) * s.sensitivity)
        pass_rsi = (rsi_curr > s.rsi_threshold) if s.use_rsi_filter else True

        # BUY: green + larger than previous + breakout + RSI OK
        if hist_curr > 0 and hist_curr > hist_prev and is_explosion and pass_rsi:
            buy_signals[i] = float(out["low"].iloc[i]) * 0.95

        # SELL (optional RSI<50 when filter enabled)
        is_dump = hist_curr < -(float(baseline) * s.sensitivity)
        pass_rsi_sell = (rsi_curr < 50.0) if s.use_rsi_filter else True

        if hist_curr < 0 and hist_curr < hist_prev and is_dump and pass_rsi_sell:
            sell_signals[i] = float(out["high"].iloc[i]) * 1.05

    out["buy_signal"] = buy_signals
    out["sell_signal"] = sell_signals
    return out


def plot_chart(df: pd.DataFrame, s: Settings) -> None:
    # Ensure headless environments can still save plots.
    if s.save_plot and not os.environ.get("DISPLAY"):
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))

    # PLOT 1: PRICE
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df["timestamp"], df["close"], label="Prijs", color="#787b86", linewidth=1)

    valid_buy = df.dropna(subset=["buy_signal"])
    valid_sell = df.dropna(subset=["sell_signal"])

    ax1.scatter(
        valid_buy["timestamp"],
        valid_buy["buy_signal"],
        color="#00E676",
        marker="^",
        s=180,
        label="Sniper Buy",
        zorder=5,
        edgecolors="black",
    )
    ax1.scatter(
        valid_sell["timestamp"],
        valid_sell["sell_signal"],
        color="#FF5252",
        marker="v",
        s=180,
        label="Sniper Sell",
        zorder=5,
        edgecolors="black",
    )

    ax1.set_title(
        f"{s.symbol} ({s.timeframe}) - V6: The Final Sniper (MACD + Volatility + RSI)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.1)

    # PLOT 2: MACD HIST + THRESHOLD
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)

    conditions = [
        (df["hist"] >= 0) & (df["hist"] > df["hist"].shift(1)),
        (df["hist"] >= 0) & (df["hist"] <= df["hist"].shift(1)),
        (df["hist"] < 0) & (df["hist"] > df["hist"].shift(1)),
        (df["hist"] < 0) & (df["hist"] <= df["hist"].shift(1)),
    ]
    colors = ["#26a69a", "#b2dfdb", "#ffcdd2", "#ff5252"]
    bar_colors = np.select(conditions, colors, default="#b2dfdb")

    ax2.bar(df["timestamp"], df["hist"], color=bar_colors, width=0.06)
    threshold_line = df["baseline_noise"] * s.sensitivity
    ax2.plot(df["timestamp"], threshold_line, color="#e91e63", linestyle="-", linewidth=1, label="Breakout Line")
    ax2.fill_between(
        df["timestamp"].values,
        (-threshold_line).values,
        threshold_line.values,
        color="#e91e63",
        alpha=0.05,
    )
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.1)

    # PLOT 3: RSI
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df["timestamp"], df["rsi"], color="#9c27b0", label="RSI")
    ax3.axhline(50, color="gray", linestyle="--")
    ax3.axhline(30, color="green", linestyle=":", alpha=0.5)
    ax3.axhline(70, color="red", linestyle=":", alpha=0.5)
    ax3.fill_between(
        df["timestamp"].values,
        50,
        df["rsi"].values,
        where=(df["rsi"].values >= 50),
        color="#9c27b0",
        alpha=0.1,
    )
    ax3.set_ylabel("RSI Momentum")
    ax3.legend()
    ax3.grid(True, alpha=0.1)

    plt.tight_layout()

    if s.save_plot:
        plt.savefig(s.save_plot, dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {s.save_plot}")

    if s.plot:
        plt.show()
    else:
        plt.close()


def _parse_args() -> Settings:
    p = argparse.ArgumentParser(description="V6 Final Sniper (MACD breakout + RSI filter) using ccxt.")
    p.add_argument("--symbol", default=Settings.symbol)
    p.add_argument("--timeframe", default=Settings.timeframe)
    p.add_argument("--exchange", default=Settings.exchange, help="ccxt exchange id (e.g. kucoin, binance)")
    p.add_argument("--limit", type=int, default=Settings.limit)

    p.add_argument("--fast-len", type=int, default=Settings.fast_len)
    p.add_argument("--slow-len", type=int, default=Settings.slow_len)
    p.add_argument("--sig-len", type=int, default=Settings.sig_len)

    p.add_argument("--lookback", type=int, default=Settings.lookback_period)
    p.add_argument("--sensitivity", type=float, default=Settings.sensitivity)

    p.add_argument("--rsi-filter", action=argparse.BooleanOptionalAction, default=Settings.use_rsi_filter)
    p.add_argument("--rsi-len", type=int, default=Settings.rsi_length)
    p.add_argument("--rsi-threshold", type=float, default=Settings.rsi_threshold)

    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-plot", default=None, help="If set, saves plot to this path (png recommended).")

    a = p.parse_args()
    return Settings(
        symbol=a.symbol,
        timeframe=a.timeframe,
        exchange=a.exchange,
        limit=a.limit,
        fast_len=a.fast_len,
        slow_len=a.slow_len,
        sig_len=a.sig_len,
        lookback_period=a.lookback,
        sensitivity=a.sensitivity,
        use_rsi_filter=a.rsi_filter,
        rsi_length=a.rsi_len,
        rsi_threshold=a.rsi_threshold,
        plot=a.plot,
        save_plot=a.save_plot,
    )


def main() -> int:
    s = _parse_args()
    print(f"Scannen met V6 (RSI + Lagged Breakout) op {s.symbol} ({s.timeframe}) via {s.exchange}...")

    df = _load_ohlcv(s.exchange, s.symbol, s.timeframe, s.limit)
    df = compute_indicators(df, s)
    df = generate_signals(df, s)

    last_buy_ts = df.dropna(subset=["buy_signal"]).tail(1)["timestamp"].tolist()
    last_sell_ts = df.dropna(subset=["sell_signal"]).tail(1)["timestamp"].tolist()
    if last_buy_ts:
        print(f"Last BUY signal:  {last_buy_ts[0]}")
    if last_sell_ts:
        print(f"Last SELL signal: {last_sell_ts[0]}")
    if not last_buy_ts and not last_sell_ts:
        print("No signals found in the last window.")

    if s.plot or s.save_plot:
        plot_chart(df, s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

