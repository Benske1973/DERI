import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class V6Config:
    timeframe: str = "4h"
    limit: int = 500
    # MACD
    fast_len: int = 12
    slow_len: int = 26
    sig_len: int = 9
    # Lagged Breakout
    lookback_period: int = 20
    sensitivity: float = 3.0
    # RSI filter
    use_rsi_filter: bool = True
    rsi_threshold: float = 50.0


def calc_ema(source: pd.Series, length: int) -> pd.Series:
    return source.ewm(span=length, adjust=False).mean()


def compute_indicators(df: pd.DataFrame, cfg: V6Config) -> pd.DataFrame:
    out = df.copy()

    # MACD histogram
    out["ma_fast"] = calc_ema(out["close"], cfg.fast_len)
    out["ma_slow"] = calc_ema(out["close"], cfg.slow_len)
    out["macd"] = out["ma_fast"] - out["ma_slow"]
    out["signal"] = calc_ema(out["macd"], cfg.sig_len)
    out["hist"] = out["macd"] - out["signal"]

    # Lagged volatility baseline
    out["abs_hist"] = out["hist"].abs()
    out["baseline_noise"] = (
        out["abs_hist"].rolling(window=cfg.lookback_period).mean().shift(1)
    )
    eps = 1e-8
    out["baseline_noise"] = out["baseline_noise"].replace(0, eps)

    # RSI (Wilder-style smoothing via EWM with alpha=1/14)
    delta = out["close"].diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    return out


def generate_signals(df: pd.DataFrame, cfg: V6Config) -> pd.DataFrame:
    out = df.copy()
    buy_signals = np.full(len(out), np.nan, dtype=float)
    sell_signals = np.full(len(out), np.nan, dtype=float)

    start_index = cfg.lookback_period + 2
    for i in range(start_index, len(out)):
        hist_curr = float(out["hist"].iloc[i])
        hist_prev = float(out["hist"].iloc[i - 1])
        baseline = out["baseline_noise"].iloc[i]
        rsi_curr = out["rsi"].iloc[i]

        if pd.isna(baseline) or pd.isna(rsi_curr):
            continue

        is_explosion = hist_curr > (float(baseline) * cfg.sensitivity)
        pass_rsi = (float(rsi_curr) > cfg.rsi_threshold) if cfg.use_rsi_filter else True

        if hist_curr > 0 and hist_curr > hist_prev and is_explosion and pass_rsi:
            buy_signals[i] = float(out["low"].iloc[i]) * 0.95

        is_dump = hist_curr < -(float(baseline) * cfg.sensitivity)
        pass_rsi_sell = (float(rsi_curr) < 50.0) if cfg.use_rsi_filter else True

        if hist_curr < 0 and hist_curr < hist_prev and is_dump and pass_rsi_sell:
            sell_signals[i] = float(out["high"].iloc[i]) * 1.05

    out["buy_signal"] = buy_signals
    out["sell_signal"] = sell_signals
    return out


def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    try:
        import ccxt  # lazy import so other scripts still run
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ccxt is not installed. Install deps from requirements.txt (pip install -r requirements.txt)."
        ) from e

    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    if not ohlcv:
        raise RuntimeError("No OHLCV data returned.")

    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def plot_v6(df: pd.DataFrame, symbol: str, cfg: V6Config, *, show: bool, save_path: str | None) -> None:
    # Headless-safe backend by default
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 12))

    # Plot 1: price + signals
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
        f"{symbol} ({cfg.timeframe}) - V6: The Final Sniper (MACD + Volatility + RSI)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.1)

    # Plot 2: MACD hist + breakout line
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
    threshold_line = df["baseline_noise"] * cfg.sensitivity
    ax2.plot(
        df["timestamp"],
        threshold_line,
        color="#e91e63",
        linestyle="-",
        linewidth=1,
        label="Breakout Line",
    )
    ax2.fill_between(
        df["timestamp"],
        -threshold_line,
        threshold_line,
        color="#e91e63",
        alpha=0.05,
    )
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.1)

    # Plot 3: RSI
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df["timestamp"], df["rsi"], color="#9c27b0", label="RSI")
    ax3.axhline(50, color="gray", linestyle="--")
    ax3.axhline(30, color="green", linestyle=":", alpha=0.5)
    ax3.axhline(70, color="red", linestyle=":", alpha=0.5)
    ax3.fill_between(
        df["timestamp"],
        50,
        df["rsi"],
        where=(df["rsi"] >= 50),
        color="#9c27b0",
        alpha=0.1,
    )
    ax3.set_ylabel("RSI Momentum")
    ax3.legend()
    ax3.grid(True, alpha=0.1)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=140)
    if show:  # pragma: no cover
        plt.show()
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6: Final Sniper (MACD + Lagged Breakout + RSI)")
    p.add_argument("--symbol", default="PRCL/USDT", help="Market symbol (ccxt format), e.g. PRCL/USDT")
    p.add_argument("--timeframe", default="4h", help="Timeframe, e.g. 15m, 1h, 4h, 1d")
    p.add_argument("--limit", type=int, default=500, help="Number of candles to fetch")
    p.add_argument("--lookback", type=int, default=20, help="Baseline lookback window")
    p.add_argument("--sensitivity", type=float, default=3.0, help="Breakout sensitivity multiplier")
    p.add_argument("--no-rsi-filter", action="store_true", help="Disable RSI filter")
    p.add_argument("--rsi-threshold", type=float, default=50.0, help="RSI threshold for buys")
    p.add_argument("--no-plot", action="store_true", help="Do not generate a plot image")
    p.add_argument("--show", action="store_true", help="Show plot window (not headless-safe)")
    p.add_argument(
        "--save",
        default=None,
        help="Save plot path (default: output/v6_<symbol>_<timeframe>.png)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = V6Config(
        timeframe=args.timeframe,
        limit=args.limit,
        lookback_period=args.lookback,
        sensitivity=args.sensitivity,
        use_rsi_filter=not args.no_rsi_filter,
        rsi_threshold=args.rsi_threshold,
    )

    print(f"Scannen met V6 (RSI + Lagged Breakout) op {args.symbol} ({cfg.timeframe})...")
    df = fetch_ohlcv_ccxt(args.symbol, cfg.timeframe, cfg.limit)
    df = compute_indicators(df, cfg)
    df = generate_signals(df, cfg)

    buys = df.dropna(subset=["buy_signal"])
    sells = df.dropna(subset=["sell_signal"])
    print(f"Buy signals: {len(buys)} | Sell signals: {len(sells)} | Candles: {len(df)}")
    if len(buys) > 0:
        last = buys.iloc[-1]
        print(f"Last BUY:  {last['timestamp']} | close={last['close']:.6g} | rsi={last['rsi']:.2f}")
    if len(sells) > 0:
        last = sells.iloc[-1]
        print(f"Last SELL: {last['timestamp']} | close={last['close']:.6g} | rsi={last['rsi']:.2f}")

    if not args.no_plot:
        if args.save is None:
            safe_symbol = args.symbol.replace("/", "_")
            save_path = os.path.join("output", f"v6_{safe_symbol}_{cfg.timeframe}.png")
        else:
            save_path = args.save
        plot_v6(df, args.symbol, cfg, show=args.show, save_path=save_path)
        if not args.show:
            print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
