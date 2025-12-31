# confluence_scanner.py - Bulls vs Bears Confluence Scanner voor KuCoin
# Gebaseerd op jouw TradingView indicator - detecteert liquidity sweeps & bottoms
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import deque

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Indicator Settings (zelfde als je Pine Script)
Z_LENGTH = 75
RSI_LENGTH = 14
EMA_FAST = 12
EMA_SLOW = 26
VOLUME_THRESHOLD = 1.5

# Score Thresholds
SCORE_STRONG = 8      # Strong signal
SCORE_MODERATE = 6    # Moderate signal
SCORE_WEAK = 4        # Weak signal

# Scan Settings
TIMEFRAME = "15min"    # Candle timeframe
MIN_VOLUME_24H = 100000
MAX_RESULTS = 20
SCAN_INTERVAL = 60     # Seconden tussen scans

# API Settings
KUCOIN_API = "https://api.kucoin.com"
MAX_CONCURRENT = 10
RATE_LIMIT_DELAY = 0.1


# ═══════════════════════════════════════════════════════════════
#                      KLEUREN
# ═══════════════════════════════════════════════════════════════
class C:
    G = '\033[92m'
    R = '\033[91m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    W = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


# ═══════════════════════════════════════════════════════════════
#                      DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalResult:
    symbol: str
    bull_score: float
    bear_score: float
    signal_type: str  # PREMIUM_BUY, STRONG_BUY, MODERATE_BUY, etc.
    z_score: float
    rsi: float
    macd_bullish: bool
    trend_bullish: bool
    volume_spike: bool
    price: float

    # Breakdown
    z_points: float
    rsi_points: float
    macd_points: float
    trend_points: float
    vol_points: float
    momentum_points: float
    bonus_points: float

    # Special signals
    crossover_signal: bool
    bottom_signal: bool
    top_signal: bool


# ═══════════════════════════════════════════════════════════════
#                   INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calculate_z_score(close: pd.Series, length: int = Z_LENGTH) -> pd.Series:
    """Z-Score = (close - SMA) / StdDev"""
    sma = close.rolling(window=length).mean()
    std = close.rolling(window=length).std()
    return (close - sma) / std


def calculate_rsi(close: pd.Series, length: int = RSI_LENGTH) -> pd.Series:
    """RSI berekening"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series, fast: int = EMA_FAST, slow: int = EMA_SLOW, signal: int = 9):
    """MACD berekening"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_ema(close: pd.Series, length: int) -> pd.Series:
    """EMA berekening"""
    return close.ewm(span=length, adjust=False).mean()


def is_rising(series: pd.Series, periods: int = 3) -> bool:
    """Check of serie stijgend is"""
    if len(series) < periods + 1:
        return False
    return all(series.iloc[-(i+1)] > series.iloc[-(i+2)] for i in range(periods))


def is_falling(series: pd.Series, periods: int = 3) -> bool:
    """Check of serie dalend is"""
    if len(series) < periods + 1:
        return False
    return all(series.iloc[-(i+1)] < series.iloc[-(i+2)] for i in range(periods))


# ═══════════════════════════════════════════════════════════════
#                   CONFLUENCE SCORE SYSTEM
# ═══════════════════════════════════════════════════════════════

def calculate_confluence_score(df: pd.DataFrame) -> Dict:
    """
    Bereken de volledige confluence score zoals in je Pine Script.
    Returns dict met alle scores en signalen.
    """
    if len(df) < Z_LENGTH + 10:
        return None

    close = df['close']
    open_ = df['open']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # ═══ INDICATOR CALCULATIONS ═══
    z_score = calculate_z_score(close, Z_LENGTH)
    z_sma = z_score.rolling(window=14).mean()
    z_slope = z_sma - z_sma.shift(3)

    rsi = calculate_rsi(close, RSI_LENGTH)
    rsi_slope = rsi - rsi.shift(3)

    rsi_scaled = (rsi - 50) / 10  # Voor crossover vergelijking

    macd_line, signal_line, hist = calculate_macd(close)

    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)

    vol_sma = volume.rolling(window=20).mean()

    # ═══ HUIDIGE WAARDEN ═══
    curr_z = z_score.iloc[-1]
    curr_rsi = rsi.iloc[-1]
    curr_rsi_scaled = rsi_scaled.iloc[-1]
    curr_z_slope = z_slope.iloc[-1] if not pd.isna(z_slope.iloc[-1]) else 0
    curr_rsi_slope = rsi_slope.iloc[-1] if not pd.isna(rsi_slope.iloc[-1]) else 0
    curr_macd = macd_line.iloc[-1]
    curr_signal = signal_line.iloc[-1]
    curr_ema_fast = ema_fast.iloc[-1]
    curr_ema_slow = ema_slow.iloc[-1]
    curr_vol = volume.iloc[-1]
    curr_vol_sma = vol_sma.iloc[-1]
    curr_close = close.iloc[-1]
    curr_open = open_.iloc[-1]

    # ═══ CONDITIONS ═══
    macd_bullish = curr_macd > curr_signal
    macd_bearish = curr_macd < curr_signal
    macd_cross_up = macd_line.iloc[-2] < signal_line.iloc[-2] and curr_macd > curr_signal
    macd_cross_down = macd_line.iloc[-2] > signal_line.iloc[-2] and curr_macd < curr_signal

    trend_bullish = curr_ema_fast > curr_ema_slow
    trend_bearish = curr_ema_fast < curr_ema_slow

    high_volume = curr_vol > curr_vol_sma * VOLUME_THRESHOLD
    volume_spike = curr_vol > curr_vol_sma * 2.0
    volume_increasing = curr_vol > volume.iloc[-2]

    bullish_candle = curr_close > curr_open
    bearish_candle = curr_close < curr_open

    z_rising = is_rising(z_score, 3)
    z_falling = is_falling(z_score, 3)
    rsi_rising = is_rising(rsi, 3)
    rsi_falling = is_falling(rsi, 3)

    # ═══ CROSSOVER SIGNALS ═══
    prev_rsi_scaled = rsi_scaled.iloc[-2]
    prev_z = z_score.iloc[-2]

    rsi_cross_z_up = prev_rsi_scaled < prev_z and curr_rsi_scaled > curr_z
    rsi_cross_z_down = prev_rsi_scaled > prev_z and curr_rsi_scaled < curr_z

    # KRACHTIG SIGNAAL: Crossover + beide stijgend + oversold
    bullish_momentum_cross = rsi_cross_z_up and rsi_rising and z_rising and curr_z < -1
    bearish_momentum_cross = rsi_cross_z_down and rsi_falling and z_falling and curr_z > 1

    # ═══ BOTTOM/TOP DETECTION ═══
    candle_range = high.iloc[-1] - low.iloc[-1]
    body_size = abs(curr_close - curr_open)

    extreme_oversold = curr_z < -2.5 or curr_rsi < 25
    extreme_overbought = curr_z > 2.5 or curr_rsi > 75

    volume_climax = volume_spike and bearish_candle
    volume_exhaustion = volume_spike and bullish_candle

    hammer_pattern = body_size < candle_range * 0.3 and low.iloc[-1] == low.tail(10).min()
    shooting_star = body_size < candle_range * 0.3 and high.iloc[-1] == high.tail(10).max()

    potential_bottom = extreme_oversold and (volume_climax or hammer_pattern)
    potential_top = extreme_overbought and (volume_exhaustion or shooting_star)

    bottom_confirmed = potential_bottom and z_rising and rsi_rising and bullish_candle
    top_confirmed = potential_top and z_falling and rsi_falling and bearish_candle

    # ═══ BULL SCORE CALCULATION ═══
    z_bull_points = 0.0
    if curr_z < -1: z_bull_points += 1.0
    if curr_z < -2: z_bull_points += 1.0
    if curr_z < -3: z_bull_points += 1.0

    rsi_bull_points = 0.0
    if curr_rsi < 40: rsi_bull_points += 1.0
    if curr_rsi < 30: rsi_bull_points += 1.0

    macd_bull_points = 0.0
    if macd_bullish: macd_bull_points += 1.0
    if macd_cross_up: macd_bull_points += 1.0

    trend_bull_points = 1.0 if trend_bullish else 0.0

    vol_bull_points = 0.0
    if high_volume and volume_increasing:
        vol_bull_points = 1.0
    elif high_volume:
        vol_bull_points = 0.5

    momentum_bull_points = 0.0
    if curr_z_slope > 0 and bullish_candle:
        momentum_bull_points = 1.0
    elif curr_z_slope > 0 or bullish_candle:
        momentum_bull_points = 0.5

    bonus_bull_points = 0.0
    if bullish_momentum_cross: bonus_bull_points += 1.5
    if bottom_confirmed:
        bonus_bull_points += 1.5
    elif potential_bottom:
        bonus_bull_points += 0.5

    bull_score = (z_bull_points + rsi_bull_points + macd_bull_points +
                  trend_bull_points + vol_bull_points + momentum_bull_points + bonus_bull_points)

    # ═══ BEAR SCORE CALCULATION ═══
    z_bear_points = 0.0
    if curr_z > 1: z_bear_points += 1.0
    if curr_z > 2: z_bear_points += 1.0
    if curr_z > 3: z_bear_points += 1.0

    rsi_bear_points = 0.0
    if curr_rsi > 60: rsi_bear_points += 1.0
    if curr_rsi > 70: rsi_bear_points += 1.0

    macd_bear_points = 0.0
    if macd_bearish: macd_bear_points += 1.0
    if macd_cross_down: macd_bear_points += 1.0

    trend_bear_points = 1.0 if trend_bearish else 0.0

    vol_bear_points = 0.0
    if high_volume and volume_increasing:
        vol_bear_points = 1.0
    elif high_volume:
        vol_bear_points = 0.5

    momentum_bear_points = 0.0
    if curr_z_slope < 0 and bearish_candle:
        momentum_bear_points = 1.0
    elif curr_z_slope < 0 or bearish_candle:
        momentum_bear_points = 0.5

    bonus_bear_points = 0.0
    if bearish_momentum_cross: bonus_bear_points += 1.5
    if top_confirmed:
        bonus_bear_points += 1.5
    elif potential_top:
        bonus_bear_points += 0.5

    bear_score = (z_bear_points + rsi_bear_points + macd_bear_points +
                  trend_bear_points + vol_bear_points + momentum_bear_points + bonus_bear_points)

    # ═══ SIGNAL CLASSIFICATION ═══
    strong_bull = bull_score >= SCORE_STRONG
    moderate_bull = bull_score >= SCORE_MODERATE and bull_score < SCORE_STRONG
    weak_bull = bull_score >= SCORE_WEAK and bull_score < SCORE_MODERATE

    strong_bear = bear_score >= SCORE_STRONG
    moderate_bear = bear_score >= SCORE_MODERATE and bear_score < SCORE_STRONG

    premium_bull = strong_bull and (bullish_momentum_cross or bottom_confirmed)
    premium_bear = strong_bear and (bearish_momentum_cross or top_confirmed)

    # Determine signal type
    if premium_bull:
        signal_type = "💎 PREMIUM BUY"
    elif premium_bear:
        signal_type = "💎 PREMIUM SELL"
    elif strong_bull:
        signal_type = "🔥 STRONG BUY"
    elif strong_bear:
        signal_type = "🔥 STRONG SELL"
    elif bullish_momentum_cross:
        signal_type = "⚡ CROSS BUY"
    elif bearish_momentum_cross:
        signal_type = "⚡ CROSS SELL"
    elif bottom_confirmed:
        signal_type = "💎 BOTTOM"
    elif top_confirmed:
        signal_type = "💎 TOP"
    elif moderate_bull:
        signal_type = "🟢 MODERATE BUY"
    elif moderate_bear:
        signal_type = "🔴 MODERATE SELL"
    elif weak_bull:
        signal_type = "🟡 WEAK BUY"
    else:
        signal_type = "NEUTRAL"

    return {
        'bull_score': bull_score,
        'bear_score': bear_score,
        'signal_type': signal_type,
        'z_score': curr_z,
        'rsi': curr_rsi,
        'macd_bullish': macd_bullish,
        'trend_bullish': trend_bullish,
        'volume_spike': volume_spike,
        'price': curr_close,

        # Breakdown
        'z_points': z_bull_points if bull_score > bear_score else z_bear_points,
        'rsi_points': rsi_bull_points if bull_score > bear_score else rsi_bear_points,
        'macd_points': macd_bull_points if bull_score > bear_score else macd_bear_points,
        'trend_points': trend_bull_points if bull_score > bear_score else trend_bear_points,
        'vol_points': vol_bull_points if bull_score > bear_score else vol_bear_points,
        'momentum_points': momentum_bull_points if bull_score > bear_score else momentum_bear_points,
        'bonus_points': bonus_bull_points if bull_score > bear_score else bonus_bear_points,

        # Special signals
        'crossover_signal': bullish_momentum_cross or bearish_momentum_cross,
        'bottom_signal': bottom_confirmed or potential_bottom,
        'top_signal': top_confirmed or potential_top,

        # Extra info
        'premium_buy': premium_bull,
        'premium_sell': premium_bear,
        'strong_buy': strong_bull,
        'strong_sell': strong_bear,
    }


# ═══════════════════════════════════════════════════════════════
#                      API FUNCTIONS
# ═══════════════════════════════════════════════════════════════

async def get_symbols(session: aiohttp.ClientSession) -> List[str]:
    """Haal high volume USDT symbols op."""
    try:
        async with session.get(f"{KUCOIN_API}/api/v1/market/allTickers") as resp:
            data = await resp.json()

        symbols = []
        for t in data['data'].get('ticker', []):
            sym = t.get('symbol', '')
            if not sym.endswith('-USDT'):
                continue
            if any(x in sym for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-']):
                continue
            try:
                vol = float(t.get('volValue', 0))
                if vol >= MIN_VOLUME_24H:
                    symbols.append(sym)
            except:
                continue

        return symbols
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


async def get_candles(session: aiohttp.ClientSession, symbol: str, semaphore: asyncio.Semaphore) -> Optional[pd.DataFrame]:
    """Haal candle data op voor een symbol."""
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_DELAY)
        try:
            url = f"{KUCOIN_API}/api/v1/market/candles?symbol={symbol}&type={TIMEFRAME}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

            if data.get('code') != '200000' or not data.get('data'):
                return None

            # KuCoin format: [timestamp, open, close, high, low, volume, turnover]
            df = pd.DataFrame(data['data'], columns=['ts', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
            df = df.astype(float)
            df = df.iloc[::-1].reset_index(drop=True)  # Oldest first

            return df
        except:
            return None


# ═══════════════════════════════════════════════════════════════
#                      SCANNER
# ═══════════════════════════════════════════════════════════════

async def scan_symbol(session: aiohttp.ClientSession, symbol: str, semaphore: asyncio.Semaphore) -> Optional[SignalResult]:
    """Scan een symbol voor confluence signals."""
    df = await get_candles(session, symbol, semaphore)
    if df is None or len(df) < Z_LENGTH + 20:
        return None

    result = calculate_confluence_score(df)
    if result is None:
        return None

    # Filter: alleen interessante signalen
    if result['signal_type'] == "NEUTRAL":
        return None

    return SignalResult(
        symbol=symbol,
        bull_score=result['bull_score'],
        bear_score=result['bear_score'],
        signal_type=result['signal_type'],
        z_score=result['z_score'],
        rsi=result['rsi'],
        macd_bullish=result['macd_bullish'],
        trend_bullish=result['trend_bullish'],
        volume_spike=result['volume_spike'],
        price=result['price'],
        z_points=result['z_points'],
        rsi_points=result['rsi_points'],
        macd_points=result['macd_points'],
        trend_points=result['trend_points'],
        vol_points=result['vol_points'],
        momentum_points=result['momentum_points'],
        bonus_points=result['bonus_points'],
        crossover_signal=result['crossover_signal'],
        bottom_signal=result['bottom_signal'],
        top_signal=result['top_signal'],
    )


def print_results(results: List[SignalResult], scan_time: float):
    """Print scan resultaten."""

    # Sort by score (highest first)
    buy_signals = [r for r in results if 'BUY' in r.signal_type or 'BOTTOM' in r.signal_type]
    sell_signals = [r for r in results if 'SELL' in r.signal_type or 'TOP' in r.signal_type]

    buy_signals.sort(key=lambda x: x.bull_score, reverse=True)
    sell_signals.sort(key=lambda x: x.bear_score, reverse=True)

    print(f"\n{'═'*70}")
    print(f"{C.C}{C.BOLD}  BULLS vs BEARS - CONFLUENCE SCANNER{C.END}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Scan time: {scan_time:.1f}s")
    print(f"{'═'*70}")

    # ═══ BUY SIGNALS ═══
    print(f"\n{C.G}{C.BOLD}  🐂 BUY SIGNALS ({len(buy_signals)}){C.END}")
    print(f"  {'─'*66}")

    if buy_signals:
        print(f"  {'Symbol':<14} {'Signal':<18} {'Score':>6} {'Z':>7} {'RSI':>5} {'Special':<12}")
        print(f"  {'─'*66}")

        for r in buy_signals[:MAX_RESULTS]:
            # Signal color
            if 'PREMIUM' in r.signal_type:
                sig_color = C.Y + C.BOLD
            elif 'STRONG' in r.signal_type:
                sig_color = C.G + C.BOLD
            elif 'CROSS' in r.signal_type:
                sig_color = C.C
            else:
                sig_color = C.G

            # Special indicators
            specials = []
            if r.crossover_signal: specials.append("⚡Cross")
            if r.bottom_signal: specials.append("💎Bot")
            if r.volume_spike: specials.append("📊Vol")
            special_str = " ".join(specials) if specials else "—"

            print(f"  {r.symbol:<14} {sig_color}{r.signal_type:<18}{C.END} "
                  f"{C.G}{r.bull_score:>5.1f}{C.END}  "
                  f"{r.z_score:>6.2f}  {r.rsi:>4.0f}  {special_str}")
    else:
        print(f"  {C.DIM}Geen buy signals gevonden{C.END}")

    # ═══ SELL SIGNALS ═══
    print(f"\n{C.R}{C.BOLD}  🐻 SELL SIGNALS ({len(sell_signals)}){C.END}")
    print(f"  {'─'*66}")

    if sell_signals:
        print(f"  {'Symbol':<14} {'Signal':<18} {'Score':>6} {'Z':>7} {'RSI':>5} {'Special':<12}")
        print(f"  {'─'*66}")

        for r in sell_signals[:MAX_RESULTS]:
            if 'PREMIUM' in r.signal_type:
                sig_color = C.Y + C.BOLD
            elif 'STRONG' in r.signal_type:
                sig_color = C.R + C.BOLD
            else:
                sig_color = C.R

            specials = []
            if r.crossover_signal: specials.append("⚡Cross")
            if r.top_signal: specials.append("💎Top")
            if r.volume_spike: specials.append("📊Vol")
            special_str = " ".join(specials) if specials else "—"

            print(f"  {r.symbol:<14} {sig_color}{r.signal_type:<18}{C.END} "
                  f"{C.R}{r.bear_score:>5.1f}{C.END}  "
                  f"{r.z_score:>6.2f}  {r.rsi:>4.0f}  {special_str}")
    else:
        print(f"  {C.DIM}Geen sell signals gevonden{C.END}")

    # ═══ TOP PICKS ═══
    print(f"\n{C.Y}{C.BOLD}  ⭐ TOP PICKS{C.END}")
    print(f"  {'─'*66}")

    # Best buy
    premium_buys = [r for r in buy_signals if 'PREMIUM' in r.signal_type or r.bull_score >= 10]
    if premium_buys:
        best = premium_buys[0]
        print(f"  {C.G}BEST BUY:{C.END}  {C.BOLD}{best.symbol}{C.END} - Score: {best.bull_score:.1f}/12")
        print(f"            Z: {best.z_score:.2f} | RSI: {best.rsi:.0f} | Price: {best.price:.8f}")
        print(f"            Breakdown: Z={best.z_points:.0f} RSI={best.rsi_points:.0f} MACD={best.macd_points:.0f} "
              f"Trend={best.trend_points:.0f} Vol={best.vol_points:.1f} Mom={best.momentum_points:.1f} Bonus={best.bonus_points:.1f}")

    # Best sell
    premium_sells = [r for r in sell_signals if 'PREMIUM' in r.signal_type or r.bear_score >= 10]
    if premium_sells:
        best = premium_sells[0]
        print(f"\n  {C.R}BEST SELL:{C.END} {C.BOLD}{best.symbol}{C.END} - Score: {best.bear_score:.1f}/12")
        print(f"            Z: {best.z_score:.2f} | RSI: {best.rsi:.0f} | Price: {best.price:.8f}")

    print(f"\n{'═'*70}")
    print(f"  {C.DIM}Score guide: 8+ Strong | 10+ Premium | Bonus voor Crossover/Bottom{C.END}")
    print(f"{'═'*70}\n")


async def run_scan():
    """Run een volledige scan."""
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Get symbols
        symbols = await get_symbols(session)
        if not symbols:
            print("Geen symbols gevonden!")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(symbols)} coins op {TIMEFRAME}...")

        # Scan all symbols
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [scan_symbol(session, sym, semaphore) for sym in symbols]
        results = await asyncio.gather(*tasks)

        # Filter None results
        results = [r for r in results if r is not None]

        scan_time = time.time() - start_time
        print_results(results, scan_time)

        return results


async def main():
    """Main loop."""
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║          BULLS vs BEARS - CONFLUENCE SCANNER                        ║
║                                                                      ║
║  🐂 Detecteert: Liquidity sweeps, bottoms, oversold bounces         ║
║  🐻 Detecteert: Tops, overbought reversals, distribution            ║
║  ⚡ Special: RSI x Z-Score crossovers, Premium signals              ║
╚══════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    print(f"{C.Y}Settings:{C.END}")
    print(f"  • Timeframe: {TIMEFRAME}")
    print(f"  • Z-Score Length: {Z_LENGTH}")
    print(f"  • RSI Length: {RSI_LENGTH}")
    print(f"  • Strong Signal: Score >= {SCORE_STRONG}")
    print(f"  • Scan Interval: {SCAN_INTERVAL}s")
    print()

    while True:
        try:
            await run_scan()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Volgende scan over {SCAN_INTERVAL}s...")
            await asyncio.sleep(SCAN_INTERVAL)
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanner gestopt")
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(30)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
