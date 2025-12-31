# liquidity_sweep_scanner.py - Liquidity Sweep Detector (Long Wick Hammers)
# Scant op 1 minuut chart voor extreme lower wicks (liquidity grabs)

import asyncio
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Wick Ratio Settings
MIN_WICK_BODY_RATIO = 2.0      # Lower wick moet minstens 2x body zijn
MIN_WICK_TOTAL_RATIO = 0.65    # Lower wick moet 65%+ van totale candle zijn
MAX_UPPER_WICK_RATIO = 0.15    # Upper wick mag max 15% van totale candle zijn
MIN_WICK_SIZE_PCT = 0.3        # Minimale wick grootte in % van prijs

# Volume Settings
MIN_VOLUME_24H = 50000         # Minimum 24h volume
VOLUME_SPIKE_MULT = 1.5        # Volume moet 1.5x gemiddelde zijn

# Scan Settings
TIMEFRAME = "1min"
MAX_CONCURRENT = 15
RATE_LIMIT_DELAY = 0.08
SCAN_INTERVAL = 30             # Scan elke 30 seconden (1min candles)
LOOKBACK_CANDLES = 3           # Check laatste 3 candles

# API
KUCOIN_API = "https://api.kucoin.com"


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
#                      DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class SweepSignal:
    symbol: str
    price: float
    sweep_type: str           # HAMMER, INVERSE_HAMMER
    wick_ratio: float         # Lower wick / body ratio
    wick_pct: float           # Lower wick % of total candle
    wick_size_pct: float      # Wick size as % of price
    volume_ratio: float       # Current vol / avg vol
    candle_ago: int           # 0 = current, 1 = previous, etc
    signal_strength: str      # EXTREME, STRONG, MODERATE
    open_price: float
    close_price: float
    high_price: float
    low_price: float


# ═══════════════════════════════════════════════════════════════
#                      DETECTION LOGIC
# ═══════════════════════════════════════════════════════════════

def analyze_candle(open_p: float, high_p: float, low_p: float, close_p: float,
                   volume: float, avg_volume: float) -> Optional[dict]:
    """
    Analyseer een candle voor liquidity sweep (hammer) patroon.

    Hammer karakteristieken:
    - Kleine body bovenaan
    - Lange lower wick (liquidity grab)
    - Kleine of geen upper wick
    """

    # Candle metrics
    body = abs(close_p - open_p)
    total_range = high_p - low_p

    if total_range == 0:
        return None

    # Wick calculations
    if close_p >= open_p:  # Bullish candle (groen)
        upper_wick = high_p - close_p
        lower_wick = open_p - low_p
    else:  # Bearish candle (rood)
        upper_wick = high_p - open_p
        lower_wick = close_p - low_p

    # Ratios
    wick_body_ratio = lower_wick / body if body > 0 else float('inf')
    wick_total_ratio = lower_wick / total_range
    upper_wick_ratio = upper_wick / total_range
    wick_size_pct = (lower_wick / close_p) * 100
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

    # ═══ HAMMER DETECTION (Bullish Sweep) ═══
    is_hammer = (
        wick_body_ratio >= MIN_WICK_BODY_RATIO and
        wick_total_ratio >= MIN_WICK_TOTAL_RATIO and
        upper_wick_ratio <= MAX_UPPER_WICK_RATIO and
        wick_size_pct >= MIN_WICK_SIZE_PCT
    )

    if not is_hammer:
        return None

    # Signal strength
    if wick_body_ratio >= 5.0 and wick_total_ratio >= 0.80 and volume_ratio >= 2.0:
        strength = "EXTREME"
    elif wick_body_ratio >= 3.5 and wick_total_ratio >= 0.70 and volume_ratio >= 1.5:
        strength = "STRONG"
    elif wick_body_ratio >= MIN_WICK_BODY_RATIO:
        strength = "MODERATE"
    else:
        return None

    return {
        'sweep_type': 'HAMMER',
        'wick_ratio': wick_body_ratio,
        'wick_pct': wick_total_ratio * 100,
        'wick_size_pct': wick_size_pct,
        'volume_ratio': volume_ratio,
        'strength': strength,
        'open': open_p,
        'close': close_p,
        'high': high_p,
        'low': low_p,
    }


def analyze_inverse_candle(open_p: float, high_p: float, low_p: float, close_p: float,
                           volume: float, avg_volume: float) -> Optional[dict]:
    """
    Analyseer voor inverse hammer (bearish sweep bovenaan).
    Lange upper wick = liquidity grab boven.
    """

    body = abs(close_p - open_p)
    total_range = high_p - low_p

    if total_range == 0:
        return None

    if close_p >= open_p:
        upper_wick = high_p - close_p
        lower_wick = open_p - low_p
    else:
        upper_wick = high_p - open_p
        lower_wick = close_p - low_p

    wick_body_ratio = upper_wick / body if body > 0 else float('inf')
    wick_total_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    wick_size_pct = (upper_wick / close_p) * 100
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

    is_inverse = (
        wick_body_ratio >= MIN_WICK_BODY_RATIO and
        wick_total_ratio >= MIN_WICK_TOTAL_RATIO and
        lower_wick_ratio <= MAX_UPPER_WICK_RATIO and
        wick_size_pct >= MIN_WICK_SIZE_PCT
    )

    if not is_inverse:
        return None

    if wick_body_ratio >= 5.0 and wick_total_ratio >= 0.80 and volume_ratio >= 2.0:
        strength = "EXTREME"
    elif wick_body_ratio >= 3.5 and wick_total_ratio >= 0.70 and volume_ratio >= 1.5:
        strength = "STRONG"
    elif wick_body_ratio >= MIN_WICK_BODY_RATIO:
        strength = "MODERATE"
    else:
        return None

    return {
        'sweep_type': 'INVERSE',
        'wick_ratio': wick_body_ratio,
        'wick_pct': wick_total_ratio * 100,
        'wick_size_pct': wick_size_pct,
        'volume_ratio': volume_ratio,
        'strength': strength,
        'open': open_p,
        'close': close_p,
        'high': high_p,
        'low': low_p,
    }


# ═══════════════════════════════════════════════════════════════
#                      API FUNCTIONS
# ═══════════════════════════════════════════════════════════════

async def get_symbols(session: aiohttp.ClientSession) -> List[str]:
    """Haal USDT symbols met > 50K volume."""
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


async def get_candles(session: aiohttp.ClientSession, symbol: str,
                      semaphore: asyncio.Semaphore) -> Optional[List[dict]]:
    """Haal 1min candles op."""
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_DELAY)
        try:
            url = f"{KUCOIN_API}/api/v1/market/candles?symbol={symbol}&type={TIMEFRAME}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

            if data.get('code') != '200000' or not data.get('data'):
                return None

            # KuCoin: [timestamp, open, close, high, low, volume, turnover]
            # Nieuwste eerst, dus we reversen niet
            candles = []
            for c in data['data'][:50]:  # Laatste 50 candles
                candles.append({
                    'ts': int(c[0]),
                    'open': float(c[1]),
                    'close': float(c[2]),
                    'high': float(c[3]),
                    'low': float(c[4]),
                    'volume': float(c[5]),
                })

            return candles
        except:
            return None


# ═══════════════════════════════════════════════════════════════
#                      SCANNER
# ═══════════════════════════════════════════════════════════════

async def scan_symbol(session: aiohttp.ClientSession, symbol: str,
                      semaphore: asyncio.Semaphore) -> List[SweepSignal]:
    """Scan een symbol voor liquidity sweeps."""
    candles = await get_candles(session, symbol, semaphore)
    if not candles or len(candles) < 20:
        return []

    # Bereken gemiddeld volume (candles 10-30)
    avg_volume = sum(c['volume'] for c in candles[10:30]) / 20

    signals = []

    # Check laatste LOOKBACK_CANDLES candles
    for i in range(LOOKBACK_CANDLES):
        if i >= len(candles):
            break

        c = candles[i]

        # Check voor bullish sweep (hammer)
        result = analyze_candle(
            c['open'], c['high'], c['low'], c['close'],
            c['volume'], avg_volume
        )

        if result:
            signals.append(SweepSignal(
                symbol=symbol,
                price=c['close'],
                sweep_type=result['sweep_type'],
                wick_ratio=result['wick_ratio'],
                wick_pct=result['wick_pct'],
                wick_size_pct=result['wick_size_pct'],
                volume_ratio=result['volume_ratio'],
                candle_ago=i,
                signal_strength=result['strength'],
                open_price=result['open'],
                close_price=result['close'],
                high_price=result['high'],
                low_price=result['low'],
            ))

        # Check voor bearish sweep (inverse hammer)
        result_inv = analyze_inverse_candle(
            c['open'], c['high'], c['low'], c['close'],
            c['volume'], avg_volume
        )

        if result_inv:
            signals.append(SweepSignal(
                symbol=symbol,
                price=c['close'],
                sweep_type=result_inv['sweep_type'],
                wick_ratio=result_inv['wick_ratio'],
                wick_pct=result_inv['wick_pct'],
                wick_size_pct=result_inv['wick_size_pct'],
                volume_ratio=result_inv['volume_ratio'],
                candle_ago=i,
                signal_strength=result_inv['strength'],
                open_price=result_inv['open'],
                close_price=result_inv['close'],
                high_price=result_inv['high'],
                low_price=result_inv['low'],
            ))

    return signals


def print_results(all_signals: List[SweepSignal], scan_time: float, total_coins: int):
    """Print scan resultaten."""

    # Separate hammer (bullish) and inverse (bearish) signals
    hammers = [s for s in all_signals if s.sweep_type == 'HAMMER']
    inverse = [s for s in all_signals if s.sweep_type == 'INVERSE']

    # Sort by strength and wick ratio
    strength_order = {'EXTREME': 0, 'STRONG': 1, 'MODERATE': 2}
    hammers.sort(key=lambda x: (strength_order.get(x.signal_strength, 99), -x.wick_ratio))
    inverse.sort(key=lambda x: (strength_order.get(x.signal_strength, 99), -x.wick_ratio))

    print(f"\n{'═'*75}")
    print(f"{C.C}{C.BOLD}  🎯 LIQUIDITY SWEEP SCANNER - 1 MIN CHART{C.END}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {total_coins} coins | {scan_time:.1f}s")
    print(f"{'═'*75}")

    # ═══ BULLISH SWEEPS (Hammers) ═══
    print(f"\n{C.G}{C.BOLD}  🔨 BULLISH LIQUIDITY SWEEPS - HAMMERS ({len(hammers)}){C.END}")
    print(f"  {C.DIM}Lange lower wick = liquidity grab onder support = LONG setup{C.END}")
    print(f"  {'─'*71}")

    if hammers:
        print(f"  {'Symbol':<14} {'Strength':<10} {'Wick/Body':>9} {'Wick%':>7} {'WickSize':>8} {'Vol':>5} {'Ago':>4}")
        print(f"  {'─'*71}")

        for s in hammers[:15]:
            # Color by strength
            if s.signal_strength == 'EXTREME':
                col = C.Y + C.BOLD
                icon = "🔥"
            elif s.signal_strength == 'STRONG':
                col = C.G + C.BOLD
                icon = "💪"
            else:
                col = C.G
                icon = "📍"

            ago_str = "NOW" if s.candle_ago == 0 else f"-{s.candle_ago}"

            print(f"  {s.symbol:<14} {col}{icon} {s.signal_strength:<8}{C.END} "
                  f"{s.wick_ratio:>8.1f}x {s.wick_pct:>6.0f}% {s.wick_size_pct:>7.2f}% "
                  f"{s.volume_ratio:>4.1f}x {ago_str:>4}")
    else:
        print(f"  {C.DIM}Geen bullish sweeps gevonden{C.END}")

    # ═══ BEARISH SWEEPS (Inverse Hammers) ═══
    print(f"\n{C.R}{C.BOLD}  🔻 BEARISH LIQUIDITY SWEEPS - INVERSE ({len(inverse)}){C.END}")
    print(f"  {C.DIM}Lange upper wick = liquidity grab boven resistance = SHORT setup{C.END}")
    print(f"  {'─'*71}")

    if inverse:
        print(f"  {'Symbol':<14} {'Strength':<10} {'Wick/Body':>9} {'Wick%':>7} {'WickSize':>8} {'Vol':>5} {'Ago':>4}")
        print(f"  {'─'*71}")

        for s in inverse[:15]:
            if s.signal_strength == 'EXTREME':
                col = C.Y + C.BOLD
                icon = "🔥"
            elif s.signal_strength == 'STRONG':
                col = C.R + C.BOLD
                icon = "💪"
            else:
                col = C.R
                icon = "📍"

            ago_str = "NOW" if s.candle_ago == 0 else f"-{s.candle_ago}"

            print(f"  {s.symbol:<14} {col}{icon} {s.signal_strength:<8}{C.END} "
                  f"{s.wick_ratio:>8.1f}x {s.wick_pct:>6.0f}% {s.wick_size_pct:>7.2f}% "
                  f"{s.volume_ratio:>4.1f}x {ago_str:>4}")
    else:
        print(f"  {C.DIM}Geen bearish sweeps gevonden{C.END}")

    # ═══ TOP PICKS ═══
    extreme_bulls = [s for s in hammers if s.signal_strength == 'EXTREME' and s.candle_ago == 0]
    extreme_bears = [s for s in inverse if s.signal_strength == 'EXTREME' and s.candle_ago == 0]

    if extreme_bulls or extreme_bears:
        print(f"\n{C.Y}{C.BOLD}  ⚡ EXTREME SIGNALS - TRADE NU!{C.END}")
        print(f"  {'─'*71}")

        for s in extreme_bulls[:3]:
            sweep_pct = ((s.high_price - s.low_price) / s.low_price) * 100
            print(f"  {C.G}🔥 LONG:{C.END} {C.BOLD}{s.symbol}{C.END}")
            print(f"     Price: {s.close_price:.8f} | Sweep: {sweep_pct:.2f}% | Wick: {s.wick_ratio:.1f}x body")
            print(f"     Entry: {s.close_price:.8f} | SL: {s.low_price:.8f} | TP: {s.high_price * 1.01:.8f}")

        for s in extreme_bears[:3]:
            sweep_pct = ((s.high_price - s.low_price) / s.high_price) * 100
            print(f"  {C.R}🔥 SHORT:{C.END} {C.BOLD}{s.symbol}{C.END}")
            print(f"     Price: {s.close_price:.8f} | Sweep: {sweep_pct:.2f}% | Wick: {s.wick_ratio:.1f}x body")
            print(f"     Entry: {s.close_price:.8f} | SL: {s.high_price:.8f} | TP: {s.low_price * 0.99:.8f}")

    print(f"\n{'═'*75}")
    print(f"  {C.DIM}Wick/Body: lower wick vs body ratio (hoger = sterker signaal){C.END}")
    print(f"  {C.DIM}Wick%: lower wick als % van totale candle (65%+ = liquidity sweep){C.END}")
    print(f"  {C.DIM}WickSize: wick grootte als % van prijs (volatiliteit){C.END}")
    print(f"{'═'*75}\n")


async def run_scan():
    """Run een volledige scan."""
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Get symbols
        symbols = await get_symbols(session)
        if not symbols:
            print("Geen symbols gevonden!")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(symbols)} coins voor liquidity sweeps...")

        # Scan all symbols
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [scan_symbol(session, sym, semaphore) for sym in symbols]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_signals = []
        for signals in results:
            all_signals.extend(signals)

        scan_time = time.time() - start_time
        print_results(all_signals, scan_time, len(symbols))

        return all_signals


async def main():
    """Main loop."""
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════════╗
║           🎯 LIQUIDITY SWEEP SCANNER                                     ║
║                                                                          ║
║  Detecteert hammer candles met extreem lange wicks op 1 MIN chart        ║
║                                                                          ║
║  🔨 HAMMER = Lange lower wick = Liquidity grab ONDER = LONG setup        ║
║  🔻 INVERSE = Lange upper wick = Liquidity grab BOVEN = SHORT setup      ║
║                                                                          ║
║  Signal Strength:                                                        ║
║  • EXTREME: Wick 5x+ body, 80%+ van candle, 2x volume                   ║
║  • STRONG:  Wick 3.5x+ body, 70%+ van candle, 1.5x volume               ║
║  • MODERATE: Wick 2x+ body, 65%+ van candle                             ║
╚══════════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    print(f"{C.Y}Settings:{C.END}")
    print(f"  • Timeframe: {TIMEFRAME}")
    print(f"  • Min Wick/Body Ratio: {MIN_WICK_BODY_RATIO}x")
    print(f"  • Min Wick % of Candle: {MIN_WICK_TOTAL_RATIO * 100:.0f}%")
    print(f"  • Min Volume 24h: ${MIN_VOLUME_24H:,}")
    print(f"  • Scan Interval: {SCAN_INTERVAL}s")
    print(f"  • Lookback: {LOOKBACK_CANDLES} candles")
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
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
