# marubozu_scanner.py - Marubozu Candle Scanner voor KuCoin
# Zoekt naar sterke momentum candles (geen/kleine wicks) op 1 min chart

import asyncio
import aiohttp
import time
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Marubozu Detection Settings
MAX_WICK_RATIO = 0.10            # Max 10% wick van totale candle (0 = perfecte marubozu)
MIN_BODY_RATIO = 0.85            # Body moet minstens 85% van candle zijn
MIN_CANDLE_SIZE_PCT = 0.3        # Minimale candle grootte in % van prijs
STRONG_CANDLE_SIZE_PCT = 0.5     # Sterke candle grootte
EXTREME_CANDLE_SIZE_PCT = 1.0    # Extreme candle grootte

# Volume Settings
MIN_VOLUME_24H = 50000           # Minimum 24h volume
VOLUME_SPIKE_MULT = 1.5          # Volume moet 1.5x gemiddelde zijn voor bevestiging

# Scan Settings
TIMEFRAME = "1min"
MAX_CONCURRENT = 15
RATE_LIMIT_DELAY = 0.08
SCAN_INTERVAL = 30               # Scan elke 30 seconden
LOOKBACK_CANDLES = 5             # Check laatste 5 candles

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
class MarubozuSignal:
    symbol: str
    price: float
    marubozu_type: str           # BULLISH, BEARISH
    strength: str                # PERFECT, STRONG, MODERATE
    body_ratio: float            # Body als % van totale candle
    candle_size_pct: float       # Candle grootte als % van prijs
    upper_wick_pct: float        # Upper wick als % van candle
    lower_wick_pct: float        # Lower wick als % van candle
    volume_ratio: float          # Volume vs gemiddelde
    candle_ago: int              # 0 = huidige candle
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    consecutive: int             # Aantal opeenvolgende marubozu's


# ═══════════════════════════════════════════════════════════════
#                      DETECTION LOGIC
# ═══════════════════════════════════════════════════════════════

def detect_marubozu(open_p: float, high_p: float, low_p: float, close_p: float,
                    volume: float, avg_volume: float) -> Optional[dict]:
    """
    Detecteer Marubozu candle patroon.

    Marubozu kenmerken:
    - Geen of zeer kleine wicks
    - Body = bijna de hele candle
    - Sterke momentum indicator

    Bullish Marubozu: Open ≈ Low, Close ≈ High (groen)
    Bearish Marubozu: Open ≈ High, Close ≈ Low (rood)
    """

    total_range = high_p - low_p
    if total_range == 0 or close_p == 0:
        return None

    body = abs(close_p - open_p)

    # Determine candle direction
    is_bullish = close_p > open_p

    if is_bullish:
        upper_wick = high_p - close_p
        lower_wick = open_p - low_p
    else:
        upper_wick = high_p - open_p
        lower_wick = close_p - low_p

    # Calculate ratios
    body_ratio = body / total_range
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    candle_size_pct = (total_range / close_p) * 100
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

    # ═══ MARUBOZU DETECTION ═══
    # Body moet groot genoeg zijn
    if body_ratio < MIN_BODY_RATIO:
        return None

    # Wicks moeten klein zijn
    if upper_wick_ratio > MAX_WICK_RATIO or lower_wick_ratio > MAX_WICK_RATIO:
        return None

    # Candle moet significant zijn
    if candle_size_pct < MIN_CANDLE_SIZE_PCT:
        return None

    # ═══ STRENGTH CLASSIFICATION ═══
    total_wick_ratio = upper_wick_ratio + lower_wick_ratio

    if total_wick_ratio < 0.02 and candle_size_pct >= EXTREME_CANDLE_SIZE_PCT:
        strength = "PERFECT"
    elif total_wick_ratio < 0.05 and candle_size_pct >= STRONG_CANDLE_SIZE_PCT:
        strength = "STRONG"
    elif body_ratio >= MIN_BODY_RATIO:
        strength = "MODERATE"
    else:
        return None

    return {
        'type': 'BULLISH' if is_bullish else 'BEARISH',
        'strength': strength,
        'body_ratio': body_ratio * 100,
        'candle_size_pct': candle_size_pct,
        'upper_wick_pct': upper_wick_ratio * 100,
        'lower_wick_pct': lower_wick_ratio * 100,
        'volume_ratio': volume_ratio,
        'open': open_p,
        'close': close_p,
        'high': high_p,
        'low': low_p,
    }


def count_consecutive_marubozu(candles: List[dict], direction: str) -> int:
    """Tel opeenvolgende marubozu candles in dezelfde richting."""
    count = 0
    avg_vol = sum(c['volume'] for c in candles[5:25]) / 20 if len(candles) > 25 else 1

    for c in candles:
        result = detect_marubozu(c['open'], c['high'], c['low'], c['close'], c['volume'], avg_vol)
        if result and result['type'] == direction:
            count += 1
        else:
            break

    return count


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

            candles = []
            for c in data['data'][:50]:
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
                      semaphore: asyncio.Semaphore) -> List[MarubozuSignal]:
    """Scan een symbol voor Marubozu candles."""
    candles = await get_candles(session, symbol, semaphore)
    if not candles or len(candles) < 25:
        return []

    # Gemiddeld volume berekenen
    avg_volume = sum(c['volume'] for c in candles[5:25]) / 20

    signals = []

    # Check laatste LOOKBACK_CANDLES
    for i in range(LOOKBACK_CANDLES):
        if i >= len(candles):
            break

        c = candles[i]
        result = detect_marubozu(c['open'], c['high'], c['low'], c['close'], c['volume'], avg_volume)

        if result:
            # Tel opeenvolgende marubozu's
            consecutive = count_consecutive_marubozu(candles[i:], result['type'])

            signals.append(MarubozuSignal(
                symbol=symbol,
                price=c['close'],
                marubozu_type=result['type'],
                strength=result['strength'],
                body_ratio=result['body_ratio'],
                candle_size_pct=result['candle_size_pct'],
                upper_wick_pct=result['upper_wick_pct'],
                lower_wick_pct=result['lower_wick_pct'],
                volume_ratio=result['volume_ratio'],
                candle_ago=i,
                open_price=result['open'],
                close_price=result['close'],
                high_price=result['high'],
                low_price=result['low'],
                consecutive=consecutive,
            ))

    return signals


def print_results(all_signals: List[MarubozuSignal], scan_time: float, total_coins: int):
    """Print scan resultaten."""

    # Separate bullish and bearish
    bullish = [s for s in all_signals if s.marubozu_type == 'BULLISH']
    bearish = [s for s in all_signals if s.marubozu_type == 'BEARISH']

    # Sort by strength and size
    strength_order = {'PERFECT': 0, 'STRONG': 1, 'MODERATE': 2}
    bullish.sort(key=lambda x: (strength_order.get(x.strength, 99), -x.candle_size_pct))
    bearish.sort(key=lambda x: (strength_order.get(x.strength, 99), -x.candle_size_pct))

    print(f"\n{'═'*80}")
    print(f"{C.C}{C.BOLD}  📊 MARUBOZU SCANNER - 1 MIN CHART{C.END}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {total_coins} coins | {scan_time:.1f}s")
    print(f"{'═'*80}")

    # ═══ BULLISH MARUBOZU ═══
    print(f"\n{C.G}{C.BOLD}  🟢 BULLISH MARUBOZU ({len(bullish)}) - LONG MOMENTUM{C.END}")
    print(f"  {C.DIM}Sterke kopers - geen verkoopdruk - prijs zal waarschijnlijk stijgen{C.END}")
    print(f"  {'─'*76}")

    if bullish:
        print(f"  {'Symbol':<14} {'Strength':<10} {'Body%':>6} {'Size%':>7} {'Wicks':>10} {'Vol':>5} {'#':>3} {'Ago':>4}")
        print(f"  {'─'*76}")

        for s in bullish[:15]:
            if s.strength == 'PERFECT':
                col = C.Y + C.BOLD
                icon = "💎"
            elif s.strength == 'STRONG':
                col = C.G + C.BOLD
                icon = "🔥"
            else:
                col = C.G
                icon = "📈"

            ago_str = "NOW" if s.candle_ago == 0 else f"-{s.candle_ago}"
            wicks = f"{s.upper_wick_pct:.1f}/{s.lower_wick_pct:.1f}%"
            consec = f"x{s.consecutive}" if s.consecutive > 1 else ""

            print(f"  {s.symbol:<14} {col}{icon} {s.strength:<8}{C.END} "
                  f"{s.body_ratio:>5.0f}% {s.candle_size_pct:>6.2f}% {wicks:>10} "
                  f"{s.volume_ratio:>4.1f}x {consec:>3} {ago_str:>4}")
    else:
        print(f"  {C.DIM}Geen bullish marubozu gevonden{C.END}")

    # ═══ BEARISH MARUBOZU ═══
    print(f"\n{C.R}{C.BOLD}  🔴 BEARISH MARUBOZU ({len(bearish)}) - SHORT MOMENTUM{C.END}")
    print(f"  {C.DIM}Sterke verkopers - geen koopdruk - prijs zal waarschijnlijk dalen{C.END}")
    print(f"  {'─'*76}")

    if bearish:
        print(f"  {'Symbol':<14} {'Strength':<10} {'Body%':>6} {'Size%':>7} {'Wicks':>10} {'Vol':>5} {'#':>3} {'Ago':>4}")
        print(f"  {'─'*76}")

        for s in bearish[:15]:
            if s.strength == 'PERFECT':
                col = C.Y + C.BOLD
                icon = "💎"
            elif s.strength == 'STRONG':
                col = C.R + C.BOLD
                icon = "🔥"
            else:
                col = C.R
                icon = "📉"

            ago_str = "NOW" if s.candle_ago == 0 else f"-{s.candle_ago}"
            wicks = f"{s.upper_wick_pct:.1f}/{s.lower_wick_pct:.1f}%"
            consec = f"x{s.consecutive}" if s.consecutive > 1 else ""

            print(f"  {s.symbol:<14} {col}{icon} {s.strength:<8}{C.END} "
                  f"{s.body_ratio:>5.0f}% {s.candle_size_pct:>6.2f}% {wicks:>10} "
                  f"{s.volume_ratio:>4.1f}x {consec:>3} {ago_str:>4}")
    else:
        print(f"  {C.DIM}Geen bearish marubozu gevonden{C.END}")

    # ═══ BEST SIGNALS ═══
    perfect_bull = [s for s in bullish if s.strength == 'PERFECT' and s.candle_ago == 0]
    perfect_bear = [s for s in bearish if s.strength == 'PERFECT' and s.candle_ago == 0]

    multi_bull = [s for s in bullish if s.consecutive >= 2 and s.candle_ago == 0]
    multi_bear = [s for s in bearish if s.consecutive >= 2 and s.candle_ago == 0]

    if perfect_bull or perfect_bear or multi_bull or multi_bear:
        print(f"\n{C.Y}{C.BOLD}  ⚡ TOP SIGNALS{C.END}")
        print(f"  {'─'*76}")

        for s in (perfect_bull + multi_bull)[:3]:
            move_pct = ((s.close_price - s.open_price) / s.open_price) * 100
            print(f"  {C.G}🚀 LONG:{C.END} {C.BOLD}{s.symbol}{C.END} | {s.strength} | +{move_pct:.2f}% candle")
            if s.consecutive > 1:
                print(f"     {C.Y}⚡ {s.consecutive} opeenvolgende bullish marubozu's!{C.END}")
            print(f"     Entry: {s.close_price:.8f} | SL: {s.low_price:.8f}")

        for s in (perfect_bear + multi_bear)[:3]:
            move_pct = ((s.open_price - s.close_price) / s.open_price) * 100
            print(f"  {C.R}📉 SHORT:{C.END} {C.BOLD}{s.symbol}{C.END} | {s.strength} | -{move_pct:.2f}% candle")
            if s.consecutive > 1:
                print(f"     {C.Y}⚡ {s.consecutive} opeenvolgende bearish marubozu's!{C.END}")
            print(f"     Entry: {s.close_price:.8f} | SL: {s.high_price:.8f}")

    print(f"\n{'═'*80}")
    print(f"  {C.DIM}Marubozu = candle zonder wicks = sterke momentum{C.END}")
    print(f"  {C.DIM}Body%: body als % van candle (hoger = sterker){C.END}")
    print(f"  {C.DIM}Size%: candle grootte als % van prijs{C.END}")
    print(f"  {C.DIM}Wicks: upper/lower wick % (lager = puurder marubozu){C.END}")
    print(f"  {C.DIM}#: aantal opeenvolgende marubozu's in zelfde richting{C.END}")
    print(f"{'═'*80}\n")


async def run_scan():
    """Run een volledige scan."""
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        symbols = await get_symbols(session)
        if not symbols:
            print("Geen symbols gevonden!")
            return

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning {len(symbols)} coins voor Marubozu candles...")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [scan_symbol(session, sym, semaphore) for sym in symbols]
        results = await asyncio.gather(*tasks)

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
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 MARUBOZU CANDLE SCANNER                                      ║
║                                                                              ║
║  Zoekt naar Marubozu candles op de 1 MIN chart                               ║
║                                                                              ║
║  Marubozu = Sterke momentum candle ZONDER wicks                              ║
║                                                                              ║
║  🟢 BULLISH MARUBOZU:                                                        ║
║     • Open ≈ Low, Close ≈ High                                               ║
║     • Kopers in volledige controle                                           ║
║     • Signaal: Prijs zal waarschijnlijk STIJGEN                              ║
║                                                                              ║
║  🔴 BEARISH MARUBOZU:                                                        ║
║     • Open ≈ High, Close ≈ Low                                               ║
║     • Verkopers in volledige controle                                        ║
║     • Signaal: Prijs zal waarschijnlijk DALEN                                ║
║                                                                              ║
║  Strength Levels:                                                            ║
║  • 💎 PERFECT: 0-2% wicks, >1% candle size                                   ║
║  • 🔥 STRONG:  <5% wicks, >0.5% candle size                                  ║
║  • 📈 MODERATE: <10% wicks, >0.3% candle size                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    print(f"{C.Y}Settings:{C.END}")
    print(f"  • Timeframe: {TIMEFRAME}")
    print(f"  • Min Body Ratio: {MIN_BODY_RATIO * 100:.0f}%")
    print(f"  • Max Wick Ratio: {MAX_WICK_RATIO * 100:.0f}%")
    print(f"  • Min Candle Size: {MIN_CANDLE_SIZE_PCT}%")
    print(f"  • Min Volume 24h: ${MIN_VOLUME_24H:,}")
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
            await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
