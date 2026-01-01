# fvg_scanner.py - FVG Order Blocks Multi-Coin Scanner
# Scant alle KuCoin pairs en geeft BUY/SELL signalen
# Gebaseerd op BigBeluga's FVG Order Blocks indicator

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

KUCOIN_API = "https://api.kucoin.com"

# Scanner Settings
TIMEFRAME = "15min"           # Timeframe voor scanning
MIN_VOLUME_USDT = 100000      # Minimum 24h volume in USDT
FILTER_PCT = 0.5              # Minimum gap grootte %
PROXIMITY_PCT = 1.0           # Hoe dicht bij zone voor signaal (%)
MAX_WORKERS = 10              # Parallel API calls

# Signal Types
SIGNAL_BUY = "🟢 BUY"
SIGNAL_SELL = "🔴 SELL"
SIGNAL_NEAR_BUY = "🟡 NEAR BUY"
SIGNAL_NEAR_SELL = "🟠 NEAR SELL"


# ═══════════════════════════════════════════════════════════════
#                      DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    start_idx: int
    top: float
    bottom: float
    is_bull: bool
    gap_pct: float
    broken: bool = False
    break_idx: Optional[int] = None


@dataclass
class Signal:
    symbol: str
    signal_type: str
    current_price: float
    zone_top: float
    zone_bottom: float
    distance_pct: float
    gap_pct: float
    volume_24h: float
    timestamp: datetime


# ═══════════════════════════════════════════════════════════════
#                      KUCOIN API
# ═══════════════════════════════════════════════════════════════

def get_all_symbols() -> List[dict]:
    """Haal alle USDT trading pairs op met volume data"""
    try:
        # Get symbols
        r = requests.get(f"{KUCOIN_API}/api/v1/symbols", timeout=10)
        symbols_data = r.json().get('data', [])

        # Get 24h stats for volume
        r2 = requests.get(f"{KUCOIN_API}/api/v1/market/allTickers", timeout=10)
        tickers = {t['symbol']: t for t in r2.json().get('data', {}).get('ticker', [])}

        result = []
        for m in symbols_data:
            symbol = m['symbol']
            if (m['quoteCurrency'] == 'USDT' and
                m['enableTrading'] and
                not any(x in symbol for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-'])):

                ticker = tickers.get(symbol, {})
                vol_value = float(ticker.get('volValue', 0) or 0)

                if vol_value >= MIN_VOLUME_USDT:
                    result.append({
                        'symbol': symbol,
                        'volume_24h': vol_value,
                        'last_price': float(ticker.get('last', 0) or 0)
                    })

        return sorted(result, key=lambda x: x['volume_24h'], reverse=True)
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


def get_candles(symbol: str, timeframe: str = "15min", limit: int = 200) -> pd.DataFrame:
    """Haal candle data op van KuCoin"""
    try:
        url = f"{KUCOIN_API}/api/v1/market/candles?symbol={symbol}&type={timeframe}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if data.get('code') != '200000' or not data.get('data'):
            return pd.DataFrame()

        df = pd.DataFrame(data['data'], columns=['ts', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df = df.astype({'ts': int, 'open': float, 'close': float, 'high': float,
                        'low': float, 'volume': float, 'turnover': float})

        df = df.iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['ts'], unit='s')

        return df.tail(limit)
    except Exception as e:
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#                      FVG DETECTION (BigBeluga Logic)
# ═══════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    return atr


def detect_fvg_orderblocks(df: pd.DataFrame, filter_pct: float = 0.5) -> List[OrderBlock]:
    """
    Detecteer FVG Order Blocks - BigBeluga's exacte logica
    """
    blocks = []

    if len(df) < 3:
        return blocks

    df = df.reset_index(drop=True)
    atr = calculate_atr(df, 200)

    for i in range(2, len(df)):
        high_2 = df.loc[i-2, 'high']
        low_2 = df.loc[i-2, 'low']
        high_1 = df.loc[i-1, 'high']
        low_1 = df.loc[i-1, 'low']
        high_0 = df.loc[i, 'high']
        low_0 = df.loc[i, 'low']
        current_atr = atr.iloc[i] if i < len(atr) else atr.iloc[-1]

        # BULLISH FVG
        fvg_up = (high_2 < low_0 and high_2 < high_1 and low_2 < low_0)
        if fvg_up:
            gap_pct = (low_0 - high_2) / high_2 * 100 if high_2 > 0 else 0
            if gap_pct > filter_pct:
                ob_top = high_2
                ob_btm = max(high_2 - current_atr * 0.3, low_2)
                blocks.append(OrderBlock(
                    start_idx=i, top=ob_top, bottom=ob_btm,
                    is_bull=True, gap_pct=gap_pct
                ))

        # BEARISH FVG
        fvg_dn = (low_2 > high_0 and low_2 > low_1 and high_2 > high_0)
        if fvg_dn:
            gap_pct = (low_2 - high_0) / low_2 * 100 if low_2 > 0 else 0
            if gap_pct > filter_pct:
                ob_top = min(low_2 + current_atr * 0.3, high_2)
                ob_btm = low_2
                blocks.append(OrderBlock(
                    start_idx=i, top=ob_top, bottom=ob_btm,
                    is_bull=False, gap_pct=gap_pct
                ))

    # Mitigation check
    for block in blocks:
        for i in range(block.start_idx + 1, len(df)):
            if block.is_bull:
                candle_low = df.loc[i, 'low']
                if candle_low <= block.top:
                    if candle_low <= block.bottom:
                        block.broken = True
                        block.break_idx = i
                        break
                    else:
                        block.top = candle_low
            else:
                candle_high = df.loc[i, 'high']
                if candle_high >= block.bottom:
                    if candle_high >= block.top:
                        block.broken = True
                        block.break_idx = i
                        break
                    else:
                        block.bottom = candle_high

    # Return only unbroken blocks
    return [b for b in blocks if not b.broken]


# ═══════════════════════════════════════════════════════════════
#                      SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════

def analyze_coin(coin_data: dict, timeframe: str, filter_pct: float) -> Optional[Signal]:
    """Analyseer een coin voor FVG signalen"""
    symbol = coin_data['symbol']

    try:
        df = get_candles(symbol, timeframe)
        if df.empty or len(df) < 50:
            return None

        blocks = detect_fvg_orderblocks(df, filter_pct)
        if not blocks:
            return None

        current_price = df.iloc[-1]['close']

        # Check proximity to each block
        best_signal = None
        best_distance = float('inf')

        for block in blocks:
            zone_mid = (block.top + block.bottom) / 2
            zone_height = block.top - block.bottom

            if block.is_bull:
                # BULLISH: Check if price is near or in the zone (potential BUY)
                if current_price <= block.top:
                    # Price is IN the zone
                    distance_pct = 0
                    signal_type = SIGNAL_BUY
                elif current_price > block.top:
                    # Price is above zone - check distance
                    distance_pct = ((current_price - block.top) / block.top) * 100
                    if distance_pct <= PROXIMITY_PCT:
                        signal_type = SIGNAL_NEAR_BUY
                    else:
                        continue
                else:
                    continue

                if distance_pct < best_distance:
                    best_distance = distance_pct
                    best_signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        current_price=current_price,
                        zone_top=block.top,
                        zone_bottom=block.bottom,
                        distance_pct=distance_pct,
                        gap_pct=block.gap_pct,
                        volume_24h=coin_data['volume_24h'],
                        timestamp=datetime.now()
                    )

            else:
                # BEARISH: Check if price is near or in the zone (potential SELL)
                if current_price >= block.bottom:
                    # Price is IN the zone
                    distance_pct = 0
                    signal_type = SIGNAL_SELL
                elif current_price < block.bottom:
                    # Price is below zone - check distance
                    distance_pct = ((block.bottom - current_price) / current_price) * 100
                    if distance_pct <= PROXIMITY_PCT:
                        signal_type = SIGNAL_NEAR_SELL
                    else:
                        continue
                else:
                    continue

                if distance_pct < best_distance:
                    best_distance = distance_pct
                    best_signal = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        current_price=current_price,
                        zone_top=block.top,
                        zone_bottom=block.bottom,
                        distance_pct=distance_pct,
                        gap_pct=block.gap_pct,
                        volume_24h=coin_data['volume_24h'],
                        timestamp=datetime.now()
                    )

        return best_signal

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════
#                      SCANNER
# ═══════════════════════════════════════════════════════════════

def scan_all_coins(timeframe: str = "15min", filter_pct: float = 0.5) -> List[Signal]:
    """Scan alle coins voor FVG signalen"""
    print(f"\n⏳ Fetching coin list...")
    coins = get_all_symbols()
    print(f"📊 Found {len(coins)} coins with volume > ${MIN_VOLUME_USDT:,.0f}")

    signals = []
    scanned = 0

    print(f"🔍 Scanning for FVG Order Blocks on {timeframe}...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_coin, coin, timeframe, filter_pct): coin
                   for coin in coins}

        for future in as_completed(futures):
            scanned += 1
            if scanned % 50 == 0:
                print(f"   Scanned {scanned}/{len(coins)} coins...")

            result = future.result()
            if result:
                signals.append(result)

    # Sort signals: BUY first, then by distance
    def signal_priority(s):
        priority = {SIGNAL_BUY: 0, SIGNAL_NEAR_BUY: 1, SIGNAL_SELL: 2, SIGNAL_NEAR_SELL: 3}
        return (priority.get(s.signal_type, 99), s.distance_pct)

    return sorted(signals, key=signal_priority)


def print_signals(signals: List[Signal]):
    """Print signalen in mooi formaat"""
    if not signals:
        print("\n❌ Geen FVG signalen gevonden")
        return

    buy_signals = [s for s in signals if 'BUY' in s.signal_type]
    sell_signals = [s for s in signals if 'SELL' in s.signal_type]

    print("\n" + "═" * 100)
    print(f"   🎯 FVG ORDER BLOCKS SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 100)

    if buy_signals:
        print(f"\n{'─' * 100}")
        print(f"   🟢 BUY SIGNALS ({len(buy_signals)} coins approaching bullish Order Blocks)")
        print(f"{'─' * 100}")
        print(f"   {'Symbol':<15} {'Signal':<15} {'Price':<12} {'Zone':<25} {'Distance':<10} {'Gap %':<10} {'Volume 24h':<15}")
        print(f"   {'-'*15} {'-'*15} {'-'*12} {'-'*25} {'-'*10} {'-'*10} {'-'*15}")

        for s in buy_signals:
            zone_str = f"{s.zone_bottom:.6f} - {s.zone_top:.6f}"
            dist_str = f"{s.distance_pct:.2f}%" if s.distance_pct > 0 else "IN ZONE"
            print(f"   {s.symbol:<15} {s.signal_type:<15} {s.current_price:<12.6f} {zone_str:<25} {dist_str:<10} {s.gap_pct:.2f}%     ${s.volume_24h:,.0f}")

    if sell_signals:
        print(f"\n{'─' * 100}")
        print(f"   🔴 SELL SIGNALS ({len(sell_signals)} coins approaching bearish Order Blocks)")
        print(f"{'─' * 100}")
        print(f"   {'Symbol':<15} {'Signal':<15} {'Price':<12} {'Zone':<25} {'Distance':<10} {'Gap %':<10} {'Volume 24h':<15}")
        print(f"   {'-'*15} {'-'*15} {'-'*12} {'-'*25} {'-'*10} {'-'*10} {'-'*15}")

        for s in sell_signals:
            zone_str = f"{s.zone_bottom:.6f} - {s.zone_top:.6f}"
            dist_str = f"{s.distance_pct:.2f}%" if s.distance_pct > 0 else "IN ZONE"
            print(f"   {s.symbol:<15} {s.signal_type:<15} {s.current_price:<12.6f} {zone_str:<25} {dist_str:<10} {s.gap_pct:.2f}%     ${s.volume_24h:,.0f}")

    print("\n" + "═" * 100)
    print(f"   💡 Tip: BUY bij bullish zone (support) | SELL bij bearish zone (resistance)")
    print("═" * 100 + "\n")


def export_signals_json(signals: List[Signal], filename: str = "fvg_signals.json"):
    """Export signalen naar JSON"""
    data = {
        'timestamp': datetime.now().isoformat(),
        'timeframe': TIMEFRAME,
        'total_signals': len(signals),
        'buy_signals': [],
        'sell_signals': []
    }

    for s in signals:
        signal_dict = {
            'symbol': s.symbol,
            'signal_type': s.signal_type,
            'current_price': s.current_price,
            'zone_top': s.zone_top,
            'zone_bottom': s.zone_bottom,
            'distance_pct': s.distance_pct,
            'gap_pct': s.gap_pct,
            'volume_24h': s.volume_24h
        }

        if 'BUY' in s.signal_type:
            data['buy_signals'].append(signal_dict)
        else:
            data['sell_signals'].append(signal_dict)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"📁 Signals exported to {filename}")


# ═══════════════════════════════════════════════════════════════
#                      MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🎯 FVG ORDER BLOCKS SCANNER [BigBeluga Style]                      ║
║                                                                              ║
║  Scant alle KuCoin USDT pairs voor FVG Order Block signalen                  ║
║  • BUY signaal: prijs bij bullish Order Block (support)                      ║
║  • SELL signaal: prijs bij bearish Order Block (resistance)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    while True:
        try:
            # Run scan
            signals = scan_all_coins(TIMEFRAME, FILTER_PCT)

            # Print results
            print_signals(signals)

            # Export to JSON
            export_signals_json(signals)

            # Wait for next scan
            print(f"⏰ Volgende scan over 5 minuten... (Ctrl+C om te stoppen)\n")
            time.sleep(300)  # 5 minutes

        except KeyboardInterrupt:
            print("\n\n👋 Scanner gestopt!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Retry in 60 seconden...")
            time.sleep(60)


if __name__ == '__main__':
    main()
