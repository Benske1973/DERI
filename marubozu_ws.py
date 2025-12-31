# marubozu_ws.py - Real-time Marubozu Scanner via WebSocket
# Bouwt 1min candles uit tick data en detecteert Marubozu's LIVE

import asyncio
import websockets
import requests
import json
import time
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Marubozu Detection
MIN_BODY_RATIO = 0.85            # Body moet 85%+ van candle zijn
MAX_WICK_RATIO = 0.10            # Max 10% wick
MIN_CANDLE_SIZE_PCT = 0.25       # Min candle grootte (% van prijs)
STRONG_SIZE_PCT = 0.5            # Sterke candle grootte
EXTREME_SIZE_PCT = 1.0           # Extreme candle grootte

# WebSocket Config
MAX_SUBS_PER_WS = 300            # KuCoin limit
BATCH_SIZE = 50                  # Symbols per subscribe
WS_TIMEOUT = 120                 # Timeout in seconden
MIN_VOLUME_24H = 50000           # Min 24h volume filter

# Candle Building
CANDLE_INTERVAL = 60             # 1 minuut candles

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
#                      DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiveCandle:
    """Real-time candle gebouwd uit ticks"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    start_time: float

    def update(self, price: float, size: float = 0):
        """Update candle met nieuwe tick"""
        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += size
        self.tick_count += 1


@dataclass
class MarubozuAlert:
    symbol: str
    type: str           # BULLISH / BEARISH
    strength: str       # PERFECT / STRONG / MODERATE
    body_pct: float
    size_pct: float
    price: float
    candle_open: float
    candle_close: float
    candle_high: float
    candle_low: float
    timestamp: str


# ═══════════════════════════════════════════════════════════════
#                      GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

# Live candles per symbol
live_candles: Dict[str, LiveCandle] = {}

# Completed candles history (laatste 5 per symbol)
candle_history: Dict[str, list] = defaultdict(list)

# Stats
stats = {
    'total_ticks': 0,
    'marubozu_count': 0,
    'last_alert': None,
    'start_time': time.time(),
    'active_symbols': set()
}

# Recent alerts (voor deduplicatie)
recent_alerts: Dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════
#                      MARUBOZU DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_marubozu(candle: LiveCandle) -> Optional[MarubozuAlert]:
    """Detecteer Marubozu patroon in completed candle"""

    total_range = candle.high - candle.low
    if total_range == 0 or candle.close == 0:
        return None

    body = abs(candle.close - candle.open)
    is_bullish = candle.close > candle.open

    if is_bullish:
        upper_wick = candle.high - candle.close
        lower_wick = candle.open - candle.low
    else:
        upper_wick = candle.high - candle.open
        lower_wick = candle.close - candle.low

    body_ratio = body / total_range
    upper_wick_ratio = upper_wick / total_range
    lower_wick_ratio = lower_wick / total_range
    candle_size_pct = (total_range / candle.close) * 100

    # Check marubozu criteria
    if body_ratio < MIN_BODY_RATIO:
        return None
    if upper_wick_ratio > MAX_WICK_RATIO or lower_wick_ratio > MAX_WICK_RATIO:
        return None
    if candle_size_pct < MIN_CANDLE_SIZE_PCT:
        return None

    # Determine strength
    total_wick = upper_wick_ratio + lower_wick_ratio

    if total_wick < 0.02 and candle_size_pct >= EXTREME_SIZE_PCT:
        strength = "PERFECT"
    elif total_wick < 0.05 and candle_size_pct >= STRONG_SIZE_PCT:
        strength = "STRONG"
    else:
        strength = "MODERATE"

    return MarubozuAlert(
        symbol=candle.symbol,
        type="BULLISH" if is_bullish else "BEARISH",
        strength=strength,
        body_pct=body_ratio * 100,
        size_pct=candle_size_pct,
        price=candle.close,
        candle_open=candle.open,
        candle_close=candle.close,
        candle_high=candle.high,
        candle_low=candle.low,
        timestamp=datetime.now().strftime('%H:%M:%S')
    )


def print_alert(alert: MarubozuAlert):
    """Print marubozu alert"""

    # Deduplicatie check
    key = f"{alert.symbol}_{alert.type}"
    now = time.time()
    if key in recent_alerts and now - recent_alerts[key] < 60:
        return  # Skip duplicate within 60 seconds
    recent_alerts[key] = now

    # Color based on type and strength
    if alert.type == "BULLISH":
        type_col = C.G
        icon = "🟢"
        direction = "LONG"
    else:
        type_col = C.R
        icon = "🔴"
        direction = "SHORT"

    if alert.strength == "PERFECT":
        str_col = C.Y + C.BOLD
        str_icon = "💎"
    elif alert.strength == "STRONG":
        str_col = C.BOLD
        str_icon = "🔥"
    else:
        str_col = ""
        str_icon = "📊"

    move_pct = ((alert.candle_close - alert.candle_open) / alert.candle_open) * 100

    print(f"\n{'═'*70}")
    print(f"  {str_icon} {str_col}MARUBOZU {alert.strength}{C.END} - {type_col}{icon} {alert.type}{C.END}")
    print(f"  {C.BOLD}{alert.symbol}{C.END} @ {alert.timestamp}")
    print(f"  {'─'*66}")
    print(f"  Price: {alert.price:.8f}")
    print(f"  Candle: O:{alert.candle_open:.8f} H:{alert.candle_high:.8f} L:{alert.candle_low:.8f} C:{alert.candle_close:.8f}")
    print(f"  Body: {alert.body_pct:.0f}% | Size: {alert.size_pct:.2f}% | Move: {move_pct:+.2f}%")
    print(f"  {type_col}{C.BOLD}→ {direction} SIGNAL{C.END}")
    print(f"{'═'*70}\n")

    stats['marubozu_count'] += 1
    stats['last_alert'] = alert


# ═══════════════════════════════════════════════════════════════
#                      CANDLE BUILDING
# ═══════════════════════════════════════════════════════════════

def get_candle_start_time(timestamp: float) -> float:
    """Get start time of current candle interval"""
    return (timestamp // CANDLE_INTERVAL) * CANDLE_INTERVAL


def process_tick(symbol: str, price: float, size: float = 0):
    """Process incoming tick and build candles"""
    now = time.time()
    candle_start = get_candle_start_time(now)

    stats['total_ticks'] += 1
    stats['active_symbols'].add(symbol)

    if symbol in live_candles:
        candle = live_candles[symbol]

        # Check if candle period ended
        if candle.start_time < candle_start:
            # Candle completed - check for marubozu
            alert = detect_marubozu(candle)
            if alert:
                print_alert(alert)

            # Save to history
            candle_history[symbol].append(candle)
            if len(candle_history[symbol]) > 5:
                candle_history[symbol].pop(0)

            # Start new candle
            live_candles[symbol] = LiveCandle(
                symbol=symbol,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
                tick_count=1,
                start_time=candle_start
            )
        else:
            # Update existing candle
            candle.update(price, size)
    else:
        # First tick for this symbol
        live_candles[symbol] = LiveCandle(
            symbol=symbol,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            tick_count=1,
            start_time=candle_start
        )


# ═══════════════════════════════════════════════════════════════
#                      WEBSOCKET FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_tradeable_symbols() -> list:
    """Get all tradeable USDT pairs with volume filter"""
    try:
        # Get symbols
        r = requests.get(f"{KUCOIN_API}/api/v1/symbols", timeout=10)
        symbols_data = r.json()['data']

        # Get tickers for volume
        r2 = requests.get(f"{KUCOIN_API}/api/v1/market/allTickers", timeout=10)
        tickers = {t['symbol']: float(t.get('volValue', 0))
                   for t in r2.json()['data']['ticker']}

        pairs = []
        for m in symbols_data:
            sym = m['symbol']
            if (m['quoteCurrency'] == 'USDT' and
                m['enableTrading'] and
                not any(x in sym for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-']) and
                tickers.get(sym, 0) >= MIN_VOLUME_24H):
                pairs.append(sym)

        return pairs
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


async def get_ws_token() -> str:
    """Get WebSocket token"""
    r = requests.post(f"{KUCOIN_API}/api/v1/bullet-public", timeout=10)
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


async def handle_websocket(ws_id: int, symbols: list):
    """Handle single WebSocket connection"""
    reconnect_delay = 5

    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Connecting for {len(symbols)} symbols...")

            ws_url = await get_ws_token()

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:

                # Subscribe in batches
                for i in range(0, len(symbols), BATCH_SIZE):
                    batch = symbols[i:i+BATCH_SIZE]
                    msg = {
                        "id": f"{ws_id}_{int(time.time()*1000)}_{i}",
                        "type": "subscribe",
                        "topic": "/market/ticker:" + ",".join(batch),
                        "privateChannel": False,
                        "response": True
                    }
                    await ws.send(json.dumps(msg))
                    await asyncio.sleep(0.3)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Subscribed to {len(symbols)} symbols")
                reconnect_delay = 5  # Reset on successful connect

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=WS_TIMEOUT)
                        data = json.loads(msg)

                        if data.get("type") == "message" and "data" in data:
                            ticker = data['data']
                            topic = data.get('topic', '')

                            if ':' in topic:
                                symbol = topic.split(':')[-1]
                                if 'price' in ticker:
                                    price = float(ticker['price'])
                                    size = float(ticker.get('size', 0))
                                    process_tick(symbol, price, size)

                    except asyncio.TimeoutError:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Timeout")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Disconnected")
                        break

        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id} error: {e}")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Reconnecting in {reconnect_delay}s...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


async def status_printer():
    """Print periodic status updates"""
    last_minute = -1

    while True:
        await asyncio.sleep(5)

        now = time.time()
        current_minute = int(now) // 60

        # Status update every 30 seconds
        runtime = int(now - stats['start_time'])
        active = len(stats['active_symbols'])
        ticks = stats['total_ticks']
        marubozu = stats['marubozu_count']

        # Calculate candles being built
        building = len(live_candles)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"📊 Active: {active} | Ticks: {ticks:,} | Building: {building} candles | "
              f"🎯 Marubozu alerts: {marubozu} | Runtime: {runtime}s")

        # Check for candle closes at minute boundary
        if current_minute != last_minute:
            last_minute = current_minute
            # Force check all candles for completion
            candle_start = get_candle_start_time(now)
            for symbol, candle in list(live_candles.items()):
                if candle.start_time < candle_start and candle.tick_count >= 2:
                    alert = detect_marubozu(candle)
                    if alert:
                        print_alert(alert)
                    # Reset candle
                    del live_candles[symbol]


async def main():
    """Main entry point"""
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║           📊 REAL-TIME MARUBOZU SCANNER (WebSocket)                          ║
║                                                                              ║
║  Bouwt 1-minuut candles LIVE uit tick data                                   ║
║  Detecteert Marubozu patronen INSTANT bij candle close                       ║
║                                                                              ║
║  🟢 BULLISH MARUBOZU = Open≈Low, Close≈High = LONG                          ║
║  🔴 BEARISH MARUBOZU = Open≈High, Close≈Low = SHORT                         ║
║                                                                              ║
║  💎 PERFECT: <2% wicks, >1% size                                            ║
║  🔥 STRONG:  <5% wicks, >0.5% size                                          ║
║  📊 MODERATE: <10% wicks, >0.25% size                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    # Get symbols
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching tradeable symbols...")
    symbols = get_tradeable_symbols()

    if not symbols:
        print("No symbols found!")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(symbols)} symbols with >${MIN_VOLUME_24H:,} volume")

    # Split across WebSocket connections
    chunks = [symbols[i:i+MAX_SUBS_PER_WS] for i in range(0, len(symbols), MAX_SUBS_PER_WS)]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {len(chunks)} WebSocket connections")

    # Create tasks
    tasks = []

    # WebSocket handlers
    for i, chunk in enumerate(chunks):
        tasks.append(asyncio.create_task(handle_websocket(i, chunk)))

    # Status printer
    tasks.append(asyncio.create_task(status_printer()))

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {C.G}Scanner running - watching for Marubozu candles...{C.END}\n")

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stopped")
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
