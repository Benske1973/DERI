# marubozu_ws_5m.py - Real-time Marubozu Scanner via WebSocket (5 MIN)
# Bouwt 5min candles uit tick data en detecteert Marubozu's LIVE

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
MIN_CANDLE_SIZE_PCT = 0.3        # Min candle grootte (% van prijs) - hoger voor 5min
STRONG_SIZE_PCT = 0.6            # Sterke candle grootte
EXTREME_SIZE_PCT = 1.2           # Extreme candle grootte

# WebSocket Config
MAX_SUBS_PER_WS = 300
BATCH_SIZE = 50
WS_TIMEOUT = 120
MIN_VOLUME_24H = 50000

# Candle Building - 5 MINUTEN
CANDLE_INTERVAL = 300            # 5 minuten = 300 seconden

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
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    tick_count: int
    start_time: float

    def update(self, price: float, size: float = 0):
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
    type: str
    strength: str
    body_pct: float
    size_pct: float
    price: float
    candle_open: float
    candle_close: float
    candle_high: float
    candle_low: float
    timestamp: str
    timeframe: str = "5m"


# ═══════════════════════════════════════════════════════════════
#                      GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

live_candles: Dict[str, LiveCandle] = {}
candle_history: Dict[str, list] = defaultdict(list)

stats = {
    'total_ticks': 0,
    'marubozu_count': 0,
    'last_alert': None,
    'start_time': time.time(),
    'active_symbols': set(),
    'candles_completed': 0
}

recent_alerts: Dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════
#                      MARUBOZU DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_marubozu(candle: LiveCandle) -> Optional[MarubozuAlert]:
    """Detecteer Marubozu patroon"""

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

    if body_ratio < MIN_BODY_RATIO:
        return None
    if upper_wick_ratio > MAX_WICK_RATIO or lower_wick_ratio > MAX_WICK_RATIO:
        return None
    if candle_size_pct < MIN_CANDLE_SIZE_PCT:
        return None

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
        timestamp=datetime.now().strftime('%H:%M:%S'),
        timeframe="5m"
    )


def print_alert(alert: MarubozuAlert):
    """Print marubozu alert"""

    key = f"{alert.symbol}_{alert.type}_5m"
    now = time.time()
    if key in recent_alerts and now - recent_alerts[key] < 300:  # 5 min dedup
        return
    recent_alerts[key] = now

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
    print(f"  {str_icon} {str_col}MARUBOZU {alert.strength}{C.END} - {type_col}{icon} {alert.type}{C.END} [{C.C}5 MIN{C.END}]")
    print(f"  {C.BOLD}{alert.symbol}{C.END} @ {alert.timestamp}")
    print(f"  {'─'*66}")
    print(f"  Price: {alert.price:.8f}")
    print(f"  Candle: O:{alert.candle_open:.8f} H:{alert.candle_high:.8f} L:{alert.candle_low:.8f} C:{alert.candle_close:.8f}")
    print(f"  Body: {alert.body_pct:.0f}% | Size: {alert.size_pct:.2f}% | Move: {move_pct:+.2f}%")
    print(f"  {type_col}{C.BOLD}→ {direction} SIGNAL (5 MIN TIMEFRAME){C.END}")
    print(f"{'═'*70}\n")

    stats['marubozu_count'] += 1
    stats['last_alert'] = alert


# ═══════════════════════════════════════════════════════════════
#                      CANDLE BUILDING
# ═══════════════════════════════════════════════════════════════

def get_candle_start_time(timestamp: float) -> float:
    """Get start of 5-minute candle interval"""
    return (timestamp // CANDLE_INTERVAL) * CANDLE_INTERVAL


def process_tick(symbol: str, price: float, size: float = 0):
    """Process tick and build 5-min candles"""
    now = time.time()
    candle_start = get_candle_start_time(now)

    stats['total_ticks'] += 1
    stats['active_symbols'].add(symbol)

    if symbol in live_candles:
        candle = live_candles[symbol]

        if candle.start_time < candle_start:
            # 5-min candle completed
            stats['candles_completed'] += 1

            alert = detect_marubozu(candle)
            if alert:
                print_alert(alert)

            candle_history[symbol].append(candle)
            if len(candle_history[symbol]) > 5:
                candle_history[symbol].pop(0)

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
            candle.update(price, size)
    else:
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
#                      WEBSOCKET
# ═══════════════════════════════════════════════════════════════

def get_tradeable_symbols() -> list:
    """Get tradeable USDT pairs"""
    try:
        r = requests.get(f"{KUCOIN_API}/api/v1/symbols", timeout=10)
        symbols_data = r.json()['data']

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
        print(f"Error: {e}")
        return []


async def get_ws_token() -> str:
    r = requests.post(f"{KUCOIN_API}/api/v1/bullet-public", timeout=10)
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


async def handle_websocket(ws_id: int, symbols: list):
    """Handle WebSocket connection"""
    reconnect_delay = 5

    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Connecting ({len(symbols)} symbols)...")

            ws_url = await get_ws_token()

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:

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

                print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Connected")
                reconnect_delay = 5

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
    """Print status updates"""
    last_5min = -1

    while True:
        await asyncio.sleep(10)

        now = time.time()
        current_5min = int(now) // 300

        runtime = int(now - stats['start_time'])
        active = len(stats['active_symbols'])
        ticks = stats['total_ticks']
        marubozu = stats['marubozu_count']
        building = len(live_candles)
        completed = stats['candles_completed']

        # Time until next 5-min candle close
        next_close = ((current_5min + 1) * 300) - now
        mins = int(next_close // 60)
        secs = int(next_close % 60)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"📊 5MIN | Active: {active} | Ticks: {ticks:,} | Building: {building} | "
              f"Completed: {completed} | 🎯 Alerts: {marubozu} | "
              f"Next close: {mins}m{secs}s")

        # Force check at 5-min boundaries
        if current_5min != last_5min:
            last_5min = current_5min
            candle_start = get_candle_start_time(now)
            for symbol, candle in list(live_candles.items()):
                if candle.start_time < candle_start and candle.tick_count >= 5:
                    alert = detect_marubozu(candle)
                    if alert:
                        print_alert(alert)
                    del live_candles[symbol]


async def main():
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║           📊 REAL-TIME MARUBOZU SCANNER - 5 MINUTE (WebSocket)               ║
║                                                                              ║
║  Bouwt 5-minuut candles LIVE uit tick data                                   ║
║  Detecteert Marubozu patronen INSTANT bij candle close                       ║
║                                                                              ║
║  🟢 BULLISH MARUBOZU = Open≈Low, Close≈High = LONG                          ║
║  🔴 BEARISH MARUBOZU = Open≈High, Close≈Low = SHORT                         ║
║                                                                              ║
║  5 MIN = Sterker signaal, minder ruis dan 1 min                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching symbols...")
    symbols = get_tradeable_symbols()

    if not symbols:
        print("No symbols found!")
        return

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(symbols)} symbols")

    chunks = [symbols[i:i+MAX_SUBS_PER_WS] for i in range(0, len(symbols), MAX_SUBS_PER_WS)]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Using {len(chunks)} WebSocket connections")

    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(asyncio.create_task(handle_websocket(i, chunk)))
    tasks.append(asyncio.create_task(status_printer()))

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {C.G}Scanner running - watching for 5MIN Marubozu candles...{C.END}\n")

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
