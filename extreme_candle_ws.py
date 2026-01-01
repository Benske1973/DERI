# extreme_candle_ws.py - Real-time EXTREME Candle Detector (INTRABAR)
# Detecteert extreme price moves TERWIJL de candle zich vormt!

import asyncio
import websockets
import requests
import json
import time
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, List

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Extreme Move Detection (INTRABAR)
EXTREME_MOVE_PCT = 1.5           # 1.5% move = extreme (alert!)
STRONG_MOVE_PCT = 1.0            # 1.0% move = strong
MODERATE_MOVE_PCT = 0.7          # 0.7% move = moderate

# Alert Thresholds
ALERT_ON_MODERATE = False        # Alert op moderate moves?
ALERT_ON_STRONG = True           # Alert op strong moves?
ALERT_ON_EXTREME = True          # Alert op extreme moves?

# Speed Detection (hoe snel de move gebeurt)
SPEED_WINDOW_TICKS = 10          # Check speed over laatste 10 ticks
SPEED_THRESHOLD_PCT = 0.5        # 0.5% in 10 ticks = snel

# WebSocket Config
MAX_SUBS_PER_WS = 300
BATCH_SIZE = 50
WS_TIMEOUT = 120
MIN_VOLUME_24H = 50000

# Candle Tracking
CANDLE_INTERVAL = 60             # 1 minuut

# API
KUCOIN_API = "https://api.kucoin.com"

# Cooldown (voorkom spam)
ALERT_COOLDOWN = 30              # Seconden tussen alerts voor zelfde coin


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

    # Background colors
    BG_R = '\033[41m'
    BG_G = '\033[42m'
    BG_Y = '\033[43m'


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
    ticks: List[tuple]  # [(timestamp, price), ...]

    # Tracking extremes
    max_move_up: float = 0.0
    max_move_down: float = 0.0
    alerted_up: bool = False
    alerted_down: bool = False

    def update(self, price: float, size: float = 0):
        now = time.time()

        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price
        self.close = price
        self.volume += size
        self.tick_count += 1

        # Track ticks (keep last 50)
        self.ticks.append((now, price))
        if len(self.ticks) > 50:
            self.ticks.pop(0)

        # Calculate moves from open
        if self.open > 0:
            move_up = ((self.high - self.open) / self.open) * 100
            move_down = ((self.open - self.low) / self.open) * 100
            self.max_move_up = max(self.max_move_up, move_up)
            self.max_move_down = max(self.max_move_down, move_down)


@dataclass
class ExtremeAlert:
    symbol: str
    direction: str          # PUMP / DUMP
    strength: str           # EXTREME / STRONG / MODERATE
    move_pct: float         # Totale move %
    current_price: float
    candle_open: float
    candle_high: float
    candle_low: float
    speed: str              # FLASH / FAST / NORMAL
    tick_count: int
    seconds_in: int         # Seconden in candle
    timestamp: str


# ═══════════════════════════════════════════════════════════════
#                      GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

live_candles: Dict[str, LiveCandle] = {}
recent_alerts: Dict[str, float] = {}

stats = {
    'total_ticks': 0,
    'extreme_count': 0,
    'strong_count': 0,
    'start_time': time.time(),
    'active_symbols': set(),
    'biggest_pump': None,
    'biggest_dump': None
}


# ═══════════════════════════════════════════════════════════════
#                      DETECTION LOGIC
# ═══════════════════════════════════════════════════════════════

def calculate_speed(ticks: List[tuple]) -> tuple:
    """Bereken hoe snel de move gebeurt"""
    if len(ticks) < SPEED_WINDOW_TICKS:
        return "NORMAL", 0

    recent = ticks[-SPEED_WINDOW_TICKS:]
    first_price = recent[0][1]
    last_price = recent[-1][1]

    if first_price == 0:
        return "NORMAL", 0

    move_pct = abs((last_price - first_price) / first_price) * 100

    if move_pct >= SPEED_THRESHOLD_PCT * 2:
        return "FLASH", move_pct
    elif move_pct >= SPEED_THRESHOLD_PCT:
        return "FAST", move_pct
    else:
        return "NORMAL", move_pct


def check_extreme_move(candle: LiveCandle) -> Optional[ExtremeAlert]:
    """Check voor extreme intrabar move"""

    if candle.open == 0 or candle.tick_count < 3:
        return None

    now = time.time()
    seconds_in = int(now - candle.start_time)

    # Calculate current moves
    move_up = ((candle.high - candle.open) / candle.open) * 100
    move_down = ((candle.open - candle.low) / candle.open) * 100

    # Check cooldown
    cooldown_key_up = f"{candle.symbol}_PUMP"
    cooldown_key_down = f"{candle.symbol}_DUMP"

    alert = None

    # Check PUMP (upward move)
    if move_up >= MODERATE_MOVE_PCT and not candle.alerted_up:
        if cooldown_key_up not in recent_alerts or now - recent_alerts[cooldown_key_up] > ALERT_COOLDOWN:

            # Determine strength
            if move_up >= EXTREME_MOVE_PCT:
                strength = "EXTREME"
                should_alert = ALERT_ON_EXTREME
            elif move_up >= STRONG_MOVE_PCT:
                strength = "STRONG"
                should_alert = ALERT_ON_STRONG
            else:
                strength = "MODERATE"
                should_alert = ALERT_ON_MODERATE

            if should_alert:
                speed, speed_pct = calculate_speed(candle.ticks)

                alert = ExtremeAlert(
                    symbol=candle.symbol,
                    direction="PUMP",
                    strength=strength,
                    move_pct=move_up,
                    current_price=candle.close,
                    candle_open=candle.open,
                    candle_high=candle.high,
                    candle_low=candle.low,
                    speed=speed,
                    tick_count=candle.tick_count,
                    seconds_in=seconds_in,
                    timestamp=datetime.now().strftime('%H:%M:%S')
                )

                candle.alerted_up = True
                recent_alerts[cooldown_key_up] = now

    # Check DUMP (downward move)
    if move_down >= MODERATE_MOVE_PCT and not candle.alerted_down:
        if cooldown_key_down not in recent_alerts or now - recent_alerts[cooldown_key_down] > ALERT_COOLDOWN:

            if move_down >= EXTREME_MOVE_PCT:
                strength = "EXTREME"
                should_alert = ALERT_ON_EXTREME
            elif move_down >= STRONG_MOVE_PCT:
                strength = "STRONG"
                should_alert = ALERT_ON_STRONG
            else:
                strength = "MODERATE"
                should_alert = ALERT_ON_MODERATE

            if should_alert:
                speed, speed_pct = calculate_speed(candle.ticks)

                alert = ExtremeAlert(
                    symbol=candle.symbol,
                    direction="DUMP",
                    strength=strength,
                    move_pct=move_down,
                    current_price=candle.close,
                    candle_open=candle.open,
                    candle_high=candle.high,
                    candle_low=candle.low,
                    speed=speed,
                    tick_count=candle.tick_count,
                    seconds_in=seconds_in,
                    timestamp=datetime.now().strftime('%H:%M:%S')
                )

                candle.alerted_down = True
                recent_alerts[cooldown_key_down] = now

    return alert


def print_alert(alert: ExtremeAlert):
    """Print extreme move alert"""

    # Colors based on direction and strength
    if alert.direction == "PUMP":
        dir_col = C.G
        bg_col = C.BG_G if alert.strength == "EXTREME" else ""
        icon = "🚀"
        arrow = "↑↑↑"
    else:
        dir_col = C.R
        bg_col = C.BG_R if alert.strength == "EXTREME" else ""
        icon = "💥"
        arrow = "↓↓↓"

    if alert.strength == "EXTREME":
        str_icon = "⚠️ "
        str_col = C.Y + C.BOLD
    elif alert.strength == "STRONG":
        str_icon = "🔥"
        str_col = C.BOLD
    else:
        str_icon = "📊"
        str_col = ""

    # Speed indicator
    if alert.speed == "FLASH":
        speed_str = f"{C.Y}⚡FLASH{C.END}"
    elif alert.speed == "FAST":
        speed_str = f"{C.C}🏃FAST{C.END}"
    else:
        speed_str = ""

    print(f"\n{'═'*75}")
    print(f"  {str_icon} {bg_col}{str_col}EXTREME {alert.strength} {alert.direction}!{C.END} {speed_str}")
    print(f"  {dir_col}{C.BOLD}{icon} {alert.symbol} {arrow} {alert.move_pct:+.2f}%{C.END}")
    print(f"  {'─'*71}")
    print(f"  {C.BOLD}INTRABAR ALERT{C.END} @ {alert.timestamp} ({alert.seconds_in}s into candle)")
    print(f"  Price: {alert.current_price:.8f} | Open: {alert.candle_open:.8f}")
    print(f"  Range: L:{alert.candle_low:.8f} → H:{alert.candle_high:.8f}")
    print(f"  Ticks: {alert.tick_count} | Speed: {alert.speed}")

    if alert.direction == "PUMP":
        print(f"  {C.G}{C.BOLD}→ POTENTIËLE LONG - Wacht op pullback of breakout bevestiging{C.END}")
    else:
        print(f"  {C.R}{C.BOLD}→ POTENTIËLE SHORT - Wacht op bounce of verdere breakdown{C.END}")

    print(f"{'═'*75}\n")

    # Update stats
    if alert.strength == "EXTREME":
        stats['extreme_count'] += 1
    else:
        stats['strong_count'] += 1

    if alert.direction == "PUMP":
        if stats['biggest_pump'] is None or alert.move_pct > stats['biggest_pump'][1]:
            stats['biggest_pump'] = (alert.symbol, alert.move_pct)
    else:
        if stats['biggest_dump'] is None or alert.move_pct > stats['biggest_dump'][1]:
            stats['biggest_dump'] = (alert.symbol, alert.move_pct)


# ═══════════════════════════════════════════════════════════════
#                      TICK PROCESSING
# ═══════════════════════════════════════════════════════════════

def get_candle_start_time(timestamp: float) -> float:
    return (timestamp // CANDLE_INTERVAL) * CANDLE_INTERVAL


def process_tick(symbol: str, price: float, size: float = 0):
    """Process tick en check voor extreme moves"""
    now = time.time()
    candle_start = get_candle_start_time(now)

    stats['total_ticks'] += 1
    stats['active_symbols'].add(symbol)

    if symbol in live_candles:
        candle = live_candles[symbol]

        if candle.start_time < candle_start:
            # New candle - reset
            live_candles[symbol] = LiveCandle(
                symbol=symbol,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=size,
                tick_count=1,
                start_time=candle_start,
                ticks=[(now, price)]
            )
        else:
            # Update existing candle
            candle.update(price, size)

            # CHECK FOR EXTREME MOVE (INTRABAR!)
            alert = check_extreme_move(candle)
            if alert:
                print_alert(alert)
    else:
        # First tick
        live_candles[symbol] = LiveCandle(
            symbol=symbol,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=size,
            tick_count=1,
            start_time=candle_start,
            ticks=[(now, price)]
        )


# ═══════════════════════════════════════════════════════════════
#                      WEBSOCKET
# ═══════════════════════════════════════════════════════════════

def get_tradeable_symbols() -> list:
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

                print(f"[{datetime.now().strftime('%H:%M:%S')}] WS-{ws_id}: Connected & subscribed")
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

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


async def status_printer():
    """Print status"""
    while True:
        await asyncio.sleep(15)

        now = time.time()
        runtime = int(now - stats['start_time'])
        active = len(stats['active_symbols'])
        ticks = stats['total_ticks']
        extreme = stats['extreme_count']
        strong = stats['strong_count']

        # Find current biggest movers
        current_movers = []
        for sym, candle in live_candles.items():
            if candle.tick_count >= 3:
                move = max(candle.max_move_up, candle.max_move_down)
                direction = "↑" if candle.max_move_up > candle.max_move_down else "↓"
                if move >= 0.5:
                    current_movers.append((sym, move, direction))

        current_movers.sort(key=lambda x: x[1], reverse=True)
        top_movers = current_movers[:3]

        mover_str = " | ".join([f"{s}{d}{m:.1f}%" for s, m, d in top_movers]) if top_movers else "—"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"⚡ Active: {active} | Ticks: {ticks:,} | "
              f"🔥 Extreme: {extreme} | Strong: {strong} | "
              f"Top: {mover_str}")


async def main():
    print(f"""
{C.R}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║           ⚡ EXTREME CANDLE DETECTOR - INTRABAR (WebSocket)                  ║
║                                                                              ║
║  Detecteert extreme price moves TERWIJL de candle zich vormt!                ║
║  Geen wachten op candle close - INSTANT alerts bij grote bewegingen          ║
║                                                                              ║
║  🚀 PUMP: Grote upward move (potential long entry)                           ║
║  💥 DUMP: Grote downward move (potential short/buy the dip)                  ║
║                                                                              ║
║  Thresholds:                                                                 ║
║  • ⚠️  EXTREME: {EXTREME_MOVE_PCT}%+ move in 1 candle                                       ║
║  • 🔥 STRONG:  {STRONG_MOVE_PCT}%+ move in 1 candle                                        ║
║  • 📊 MODERATE: {MODERATE_MOVE_PCT}%+ move in 1 candle                                      ║
║                                                                              ║
║  Speed Detection:                                                            ║
║  • ⚡FLASH: Extreme snelle move (manipulatie/news?)                          ║
║  • 🏃FAST:  Snelle move                                                      ║
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

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {C.Y}⚡ INTRABAR scanner running - watching for extreme moves...{C.END}\n")

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
