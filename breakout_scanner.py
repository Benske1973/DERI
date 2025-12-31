# breakout_scanner.py - Detecteert squeeze/consolidatie en breakouts in real-time
import asyncio
import websockets
import requests
import json
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

# ============= CONFIGURATIE =============
# Squeeze detectie
LOOKBACK_CANDLES = 20        # Aantal kaarsen voor squeeze detectie
SQUEEZE_THRESHOLD = 0.3      # Max avg body size % voor squeeze (0.3% = kleine kaarsen)
MIN_SQUEEZE_CANDLES = 5      # Minimum aantal squeeze kaarsen voor valid setup

# Breakout detectie
BREAKOUT_MULTIPLIER = 3.0    # Breakout = move > X keer gemiddelde range
VOLUME_SPIKE_MULT = 2.0      # Volume spike multiplier
MIN_BREAKOUT_PERCENT = 1.0   # Minimum % move voor breakout alert

# Filtering
MIN_VOLUME_USDT = 100000     # Minimum 24h volume
MAX_SIGNALS = 20             # Max aantal actieve setups tonen
CANDLE_TIMEFRAME = "1min"    # Timeframe voor candle data

# WebSocket
MAX_SUBSCRIPTIONS = 300
BATCH_SIZE = 50
UPDATE_INTERVAL = 2

# ============= KLEUREN =============
class C:
    G = '\033[92m'   # Green
    R = '\033[91m'   # Red
    Y = '\033[93m'   # Yellow
    B = '\033[94m'   # Blue
    M = '\033[95m'   # Magenta
    C = '\033[96m'   # Cyan
    W = '\033[97m'   # White
    BOLD = '\033[1m'
    END = '\033[0m'


@dataclass
class CoinState:
    symbol: str
    prices: deque           # (timestamp, price) tuples
    candles: list          # Historical candles [open, high, low, close, volume]
    avg_range: float = 0    # Gemiddelde candle range
    avg_body: float = 0     # Gemiddelde body size
    in_squeeze: bool = False
    squeeze_candles: int = 0
    last_breakout: float = 0
    breakout_direction: str = ""


def timestamp() -> str:
    return datetime.now().strftime('%H:%M:%S')


def get_high_volume_symbols() -> list:
    """Haal high volume symbols op."""
    print(f"[{timestamp()}] Ophalen symbols met volume > ${MIN_VOLUME_USDT:,}...")

    try:
        url = "https://api.kucoin.com/api/v1/market/allTickers"
        r = requests.get(url, timeout=10)
        data = r.json()

        symbols = []
        for t in data['data'].get('ticker', []):
            sym = t.get('symbol', '')
            if not sym.endswith('-USDT'):
                continue
            if any(x in sym for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-']):
                continue

            try:
                vol = float(t.get('volValue', 0))
                if vol >= MIN_VOLUME_USDT:
                    symbols.append({'symbol': sym, 'volume': vol})
            except:
                continue

        symbols.sort(key=lambda x: x['volume'], reverse=True)
        print(f"[{timestamp()}] Gevonden: {len(symbols)} symbols")
        return [s['symbol'] for s in symbols]

    except Exception as e:
        print(f"Error: {e}")
        return []


def fetch_candles(symbol: str) -> list:
    """Haal historische candles op voor squeeze analyse."""
    try:
        url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type={CANDLE_TIMEFRAME}"
        r = requests.get(url, timeout=5)
        data = r.json()

        if data.get('code') != '200000' or not data.get('data'):
            return []

        # KuCoin returns: [timestamp, open, close, high, low, volume, turnover]
        candles = []
        for c in data['data'][:LOOKBACK_CANDLES]:
            candles.append({
                'open': float(c[1]),
                'close': float(c[2]),
                'high': float(c[3]),
                'low': float(c[4]),
                'volume': float(c[5])
            })

        return list(reversed(candles))  # Oldest first

    except:
        return []


def analyze_squeeze(candles: list) -> tuple:
    """
    Analyseer of coin in squeeze zit.
    Returns: (in_squeeze, squeeze_candles, avg_range, avg_body)
    """
    if len(candles) < LOOKBACK_CANDLES:
        return False, 0, 0, 0

    ranges = []
    bodies = []

    for c in candles:
        price = (c['high'] + c['low']) / 2
        if price == 0:
            continue

        # Range als % van prijs
        range_pct = ((c['high'] - c['low']) / price) * 100
        ranges.append(range_pct)

        # Body als % van prijs
        body_pct = abs(c['close'] - c['open']) / price * 100
        bodies.append(body_pct)

    if not ranges:
        return False, 0, 0, 0

    avg_range = sum(ranges) / len(ranges)
    avg_body = sum(bodies) / len(bodies)

    # Tel squeeze candles (kleine bodies aan het einde)
    squeeze_count = 0
    for body in reversed(bodies):
        if body <= SQUEEZE_THRESHOLD:
            squeeze_count += 1
        else:
            break

    in_squeeze = squeeze_count >= MIN_SQUEEZE_CANDLES and avg_body <= SQUEEZE_THRESHOLD * 1.5

    return in_squeeze, squeeze_count, avg_range, avg_body


class BreakoutScanner:
    def __init__(self):
        self.coins: Dict[str, CoinState] = {}
        self.breakouts = []  # Recent breakout alerts
        self.squeezes = []   # Coins currently in squeeze
        self.message_count = 0
        self.start_time = time.time()

    def initialize_coin(self, symbol: str):
        """Initialiseer coin met historische data."""
        candles = fetch_candles(symbol)
        in_squeeze, squeeze_count, avg_range, avg_body = analyze_squeeze(candles)

        self.coins[symbol] = CoinState(
            symbol=symbol,
            prices=deque(maxlen=500),
            candles=candles,
            avg_range=avg_range,
            avg_body=avg_body,
            in_squeeze=in_squeeze,
            squeeze_candles=squeeze_count
        )

        if in_squeeze:
            self.squeezes.append(symbol)

    def update_price(self, symbol: str, price: float, volume: float = 0):
        """Update prijs en check voor breakout."""
        now = time.time()
        self.message_count += 1

        if symbol not in self.coins:
            return

        coin = self.coins[symbol]
        coin.prices.append((now, price))

        # Check voor breakout als in squeeze
        if coin.in_squeeze and coin.avg_range > 0 and len(coin.prices) >= 2:
            # Bereken huidige move
            recent_prices = [p[1] for p in list(coin.prices)[-10:]]
            if len(recent_prices) >= 2:
                min_price = min(recent_prices)
                max_price = max(recent_prices)

                if min_price > 0:
                    move_pct = ((max_price - min_price) / min_price) * 100

                    # Is dit een breakout?
                    if move_pct >= coin.avg_range * BREAKOUT_MULTIPLIER and move_pct >= MIN_BREAKOUT_PERCENT:
                        # Bepaal richting
                        first_price = recent_prices[0]
                        direction = "UP" if price > first_price else "DOWN"

                        # Voorkom duplicate alerts
                        if now - coin.last_breakout > 60:  # 60s cooldown
                            coin.last_breakout = now
                            coin.breakout_direction = direction

                            self.trigger_breakout(symbol, price, move_pct, direction, coin.squeeze_candles)

    def trigger_breakout(self, symbol: str, price: float, move_pct: float, direction: str, squeeze_candles: int):
        """Trigger breakout alert."""
        alert = {
            'time': timestamp(),
            'symbol': symbol,
            'price': price,
            'move': move_pct,
            'direction': direction,
            'squeeze_candles': squeeze_candles
        }
        self.breakouts.append(alert)

        # Keep last 20 alerts
        self.breakouts = self.breakouts[-20:]

        # Print alert
        emoji = "🚀" if direction == "UP" else "💥"
        color = C.G if direction == "UP" else C.R

        print(f"\n{C.BOLD}{emoji} BREAKOUT ALERT! {emoji}{C.END}")
        print(f"{color}{C.BOLD}{symbol}{C.END} {direction} {color}+{move_pct:.2f}%{C.END}")
        print(f"Price: {price:.8f} | Squeeze: {squeeze_candles} candles")
        print(f"{'='*50}")

    def refresh_squeeze_data(self):
        """Ververs squeeze data voor alle coins."""
        self.squeezes = []

        for symbol, coin in self.coins.items():
            candles = fetch_candles(symbol)
            if candles:
                in_squeeze, squeeze_count, avg_range, avg_body = analyze_squeeze(candles)
                coin.candles = candles
                coin.avg_range = avg_range
                coin.avg_body = avg_body
                coin.in_squeeze = in_squeeze
                coin.squeeze_candles = squeeze_count

                if in_squeeze:
                    self.squeezes.append(symbol)

            time.sleep(0.05)  # Rate limit

    def print_dashboard(self):
        """Print dashboard."""
        runtime = time.time() - self.start_time

        print(f"\n{'='*60}")
        print(f"{C.C}{C.BOLD}BREAKOUT SCANNER{C.END} - {timestamp()}")
        print(f"Runtime: {runtime/60:.1f}min | Ticks: {self.message_count:,}")
        print(f"{'='*60}")

        # Coins in squeeze (potential breakouts)
        print(f"\n{C.Y}{C.BOLD}🎯 COINS IN SQUEEZE ({len(self.squeezes)}){C.END}")
        print(f"{'─'*50}")

        squeeze_data = []
        for sym in self.squeezes[:MAX_SIGNALS]:
            coin = self.coins.get(sym)
            if coin:
                squeeze_data.append({
                    'symbol': sym,
                    'squeeze': coin.squeeze_candles,
                    'avg_body': coin.avg_body,
                    'price': list(coin.prices)[-1][1] if coin.prices else 0
                })

        # Sort by squeeze length
        squeeze_data.sort(key=lambda x: x['squeeze'], reverse=True)

        for i, s in enumerate(squeeze_data[:15], 1):
            bars = "█" * min(s['squeeze'], 20)
            print(f"  {i:2}. {s['symbol']:15} {C.Y}{bars}{C.END} ({s['squeeze']} candles)")

        if not squeeze_data:
            print(f"  Geen coins in squeeze gevonden")

        # Recent breakouts
        if self.breakouts:
            print(f"\n{C.M}{C.BOLD}⚡ RECENT BREAKOUTS{C.END}")
            print(f"{'─'*50}")

            for b in reversed(self.breakouts[-10:]):
                color = C.G if b['direction'] == 'UP' else C.R
                emoji = "🚀" if b['direction'] == 'UP' else "💥"
                print(f"  {emoji} [{b['time']}] {b['symbol']:15} {color}{b['direction']} +{b['move']:.2f}%{C.END}")

        print(f"\n{'='*60}")
        print(f"{C.W}Squeeze threshold: {SQUEEZE_THRESHOLD}% | Breakout mult: {BREAKOUT_MULTIPLIER}x{C.END}")


async def get_ws_token() -> str:
    r = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


async def handle_websocket(ws_id: int, symbols: list, scanner: BreakoutScanner):
    """Handle WebSocket."""
    reconnect_delay = 5

    while True:
        try:
            print(f"[{timestamp()}] WS-{ws_id}: Verbinden ({len(symbols)} coins)...")
            ws_url = await get_ws_token()

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                # Subscribe
                batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
                for i, batch in enumerate(batches):
                    msg = {
                        "id": f"{ws_id}_{int(time.time()*1000)}_{i}",
                        "type": "subscribe",
                        "topic": "/market/ticker:" + ",".join(batch),
                        "privateChannel": False,
                        "response": True
                    }
                    await ws.send(json.dumps(msg))
                    await asyncio.sleep(0.2)

                print(f"[{timestamp()}] WS-{ws_id}: Connected!")
                reconnect_delay = 5

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)

                        if data.get("type") == "message" and "data" in data:
                            topic = data.get("topic", "")
                            if ":" in topic:
                                symbol = topic.split(":")[-1]
                                ticker = data.get("data", {})

                                if "price" in ticker:
                                    price = float(ticker['price'])
                                    vol = float(ticker.get('size', 0))
                                    scanner.update_price(symbol, price, vol)

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"type": "ping"}))
                    except websockets.exceptions.ConnectionClosed:
                        break

        except Exception as e:
            print(f"[{timestamp()}] WS-{ws_id} Error: {e}")

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


async def dashboard_updater(scanner: BreakoutScanner):
    """Update dashboard."""
    last_refresh = 0

    while True:
        await asyncio.sleep(UPDATE_INTERVAL)

        # Refresh squeeze data elke 2 minuten
        if time.time() - last_refresh > 120:
            print(f"\n[{timestamp()}] Refreshing squeeze data...")
            scanner.refresh_squeeze_data()
            last_refresh = time.time()

        scanner.print_dashboard()


async def main():
    print(f"""
{C.C}{C.BOLD}
╔═══════════════════════════════════════════════════════════╗
║         SQUEEZE & BREAKOUT SCANNER                        ║
║                                                           ║
║  Detecteert consolidatie (kleine kaarsen) → explosie      ║
╚═══════════════════════════════════════════════════════════╝
{C.END}
""")

    # Get symbols
    symbols = get_high_volume_symbols()
    if not symbols:
        print("Geen symbols!")
        return

    # Create scanner
    scanner = BreakoutScanner()

    # Initialize coins with historical data
    print(f"[{timestamp()}] Initialiseren van {len(symbols)} coins...")
    for i, sym in enumerate(symbols):
        scanner.initialize_coin(sym)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(symbols)} geinitialiseerd...")
        time.sleep(0.05)  # Rate limit

    print(f"[{timestamp()}] Gevonden: {len(scanner.squeezes)} coins in squeeze")

    # Split symbols
    chunks = [symbols[i:i+MAX_SUBSCRIPTIONS] for i in range(0, len(symbols), MAX_SUBSCRIPTIONS)]
    print(f"[{timestamp()}] Gebruik {len(chunks)} WebSocket(s)")

    # Tasks
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(asyncio.create_task(handle_websocket(i, chunk, scanner)))

    tasks.append(asyncio.create_task(dashboard_updater(scanner)))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Gestopt")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
