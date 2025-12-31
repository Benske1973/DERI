# mover_scanner.py - Real-time mover detector via KuCoin WebSocket
import asyncio
import websockets
import requests
import json
import time
from collections import deque
from datetime import datetime

# ============= CONFIGURATIE =============
MIN_MOVE_PERCENT = 0.5       # Minimum % beweging om als mover te tellen
WINDOW_SECONDS = 60          # Tijdvenster voor % berekening (60s = 1 min)
TOP_MOVERS_COUNT = 10        # Aantal top movers om te tonen
UPDATE_INTERVAL = 3          # Seconden tussen updates
MIN_VOLUME_USDT = 50000      # Minimum 24h volume
ALERT_THRESHOLD = 2.0        # % voor speciale alert
MAX_SUBSCRIPTIONS = 300      # Max symbols per WebSocket
BATCH_SIZE = 50              # Symbols per subscribe call

# ============= KLEUREN =============
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


def color_percent(pct: float) -> str:
    """Kleur percentage groen/rood."""
    if pct >= ALERT_THRESHOLD:
        return f"{Colors.BOLD}{Colors.GREEN}+{pct:.2f}%{Colors.END}"
    elif pct > 0:
        return f"{Colors.GREEN}+{pct:.2f}%{Colors.END}"
    elif pct <= -ALERT_THRESHOLD:
        return f"{Colors.BOLD}{Colors.RED}{pct:.2f}%{Colors.END}"
    else:
        return f"{Colors.RED}{pct:.2f}%{Colors.END}"


def get_high_volume_symbols() -> list:
    """Haal symbols op met hoog volume."""
    print(f"[{timestamp()}] Ophalen van symbols met volume > ${MIN_VOLUME_USDT:,}...")

    try:
        url = "https://api.kucoin.com/api/v1/market/allTickers"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        symbols = []
        for ticker in data['data'].get('ticker', []):
            symbol = ticker.get('symbol', '')
            if not symbol.endswith('-USDT'):
                continue

            # Filter leveraged tokens
            if any(x in symbol for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-']):
                continue

            try:
                vol = float(ticker.get('volValue', 0))
                if vol >= MIN_VOLUME_USDT:
                    symbols.append({
                        'symbol': symbol,
                        'volume': vol,
                        'price': float(ticker.get('last', 0))
                    })
            except:
                continue

        symbols.sort(key=lambda x: x['volume'], reverse=True)
        print(f"[{timestamp()}] Gevonden: {len(symbols)} symbols")
        return [s['symbol'] for s in symbols]

    except Exception as e:
        print(f"Error: {e}")
        return []


def timestamp() -> str:
    """Huidige tijd string."""
    return datetime.now().strftime('%H:%M:%S')


async def get_ws_token() -> str:
    """Haal WebSocket token op."""
    r = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    r.raise_for_status()
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


class MoverScanner:
    def __init__(self):
        self.price_history = {}  # symbol -> deque of (timestamp, price)
        self.current_prices = {}  # symbol -> latest price
        self.volume_data = {}     # symbol -> volume info
        self.message_count = 0
        self.start_time = time.time()
        self.alerts = []

    def update_price(self, symbol: str, price: float, volume: float = None):
        """Update prijs voor een symbol."""
        now = time.time()

        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=1000)

        self.price_history[symbol].append((now, price))
        self.current_prices[symbol] = price

        if volume:
            self.volume_data[symbol] = volume

        self.message_count += 1

    def cleanup_old_data(self):
        """Verwijder oude prijsdata."""
        now = time.time()
        cutoff = now - WINDOW_SECONDS

        for symbol, history in self.price_history.items():
            while history and history[0][0] < cutoff:
                history.popleft()

    def calculate_change(self, symbol: str) -> float:
        """Bereken % verandering in window."""
        if symbol not in self.price_history:
            return 0

        history = self.price_history[symbol]
        if len(history) < 2:
            return 0

        oldest_price = history[0][1]
        newest_price = history[-1][1]

        if oldest_price == 0:
            return 0

        return ((newest_price - oldest_price) / oldest_price) * 100

    def get_top_movers(self) -> tuple:
        """Haal top gainers en losers op."""
        changes = []

        for symbol in self.price_history:
            change = self.calculate_change(symbol)
            if abs(change) >= MIN_MOVE_PERCENT:
                changes.append({
                    'symbol': symbol,
                    'change': change,
                    'price': self.current_prices.get(symbol, 0)
                })

        # Sort by change
        gainers = sorted([c for c in changes if c['change'] > 0],
                        key=lambda x: x['change'], reverse=True)[:TOP_MOVERS_COUNT]

        losers = sorted([c for c in changes if c['change'] < 0],
                       key=lambda x: x['change'])[:TOP_MOVERS_COUNT]

        return gainers, losers

    def check_alerts(self, symbol: str, change: float):
        """Check voor alert triggers."""
        if abs(change) >= ALERT_THRESHOLD:
            alert = {
                'time': timestamp(),
                'symbol': symbol,
                'change': change,
                'price': self.current_prices.get(symbol, 0)
            }
            self.alerts.append(alert)

            # Print alert
            direction = "PUMP" if change > 0 else "DUMP"
            emoji = "🚀" if change > 0 else "💥"
            print(f"\n{emoji} {Colors.BOLD}ALERT: {symbol} {direction}! {color_percent(change)}{Colors.END} @ {alert['price']:.8f}")

    def print_dashboard(self):
        """Print het dashboard."""
        self.cleanup_old_data()
        gainers, losers = self.get_top_movers()

        # Clear screen (optional)
        # print("\033[2J\033[H", end="")

        runtime = time.time() - self.start_time
        active_coins = len([h for h in self.price_history.values() if len(h) >= 2])

        print(f"\n{'='*60}")
        print(f"{Colors.CYAN}{Colors.BOLD}KUCOIN MOVER SCANNER{Colors.END} - {timestamp()}")
        print(f"Runtime: {runtime/60:.1f}min | Messages: {self.message_count:,} | Active: {active_coins} coins")
        print(f"Window: {WINDOW_SECONDS}s | Min move: {MIN_MOVE_PERCENT}%")
        print(f"{'='*60}")

        # Top Gainers
        print(f"\n{Colors.GREEN}{Colors.BOLD}📈 TOP GAINERS{Colors.END}")
        print(f"{'-'*45}")
        if gainers:
            for i, g in enumerate(gainers, 1):
                print(f"  {i:2}. {g['symbol']:15} {color_percent(g['change']):>15}  @ {g['price']:.8f}")
        else:
            print(f"  Geen gainers > {MIN_MOVE_PERCENT}%")

        # Top Losers
        print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP LOSERS{Colors.END}")
        print(f"{'-'*45}")
        if losers:
            for i, l in enumerate(losers, 1):
                print(f"  {i:2}. {l['symbol']:15} {color_percent(l['change']):>15}  @ {l['price']:.8f}")
        else:
            print(f"  Geen losers > {MIN_MOVE_PERCENT}%")

        # Recent alerts
        if self.alerts:
            recent_alerts = self.alerts[-5:]  # Last 5
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚡ RECENT ALERTS{Colors.END}")
            print(f"{'-'*45}")
            for a in reversed(recent_alerts):
                print(f"  [{a['time']}] {a['symbol']:15} {color_percent(a['change'])}")

        print(f"\n{'='*60}")


async def handle_websocket(ws_id: int, symbols: list, scanner: MoverScanner):
    """Handle WebSocket connectie."""
    reconnect_delay = 5

    while True:
        try:
            print(f"[{timestamp()}] WS-{ws_id}: Verbinden voor {len(symbols)} coins...")
            ws_url = await get_ws_token()

            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:

                # Subscribe in batches
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

                print(f"[{timestamp()}] WS-{ws_id}: Verbonden!")
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
                                    volume = float(ticker.get('size', 0))

                                    # Update scanner
                                    scanner.update_price(symbol, price, volume)

                                    # Check for alerts
                                    change = scanner.calculate_change(symbol)
                                    if abs(change) >= ALERT_THRESHOLD:
                                        scanner.check_alerts(symbol, change)

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"type": "ping"}))
                    except websockets.exceptions.ConnectionClosed:
                        print(f"[{timestamp()}] WS-{ws_id}: Connectie gesloten")
                        break

        except Exception as e:
            print(f"[{timestamp()}] WS-{ws_id} Error: {e}")

        print(f"[{timestamp()}] WS-{ws_id}: Reconnect in {reconnect_delay}s...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


async def dashboard_updater(scanner: MoverScanner):
    """Update dashboard periodiek."""
    await asyncio.sleep(5)  # Wacht tot data binnenkomt

    while True:
        scanner.print_dashboard()
        await asyncio.sleep(UPDATE_INTERVAL)


async def main():
    print(f"""
{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════════╗
║           KUCOIN REAL-TIME MOVER SCANNER                 ║
║                                                           ║
║  Detecteert coins met grote prijsbewegingen in real-time ║
╚═══════════════════════════════════════════════════════════╝
{Colors.END}
""")

    # Get symbols
    symbols = get_high_volume_symbols()
    if not symbols:
        print("Geen symbols gevonden!")
        return

    # Split over WebSocket connections
    chunks = [symbols[i:i+MAX_SUBSCRIPTIONS] for i in range(0, len(symbols), MAX_SUBSCRIPTIONS)]
    print(f"[{timestamp()}] Gebruik {len(chunks)} WebSocket connectie(s)")

    # Create scanner
    scanner = MoverScanner()

    # Create tasks
    tasks = []

    # WebSocket tasks
    for i, chunk in enumerate(chunks):
        task = asyncio.create_task(handle_websocket(i, chunk, scanner))
        tasks.append(task)

    # Dashboard task
    dashboard_task = asyncio.create_task(dashboard_updater(scanner))
    tasks.append(dashboard_task)

    # Run
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n[{timestamp()}] Gestopt door gebruiker")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgramma beëindigd")
