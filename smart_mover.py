# smart_mover.py - Intelligente mover detectie met early entry focus
import asyncio
import websockets
import requests
import json
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============= CONFIGURATIE =============
# Volume & Filter
MIN_VOLUME_24H = 100000      # Min 24h volume USDT
MIN_PRICE = 0.00000001       # Min prijs (filter dust)

# Move Detectie
LOOKBACK_SECONDS = 120       # Window voor baseline (2 min)
EARLY_WINDOW = 10            # Window voor vroege move detectie (10 sec)
MIN_MOVE_PERCENT = 0.5       # Minimum move % voor interesse
MAX_MOVE_PERCENT = 5.0       # Maximum - boven dit is te laat (piek)

# Volume Analyse
VOLUME_SPIKE_MULT = 2.0      # Volume moet 2x hoger zijn dan gemiddeld
MIN_TRADES_FOR_SIGNAL = 3    # Minimum aantal trades in EARLY_WINDOW

# Fase Detectie
PHASE_EARLY_MAX = 1.5        # 0-1.5% = EARLY phase (instap moment!)
PHASE_MID_MAX = 3.0          # 1.5-3% = MID phase (nog ok)
# > 3% = LATE phase (te laat, niet kopen)

# Alert Settings
ALERT_COOLDOWN = 60          # Seconden tussen alerts voor zelfde coin
MAX_ACTIVE_SIGNALS = 10      # Max actieve signalen tonen

# WebSocket
MAX_SUBS_PER_WS = 300
BATCH_SIZE = 50

# ============= KLEUREN =============
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


@dataclass
class PriceTick:
    timestamp: float
    price: float
    volume: float


@dataclass
class CoinData:
    symbol: str
    ticks: deque = field(default_factory=lambda: deque(maxlen=2000))
    baseline_price: float = 0
    baseline_volume: float = 0
    current_move: float = 0
    move_phase: str = "NONE"
    volume_ratio: float = 0
    last_alert: float = 0
    trend_start_price: float = 0
    trend_start_time: float = 0
    tick_count_early: int = 0


@dataclass
class Signal:
    timestamp: str
    symbol: str
    phase: str
    move_percent: float
    volume_ratio: float
    price: float
    trend_duration: float


def ts() -> str:
    return datetime.now().strftime('%H:%M:%S')


def get_symbols() -> list:
    """Haal tradeable high-volume symbols op."""
    print(f"[{ts()}] Ophalen symbols...")
    try:
        r = requests.get("https://api.kucoin.com/api/v1/market/allTickers", timeout=10)
        data = r.json()

        symbols = []
        for t in data['data'].get('ticker', []):
            sym = t.get('symbol', '')
            if not sym.endswith('-USDT'):
                continue
            # Filter leveraged/special tokens
            if any(x in sym for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-', '2L-', '2S-']):
                continue
            try:
                vol = float(t.get('volValue', 0))
                price = float(t.get('last', 0))
                if vol >= MIN_VOLUME_24H and price >= MIN_PRICE:
                    symbols.append(sym)
            except:
                continue

        print(f"[{ts()}] Gevonden: {len(symbols)} symbols")
        return symbols
    except Exception as e:
        print(f"Error: {e}")
        return []


class SmartMoverScanner:
    def __init__(self):
        self.coins: Dict[str, CoinData] = {}
        self.signals: List[Signal] = []
        self.msg_count = 0
        self.start_time = time.time()

    def process_tick(self, symbol: str, price: float, volume: float):
        """Verwerk een nieuwe tick."""
        now = time.time()
        self.msg_count += 1

        # Init coin als nodig
        if symbol not in self.coins:
            self.coins[symbol] = CoinData(symbol=symbol)

        coin = self.coins[symbol]
        coin.ticks.append(PriceTick(now, price, volume))

        # Bereken baseline (gemiddelde prijs over LOOKBACK_SECONDS)
        self._update_baseline(coin, now)

        # Alleen verder als we genoeg data hebben
        if coin.baseline_price == 0:
            return

        # Bereken huidige move
        coin.current_move = ((price - coin.baseline_price) / coin.baseline_price) * 100

        # Bepaal fase
        coin.move_phase = self._determine_phase(coin.current_move)

        # Check voor EARLY entry signaal
        if coin.move_phase == "EARLY" and coin.current_move >= MIN_MOVE_PERCENT:
            self._check_signal(coin, now, price)

    def _update_baseline(self, coin: CoinData, now: float):
        """Update baseline prijs en volume."""
        cutoff = now - LOOKBACK_SECONDS

        # Filter oude ticks voor baseline
        baseline_ticks = [t for t in coin.ticks if t.timestamp < cutoff - EARLY_WINDOW]

        if len(baseline_ticks) < 10:
            return

        # Baseline = gemiddelde prijs van oude ticks
        prices = [t.price for t in baseline_ticks[-50:]]  # Laatste 50 baseline ticks
        coin.baseline_price = sum(prices) / len(prices)

        # Baseline volume
        volumes = [t.volume for t in baseline_ticks[-50:]]
        coin.baseline_volume = sum(volumes) / len(volumes) if volumes else 0

    def _determine_phase(self, move_pct: float) -> str:
        """Bepaal de fase van de move."""
        if move_pct < MIN_MOVE_PERCENT:
            return "NONE"
        elif move_pct <= PHASE_EARLY_MAX:
            return "EARLY"  # 🟢 INSTAP MOMENT
        elif move_pct <= PHASE_MID_MAX:
            return "MID"    # 🟡 Nog acceptabel
        else:
            return "LATE"   # 🔴 Te laat, niet kopen!

    def _check_signal(self, coin: CoinData, now: float, price: float):
        """Check of dit een valide early entry signaal is."""

        # Cooldown check
        if now - coin.last_alert < ALERT_COOLDOWN:
            return

        # Tel recente ticks (voor volume/momentum check)
        early_ticks = [t for t in coin.ticks if now - t.timestamp <= EARLY_WINDOW]
        coin.tick_count_early = len(early_ticks)

        if coin.tick_count_early < MIN_TRADES_FOR_SIGNAL:
            return  # Niet genoeg activiteit

        # Volume ratio berekenen
        if coin.baseline_volume > 0:
            recent_vol = sum(t.volume for t in early_ticks) / len(early_ticks)
            coin.volume_ratio = recent_vol / coin.baseline_volume
        else:
            coin.volume_ratio = 1.0

        # Volume spike check
        if coin.volume_ratio < VOLUME_SPIKE_MULT:
            return  # Geen volume confirmatie

        # Track trend start
        if coin.trend_start_time == 0 or now - coin.trend_start_time > 300:
            coin.trend_start_price = coin.baseline_price
            coin.trend_start_time = now

        trend_duration = now - coin.trend_start_time

        # SIGNAAL!
        signal = Signal(
            timestamp=ts(),
            symbol=coin.symbol,
            phase=coin.move_phase,
            move_percent=coin.current_move,
            volume_ratio=coin.volume_ratio,
            price=price,
            trend_duration=trend_duration
        )

        self.signals.append(signal)
        self.signals = self.signals[-50:]  # Keep last 50

        coin.last_alert = now

        # Print alert
        self._print_alert(signal)

    def _print_alert(self, sig: Signal):
        """Print een entry alert."""
        print(f"\n{'='*55}")
        print(f"{C.G}{C.BOLD}🎯 EARLY ENTRY SIGNAL!{C.END}")
        print(f"{'='*55}")
        print(f"{C.BOLD}{sig.symbol}{C.END}")
        print(f"  Phase:    {C.G}EARLY{C.END} (goed instapmoment)")
        print(f"  Move:     {C.G}+{sig.move_percent:.2f}%{C.END} vanaf baseline")
        print(f"  Volume:   {C.C}{sig.volume_ratio:.1f}x{C.END} normaal")
        print(f"  Price:    {sig.price:.8f}")
        print(f"  Duration: {sig.trend_duration:.0f}s sinds start")
        print(f"{'='*55}")

    def print_dashboard(self):
        """Print het dashboard."""
        now = time.time()
        runtime = now - self.start_time

        # Verzamel actieve movers
        active_movers = []
        for sym, coin in self.coins.items():
            if coin.move_phase != "NONE" and coin.current_move >= MIN_MOVE_PERCENT:
                active_movers.append({
                    'symbol': sym,
                    'move': coin.current_move,
                    'phase': coin.move_phase,
                    'volume': coin.volume_ratio,
                    'ticks': coin.tick_count_early
                })

        # Sort by move %
        active_movers.sort(key=lambda x: x['move'], reverse=True)

        print(f"\n{'═'*60}")
        print(f"{C.C}{C.BOLD}SMART MOVER SCANNER{C.END} - {ts()}")
        print(f"Runtime: {runtime/60:.1f}min | Ticks: {self.msg_count:,} | Active: {len(self.coins)}")
        print(f"{'═'*60}")

        # Active Movers by Phase
        print(f"\n{C.BOLD}📊 ACTIVE MOVERS{C.END}")
        print(f"{'─'*60}")
        print(f"{'Symbol':<14} {'Move':>8} {'Phase':<7} {'Vol':>6} {'Status':<15}")
        print(f"{'─'*60}")

        for m in active_movers[:15]:
            # Phase kleuren
            if m['phase'] == 'EARLY':
                phase_str = f"{C.G}EARLY{C.END}"
                status = f"{C.G}✓ ENTRY ZONE{C.END}"
            elif m['phase'] == 'MID':
                phase_str = f"{C.Y}MID{C.END}"
                status = f"{C.Y}⚠ Risky{C.END}"
            else:
                phase_str = f"{C.R}LATE{C.END}"
                status = f"{C.R}✗ Te laat{C.END}"

            # Move kleur
            move_str = f"{C.G}+{m['move']:.2f}%{C.END}" if m['move'] > 0 else f"{C.R}{m['move']:.2f}%{C.END}"

            # Volume indicator
            vol_str = f"{m['volume']:.1f}x" if m['volume'] > 0 else "N/A"
            if m['volume'] >= VOLUME_SPIKE_MULT:
                vol_str = f"{C.C}{vol_str}{C.END}"

            print(f"{m['symbol']:<14} {move_str:>16} {phase_str:<15} {vol_str:>10} {status}")

        if not active_movers:
            print(f"  {C.DIM}Geen actieve movers gevonden{C.END}")

        # Recent Signals
        if self.signals:
            print(f"\n{C.BOLD}⚡ RECENT EARLY SIGNALS{C.END}")
            print(f"{'─'*60}")
            for sig in reversed(self.signals[-5:]):
                print(f"  [{sig.timestamp}] {C.G}{sig.symbol:<12}{C.END} +{sig.move_percent:.2f}% vol:{sig.volume_ratio:.1f}x")

        print(f"\n{'═'*60}")
        print(f"{C.DIM}Entry zone: {MIN_MOVE_PERCENT}-{PHASE_EARLY_MAX}% | Te laat: >{PHASE_MID_MAX}%{C.END}")
        print(f"{C.DIM}Volume spike: {VOLUME_SPIKE_MULT}x | Baseline: {LOOKBACK_SECONDS}s{C.END}")


async def get_ws_token():
    r = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


async def ws_handler(ws_id: int, symbols: list, scanner: SmartMoverScanner):
    """WebSocket handler."""
    reconnect_delay = 5

    while True:
        try:
            print(f"[{ts()}] WS-{ws_id}: Verbinden ({len(symbols)} coins)...")
            ws_url = await get_ws_token()

            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                # Subscribe
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
                    await asyncio.sleep(0.15)

                print(f"[{ts()}] WS-{ws_id}: Connected!")
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
                                    scanner.process_tick(symbol, price, vol)

                    except asyncio.TimeoutError:
                        await ws.send(json.dumps({"type": "ping"}))
                    except websockets.exceptions.ConnectionClosed:
                        break

        except Exception as e:
            print(f"[{ts()}] WS-{ws_id} Error: {e}")

        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


async def dashboard_loop(scanner: SmartMoverScanner):
    """Dashboard update loop."""
    await asyncio.sleep(5)
    while True:
        scanner.print_dashboard()
        await asyncio.sleep(3)


async def main():
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║              SMART MOVER SCANNER v2                          ║
║                                                              ║
║  🎯 Detecteert VROEGE moves (niet de piek!)                  ║
║  📊 Volume confirmatie voor echte moves                      ║
║  ⚡ Real-time via WebSocket                                  ║
╚══════════════════════════════════════════════════════════════╝
{C.END}
""")

    print(f"{C.Y}Instellingen:{C.END}")
    print(f"  • Entry zone: {MIN_MOVE_PERCENT}% - {PHASE_EARLY_MAX}%")
    print(f"  • Te laat zone: > {PHASE_MID_MAX}%")
    print(f"  • Volume spike vereist: {VOLUME_SPIKE_MULT}x normaal")
    print(f"  • Baseline window: {LOOKBACK_SECONDS}s")
    print()

    symbols = get_symbols()
    if not symbols:
        return

    scanner = SmartMoverScanner()

    chunks = [symbols[i:i+MAX_SUBS_PER_WS] for i in range(0, len(symbols), MAX_SUBS_PER_WS)]
    print(f"[{ts()}] {len(chunks)} WebSocket(s) nodig")

    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(asyncio.create_task(ws_handler(i, chunk, scanner)))

    tasks.append(asyncio.create_task(dashboard_loop(scanner)))

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] Gestopt")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
