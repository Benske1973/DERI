# coin_health_scanner.py - KuCoin Coin Health Analyzer
# Analyseert welke coins "dood/stervend" zijn vs handelbaar
# Output: JSON bestand met classificaties, scant elk uur

import asyncio
import aiohttp
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

# Health Thresholds
VOLUME_24H_DEAD = 10000          # <10K = dood
VOLUME_24H_DYING = 50000         # <50K = stervend
VOLUME_24H_WEAK = 100000         # <100K = zwak
VOLUME_24H_HEALTHY = 500000      # >500K = gezond
VOLUME_24H_HOT = 2000000         # >2M = hot

# Price Change Thresholds (%)
PRICE_DYING_THRESHOLD = -50      # -50% in 30 dagen = stervend
PRICE_WEAK_THRESHOLD = -30       # -30% = zwak
PRICE_PUMP_THRESHOLD = 100       # +100% = pump (gevaarlijk)

# Volatility Thresholds (%)
VOLATILITY_DEAD = 0.5            # <0.5% dagelijkse range = dood
VOLATILITY_LOW = 1.0             # <1% = lage volatiliteit
VOLATILITY_HEALTHY = 2.0         # 2-8% = gezond
VOLATILITY_HIGH = 10.0           # >10% = hoge volatiliteit

# Volume Trend (vergelijking recent vs historisch)
VOLUME_DECLINE_SEVERE = 0.3      # Volume <30% van gemiddelde = severe decline
VOLUME_DECLINE_MODERATE = 0.6    # <60% = moderate decline
VOLUME_INCREASE_MODERATE = 1.5   # >150% = moderate increase
VOLUME_INCREASE_HIGH = 3.0       # >300% = high increase

# Scan Settings
SCAN_INTERVAL = 3600             # 1 uur in seconden
MAX_CONCURRENT = 20
RATE_LIMIT_DELAY = 0.05
OUTPUT_FILE = "coin_health.json"
HISTORY_FILE = "coin_health_history.json"

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
class CoinHealth:
    symbol: str
    price: float
    volume_24h: float
    change_24h: float

    # Classification
    health_status: str          # DEAD, DYING, WEAK, NEUTRAL, HEALTHY, HOT
    health_score: int           # 0-100
    tradeable: bool             # True/False
    risk_level: str             # EXTREME, HIGH, MEDIUM, LOW

    # Metrics
    volatility: float           # Dagelijkse volatiliteit %
    volume_trend: str           # SEVERE_DECLINE, DECLINING, STABLE, INCREASING, SURGING
    volume_ratio: float         # Recent volume / avg volume
    price_trend: str            # CRASHING, FALLING, STABLE, RISING, PUMPING

    # Warnings
    warnings: List[str]

    # Recommendation
    recommendation: str         # AVOID, CAUTION, NEUTRAL, CONSIDER, WATCH

    # Timestamps
    last_updated: str


@dataclass
class ScanSummary:
    timestamp: str
    total_coins: int
    dead_coins: int
    dying_coins: int
    weak_coins: int
    neutral_coins: int
    healthy_coins: int
    hot_coins: int
    tradeable_count: int
    avoid_count: int
    top_healthy: List[str]
    top_volume: List[str]
    biggest_gainers: List[str]
    biggest_losers: List[str]
    new_deaths: List[str]       # Coins die recent "dood" werden
    recovering: List[str]       # Coins die beter worden


# ═══════════════════════════════════════════════════════════════
#                      API FUNCTIONS
# ═══════════════════════════════════════════════════════════════

async def get_all_tickers(session: aiohttp.ClientSession) -> List[Dict]:
    """Haal alle ticker data op."""
    try:
        async with session.get(f"{KUCOIN_API}/api/v1/market/allTickers") as resp:
            data = await resp.json()

        if data.get('code') != '200000':
            return []

        tickers = []
        for t in data['data'].get('ticker', []):
            sym = t.get('symbol', '')
            if not sym.endswith('-USDT'):
                continue
            # Skip leveraged tokens
            if any(x in sym for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-', '2L-', '2S-']):
                continue

            try:
                tickers.append({
                    'symbol': sym,
                    'price': float(t.get('last', 0)),
                    'volume': float(t.get('volValue', 0)),
                    'change': float(t.get('changeRate', 0)) * 100,
                    'high': float(t.get('high', 0)),
                    'low': float(t.get('low', 0)),
                    'buy': float(t.get('buy', 0)),
                    'sell': float(t.get('sell', 0)),
                })
            except:
                continue

        return tickers
    except Exception as e:
        print(f"Error getting tickers: {e}")
        return []


async def get_candles(session: aiohttp.ClientSession, symbol: str,
                      timeframe: str, semaphore: asyncio.Semaphore) -> Optional[List]:
    """Haal candle data op."""
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_DELAY)
        try:
            url = f"{KUCOIN_API}/api/v1/market/candles?symbol={symbol}&type={timeframe}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

            if data.get('code') != '200000' or not data.get('data'):
                return None

            return data['data']
        except:
            return None


# ═══════════════════════════════════════════════════════════════
#                      ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def analyze_volatility(high: float, low: float, price: float) -> float:
    """Bereken dagelijkse volatiliteit."""
    if price == 0:
        return 0
    return ((high - low) / price) * 100


def analyze_volume_trend(recent_volumes: List[float]) -> tuple:
    """Analyseer volume trend."""
    if not recent_volumes or len(recent_volumes) < 10:
        return "UNKNOWN", 1.0

    # Recent (laatste 5) vs historisch (5-20)
    recent = sum(recent_volumes[:5]) / 5
    historical = sum(recent_volumes[5:20]) / min(15, len(recent_volumes[5:20])) if len(recent_volumes) > 5 else recent

    if historical == 0:
        return "UNKNOWN", 1.0

    ratio = recent / historical

    if ratio < VOLUME_DECLINE_SEVERE:
        return "SEVERE_DECLINE", ratio
    elif ratio < VOLUME_DECLINE_MODERATE:
        return "DECLINING", ratio
    elif ratio < VOLUME_INCREASE_MODERATE:
        return "STABLE", ratio
    elif ratio < VOLUME_INCREASE_HIGH:
        return "INCREASING", ratio
    else:
        return "SURGING", ratio


def analyze_price_trend(changes: List[float]) -> str:
    """Analyseer prijstrend over meerdere periodes."""
    if not changes:
        return "UNKNOWN"

    total_change = sum(changes)

    if total_change < PRICE_DYING_THRESHOLD:
        return "CRASHING"
    elif total_change < PRICE_WEAK_THRESHOLD:
        return "FALLING"
    elif total_change > PRICE_PUMP_THRESHOLD:
        return "PUMPING"
    elif total_change > 30:
        return "RISING"
    else:
        return "STABLE"


def calculate_health_score(volume: float, volatility: float, volume_ratio: float,
                           change_24h: float, warnings: List[str]) -> int:
    """Bereken health score 0-100."""
    score = 50  # Start at neutral

    # Volume score (max +/- 25)
    if volume >= VOLUME_24H_HOT:
        score += 25
    elif volume >= VOLUME_24H_HEALTHY:
        score += 20
    elif volume >= VOLUME_24H_WEAK:
        score += 10
    elif volume >= VOLUME_24H_DYING:
        score -= 10
    elif volume >= VOLUME_24H_DEAD:
        score -= 20
    else:
        score -= 25

    # Volatility score (max +/- 15)
    if VOLATILITY_HEALTHY <= volatility <= VOLATILITY_HIGH:
        score += 15
    elif VOLATILITY_LOW <= volatility < VOLATILITY_HEALTHY:
        score += 5
    elif volatility < VOLATILITY_DEAD:
        score -= 15
    elif volatility > VOLATILITY_HIGH * 2:
        score -= 10  # Too volatile

    # Volume trend score (max +/- 15)
    if volume_ratio >= VOLUME_INCREASE_HIGH:
        score += 15
    elif volume_ratio >= VOLUME_INCREASE_MODERATE:
        score += 10
    elif volume_ratio < VOLUME_DECLINE_SEVERE:
        score -= 15
    elif volume_ratio < VOLUME_DECLINE_MODERATE:
        score -= 10

    # 24h change modifier (max +/- 10)
    if -10 <= change_24h <= 20:
        score += 5  # Stable/slight growth is healthy
    elif change_24h < -30:
        score -= 10
    elif change_24h > 50:
        score -= 5  # Pump might be risky

    # Warning penalties
    score -= len(warnings) * 3

    return max(0, min(100, score))


def classify_coin(ticker: Dict, candle_data: Optional[List]) -> CoinHealth:
    """Classificeer een coin op basis van alle metrics."""

    symbol = ticker['symbol']
    price = ticker['price']
    volume = ticker['volume']
    change_24h = ticker['change']
    volatility = analyze_volatility(ticker['high'], ticker['low'], price)

    warnings = []

    # Analyze candle data for volume trend
    volume_trend = "UNKNOWN"
    volume_ratio = 1.0
    price_trend = "UNKNOWN"

    if candle_data and len(candle_data) > 10:
        # Extract volumes from candles
        volumes = [float(c[5]) for c in candle_data[:30]]
        volume_trend, volume_ratio = analyze_volume_trend(volumes)

        # Extract price changes
        closes = [float(c[2]) for c in candle_data[:30]]
        if len(closes) >= 2:
            changes = []
            for i in range(0, min(len(closes)-1, 7)):
                if closes[i+1] != 0:
                    changes.append(((closes[i] - closes[i+1]) / closes[i+1]) * 100)
            price_trend = analyze_price_trend(changes)

    # ═══ GENERATE WARNINGS ═══
    if volume < VOLUME_24H_DEAD:
        warnings.append("EXTREMELY_LOW_VOLUME")
    elif volume < VOLUME_24H_DYING:
        warnings.append("VERY_LOW_VOLUME")

    if volatility < VOLATILITY_DEAD:
        warnings.append("NO_VOLATILITY")
    elif volatility > VOLATILITY_HIGH * 2:
        warnings.append("EXTREME_VOLATILITY")

    if volume_trend == "SEVERE_DECLINE":
        warnings.append("VOLUME_COLLAPSING")

    if price_trend == "CRASHING":
        warnings.append("PRICE_CRASHING")
    elif price_trend == "PUMPING":
        warnings.append("POTENTIAL_PUMP_DUMP")

    if change_24h < -20:
        warnings.append("MAJOR_DUMP_24H")
    elif change_24h > 50:
        warnings.append("MAJOR_PUMP_24H")

    # Check spread
    if ticker['buy'] > 0 and ticker['sell'] > 0:
        spread = ((ticker['sell'] - ticker['buy']) / ticker['buy']) * 100
        if spread > 2:
            warnings.append("WIDE_SPREAD")

    # ═══ HEALTH CLASSIFICATION ═══
    health_score = calculate_health_score(volume, volatility, volume_ratio, change_24h, warnings)

    if health_score < 20 or volume < VOLUME_24H_DEAD:
        health_status = "DEAD"
        tradeable = False
        risk_level = "EXTREME"
        recommendation = "AVOID"
    elif health_score < 35 or volume < VOLUME_24H_DYING:
        health_status = "DYING"
        tradeable = False
        risk_level = "EXTREME"
        recommendation = "AVOID"
    elif health_score < 50 or volume < VOLUME_24H_WEAK:
        health_status = "WEAK"
        tradeable = True
        risk_level = "HIGH"
        recommendation = "CAUTION"
    elif health_score < 65:
        health_status = "NEUTRAL"
        tradeable = True
        risk_level = "MEDIUM"
        recommendation = "NEUTRAL"
    elif health_score < 80 or volume >= VOLUME_24H_HEALTHY:
        health_status = "HEALTHY"
        tradeable = True
        risk_level = "LOW"
        recommendation = "CONSIDER"
    else:
        health_status = "HOT"
        tradeable = True
        risk_level = "LOW"
        recommendation = "WATCH"

    # Override for extreme cases
    if "VOLUME_COLLAPSING" in warnings and "PRICE_CRASHING" in warnings:
        health_status = "DYING"
        tradeable = False
        recommendation = "AVOID"

    if "POTENTIAL_PUMP_DUMP" in warnings:
        risk_level = "HIGH"
        recommendation = "CAUTION"

    return CoinHealth(
        symbol=symbol,
        price=price,
        volume_24h=volume,
        change_24h=change_24h,
        health_status=health_status,
        health_score=health_score,
        tradeable=tradeable,
        risk_level=risk_level,
        volatility=round(volatility, 2),
        volume_trend=volume_trend,
        volume_ratio=round(volume_ratio, 2),
        price_trend=price_trend,
        warnings=warnings,
        recommendation=recommendation,
        last_updated=datetime.now().isoformat()
    )


# ═══════════════════════════════════════════════════════════════
#                      SCANNER
# ═══════════════════════════════════════════════════════════════

async def scan_all_coins() -> tuple:
    """Scan alle coins en classificeer ze."""

    async with aiohttp.ClientSession() as session:
        # Get all tickers
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching all tickers...")
        tickers = await get_all_tickers(session)

        if not tickers:
            print("Geen tickers gevonden!")
            return [], None

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(tickers)} USDT pairs")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching candle data for analysis...")

        # Get candle data for top coins (by volume) for detailed analysis
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        # Sort by volume and get candles for top 200
        tickers_sorted = sorted(tickers, key=lambda x: x['volume'], reverse=True)
        top_symbols = [t['symbol'] for t in tickers_sorted[:200]]

        # Fetch candles
        candle_tasks = {
            sym: get_candles(session, sym, "1day", semaphore)
            for sym in top_symbols
        }

        candle_results = {}
        for sym, task in candle_tasks.items():
            candle_results[sym] = await task

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Analyzing {len(tickers)} coins...")

        # Classify all coins
        health_results = []
        for ticker in tickers:
            candles = candle_results.get(ticker['symbol'])
            health = classify_coin(ticker, candles)
            health_results.append(health)

        # Create summary
        summary = create_summary(health_results)

        return health_results, summary


def create_summary(results: List[CoinHealth]) -> ScanSummary:
    """Maak een samenvatting van de scan."""

    dead = [r for r in results if r.health_status == "DEAD"]
    dying = [r for r in results if r.health_status == "DYING"]
    weak = [r for r in results if r.health_status == "WEAK"]
    neutral = [r for r in results if r.health_status == "NEUTRAL"]
    healthy = [r for r in results if r.health_status == "HEALTHY"]
    hot = [r for r in results if r.health_status == "HOT"]

    tradeable = [r for r in results if r.tradeable]
    avoid = [r for r in results if r.recommendation == "AVOID"]

    # Top lists
    by_score = sorted(results, key=lambda x: x.health_score, reverse=True)
    by_volume = sorted(results, key=lambda x: x.volume_24h, reverse=True)
    by_change = sorted(results, key=lambda x: x.change_24h, reverse=True)

    return ScanSummary(
        timestamp=datetime.now().isoformat(),
        total_coins=len(results),
        dead_coins=len(dead),
        dying_coins=len(dying),
        weak_coins=len(weak),
        neutral_coins=len(neutral),
        healthy_coins=len(healthy),
        hot_coins=len(hot),
        tradeable_count=len(tradeable),
        avoid_count=len(avoid),
        top_healthy=[r.symbol for r in by_score[:10]],
        top_volume=[r.symbol for r in by_volume[:10]],
        biggest_gainers=[f"{r.symbol} (+{r.change_24h:.1f}%)" for r in by_change[:5]],
        biggest_losers=[f"{r.symbol} ({r.change_24h:.1f}%)" for r in by_change[-5:]],
        new_deaths=[],  # Would need history comparison
        recovering=[]   # Would need history comparison
    )


def save_results(results: List[CoinHealth], summary: ScanSummary):
    """Sla resultaten op als JSON."""

    # Convert to dict
    results_dict = {
        "scan_time": datetime.now().isoformat(),
        "summary": asdict(summary),
        "coins": {
            "dead": [asdict(r) for r in results if r.health_status == "DEAD"],
            "dying": [asdict(r) for r in results if r.health_status == "DYING"],
            "weak": [asdict(r) for r in results if r.health_status == "WEAK"],
            "neutral": [asdict(r) for r in results if r.health_status == "NEUTRAL"],
            "healthy": [asdict(r) for r in results if r.health_status == "HEALTHY"],
            "hot": [asdict(r) for r in results if r.health_status == "HOT"],
        },
        "tradeable_list": [r.symbol for r in results if r.tradeable],
        "avoid_list": [r.symbol for r in results if not r.tradeable],
        "all_coins": {r.symbol: asdict(r) for r in results}
    }

    # Save main file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Results saved to {OUTPUT_FILE}")

    # Append to history
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "summary": asdict(summary)
    }

    try:
        history = []
        if Path(HISTORY_FILE).exists():
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)

        history.append(history_entry)
        # Keep last 168 entries (1 week of hourly scans)
        history = history[-168:]

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except:
        pass


def print_results(results: List[CoinHealth], summary: ScanSummary, scan_time: float):
    """Print resultaten naar console."""

    print(f"\n{'═'*80}")
    print(f"{C.C}{C.BOLD}  📊 KUCOIN COIN HEALTH SCANNER{C.END}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Scan time: {scan_time:.1f}s")
    print(f"{'═'*80}")

    # ═══ SUMMARY ═══
    print(f"\n{C.Y}{C.BOLD}  📈 MARKET SUMMARY{C.END}")
    print(f"  {'─'*76}")

    total = summary.total_coins
    print(f"  Total coins scanned: {C.BOLD}{total}{C.END}")
    print()
    print(f"  {C.R}💀 DEAD:     {summary.dead_coins:>4}{C.END} ({summary.dead_coins/total*100:>5.1f}%)  │  "
          f"{C.G}✅ HEALTHY: {summary.healthy_coins:>4}{C.END} ({summary.healthy_coins/total*100:>5.1f}%)")
    print(f"  {C.R}📉 DYING:    {summary.dying_coins:>4}{C.END} ({summary.dying_coins/total*100:>5.1f}%)  │  "
          f"{C.G}🔥 HOT:     {summary.hot_coins:>4}{C.END} ({summary.hot_coins/total*100:>5.1f}%)")
    print(f"  {C.Y}⚠️  WEAK:     {summary.weak_coins:>4}{C.END} ({summary.weak_coins/total*100:>5.1f}%)  │  "
          f"{C.B}➖ NEUTRAL: {summary.neutral_coins:>4}{C.END} ({summary.neutral_coins/total*100:>5.1f}%)")
    print()
    print(f"  {C.G}Tradeable: {summary.tradeable_count}{C.END} | {C.R}Avoid: {summary.avoid_count}{C.END}")

    # ═══ DEAD/DYING COINS ═══
    dead_dying = [r for r in results if r.health_status in ["DEAD", "DYING"]]
    dead_dying.sort(key=lambda x: x.health_score)

    print(f"\n{C.R}{C.BOLD}  💀 DEAD & DYING COINS - AVOID THESE ({len(dead_dying)}){C.END}")
    print(f"  {'─'*76}")

    if dead_dying:
        print(f"  {'Symbol':<14} {'Status':<8} {'Score':>5} {'Volume':>12} {'24h':>8} {'Warnings':<30}")
        print(f"  {'─'*76}")

        for r in dead_dying[:20]:
            status_col = C.R if r.health_status == "DEAD" else C.Y
            vol_str = f"${r.volume_24h/1000:.0f}K" if r.volume_24h >= 1000 else f"${r.volume_24h:.0f}"
            warns = ", ".join(r.warnings[:2]) if r.warnings else "—"

            print(f"  {r.symbol:<14} {status_col}{r.health_status:<8}{C.END} "
                  f"{r.health_score:>5} {vol_str:>12} {r.change_24h:>+7.1f}% {C.DIM}{warns[:30]}{C.END}")

    # ═══ HEALTHY/HOT COINS ═══
    healthy_hot = [r for r in results if r.health_status in ["HEALTHY", "HOT"]]
    healthy_hot.sort(key=lambda x: x.health_score, reverse=True)

    print(f"\n{C.G}{C.BOLD}  ✅ HEALTHY & HOT COINS - TRADEABLE ({len(healthy_hot)}){C.END}")
    print(f"  {'─'*76}")

    if healthy_hot:
        print(f"  {'Symbol':<14} {'Status':<8} {'Score':>5} {'Volume':>12} {'24h':>8} {'Vol Trend':<15}")
        print(f"  {'─'*76}")

        for r in healthy_hot[:20]:
            status_col = C.G if r.health_status == "HOT" else C.B
            icon = "🔥" if r.health_status == "HOT" else "✅"
            vol_str = f"${r.volume_24h/1000000:.1f}M" if r.volume_24h >= 1000000 else f"${r.volume_24h/1000:.0f}K"

            print(f"  {r.symbol:<14} {status_col}{icon} {r.health_status:<6}{C.END} "
                  f"{r.health_score:>5} {vol_str:>12} {r.change_24h:>+7.1f}% {r.volume_trend:<15}")

    # ═══ TOP MOVERS ═══
    print(f"\n{C.Y}{C.BOLD}  📊 TOP MOVERS{C.END}")
    print(f"  {'─'*76}")
    print(f"  {C.G}Gainers:{C.END} {' | '.join(summary.biggest_gainers)}")
    print(f"  {C.R}Losers:{C.END}  {' | '.join(summary.biggest_losers)}")

    # ═══ RECOMMENDATIONS ═══
    print(f"\n{C.C}{C.BOLD}  💡 SCAN RECOMMENDATIONS{C.END}")
    print(f"  {'─'*76}")
    print(f"  • Focus op {C.G}HEALTHY{C.END} en {C.G}HOT{C.END} coins voor trading")
    print(f"  • {C.R}VERMIJD{C.END} coins met DEAD/DYING status")
    print(f"  • Check {C.Y}WEAK{C.END} coins alleen bij sterke setups")
    print(f"  • {C.BOLD}{len([r for r in results if r.tradeable and r.health_score >= 70])}{C.END} coins zijn optimaal voor trading (score >= 70)")

    print(f"\n{'═'*80}")
    print(f"  {C.DIM}Output saved to: {OUTPUT_FILE}{C.END}")
    print(f"{'═'*80}\n")


async def run_scan():
    """Run een volledige scan."""
    start_time = time.time()

    results, summary = await scan_all_coins()

    if not results:
        print("Scan failed!")
        return

    scan_time = time.time() - start_time

    # Save to JSON
    save_results(results, summary)

    # Print to console
    print_results(results, summary, scan_time)

    return results, summary


async def main():
    """Main loop."""
    print(f"""
{C.C}{C.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║              📊 KUCOIN COIN HEALTH SCANNER                                   ║
║                                                                              ║
║  Analyseert alle coins op "gezondheid" en classificeert:                     ║
║                                                                              ║
║  💀 DEAD   - Vrijwel geen volume, niet handelbaar                           ║
║  📉 DYING  - Sterk dalend volume, stervende coin                            ║
║  ⚠️  WEAK   - Laag volume, risicovol                                         ║
║  ➖ NEUTRAL - Gemiddeld, handelbaar met voorzichtigheid                       ║
║  ✅ HEALTHY - Goed volume, stabiel, handelbaar                               ║
║  🔥 HOT    - Hoog volume, actief, beste kansen                              ║
║                                                                              ║
║  Output: {OUTPUT_FILE:<67}║
║  Scan interval: {SCAN_INTERVAL/60:.0f} minuten                                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
{C.END}
""")

    scan_count = 0
    while True:
        try:
            scan_count += 1
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting scan #{scan_count}...")

            await run_scan()

            next_scan = datetime.now().timestamp() + SCAN_INTERVAL
            next_scan_str = datetime.fromtimestamp(next_scan).strftime('%H:%M:%S')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Next scan at {next_scan_str} (in {SCAN_INTERVAL//60} min)")

            await asyncio.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Scanner gestopt")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBeëindigd")
