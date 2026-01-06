"""
MACD MONEY MAP - MULTI-SCANNER
===============================
Real-time scanner voor ALLE KuCoin coins met >50k volume
Combineert WebSocket streaming met MACD Money Map analyse

Features:
- WebSocket streaming van alle coins
- Volume filter (>50k USDT)
- Real-time MACD analyse op bewegende coins
- Detectie van System 1, 2, 3 setups
- Alerts bij A+ setups
"""

import asyncio
import websockets
import requests
import json
import time
import sqlite3
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

# ==========================================
# CONFIGURATIE
# ==========================================

CONFIG = {
    # Volume filter
    'min_volume_24h': 50000,  # Minimum 50k USDT volume
    
    # WebSocket settings
    'ws_timeout': 120,
    'max_subscriptions_per_ws': 300,
    'batch_size': 50,
    
    # Scanner settings
    'price_change_threshold': 0.5,  # 0.5% change triggers analysis
    'window_seconds': 60,  # Look at price change over 60 seconds
    'scan_interval': 5,  # How often to check for setups (seconds)
    
    # MACD Settings
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    
    # Timeframes for analysis
    'timeframes': {
        'trend': '1d',
        'setup': '4h', 
        'entry': '1h'
    },
    
    # System 1 settings
    'trend_system': {
        'distance_threshold': 0.5,
        'wait_candles': 2
    },
    
    # Alerts
    'alert_cooldown': 300,  # Don't alert same coin within 5 minutes
}

# ==========================================
# GLOBAL STATE
# ==========================================

class ScannerState:
    def __init__(self):
        self.price_history: Dict[str, deque] = {}
        self.volume_24h: Dict[str, float] = {}
        self.last_analysis: Dict[str, float] = {}
        self.active_signals: Dict[str, dict] = {}
        self.stats = {
            'total_messages': 0,
            'coins_monitored': 0,
            'signals_found': 0,
            'last_update': 0
        }
        self.lock = threading.Lock()

state = ScannerState()

# ==========================================
# KUCOIN API FUNCTIONS
# ==========================================

def get_high_volume_symbols(min_volume: float = 50000) -> List[str]:
    """Get all USDT pairs with volume > min_volume"""
    try:
        # Get all symbols
        url = "https://api.kucoin.com/api/v1/symbols"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        symbols_data = r.json()['data']
        
        # Get 24h ticker data for volume
        ticker_url = "https://api.kucoin.com/api/v1/market/allTickers"
        r2 = requests.get(ticker_url, timeout=10)
        r2.raise_for_status()
        tickers = {t['symbol']: t for t in r2.json()['data']['ticker']}
        
        pairs = []
        for m in symbols_data:
            symbol = m['symbol']
            
            # Filter criteria
            if (
                m['quoteCurrency'] == 'USDT'
                and m['enableTrading']
                and not m.get('st', False)
                and not symbol.endswith('3L-USDT')
                and not symbol.endswith('3S-USDT')
                and not symbol.endswith('UP-USDT')
                and not symbol.endswith('DOWN-USDT')
                and not symbol.endswith('BULL-USDT')
                and not symbol.endswith('BEAR-USDT')
                and not m.get('callauctionIsEnabled', False)
            ):
                # Check volume
                ticker = tickers.get(symbol, {})
                vol_value = float(ticker.get('volValue', 0))
                
                if vol_value >= min_volume:
                    pairs.append(symbol)
                    state.volume_24h[symbol] = vol_value
        
        # Sort by volume (highest first)
        pairs.sort(key=lambda x: state.volume_24h.get(x, 0), reverse=True)
        
        return pairs
        
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 200) -> Optional[List]:
    """Fetch OHLCV data from KuCoin REST API"""
    try:
        # Convert symbol format: BTC-USDT -> BTC-USDT (already correct)
        # Convert timeframe: 4h -> 4hour, 1d -> 1day, 1h -> 1hour
        tf_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1hour', '2h': '2hour', '4h': '4hour', '6h': '6hour',
            '8h': '8hour', '12h': '12hour', '1d': '1day', '1w': '1week'
        }
        kucoin_tf = tf_map.get(timeframe, timeframe)
        
        url = f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol}&type={kucoin_tf}&limit={limit}"
        r = requests.get(url, timeout=10)
        
        if r.status_code != 200:
            return None
            
        data = r.json().get('data', [])
        if not data:
            return None
        
        # KuCoin returns: [time, open, close, high, low, volume, turnover]
        # Convert to standard format: [timestamp, open, high, low, close, volume]
        ohlcv = []
        for candle in reversed(data):  # KuCoin returns newest first
            ohlcv.append([
                int(candle[0]) * 1000,  # timestamp in ms
                float(candle[1]),  # open
                float(candle[3]),  # high
                float(candle[4]),  # low
                float(candle[2]),  # close
                float(candle[5])   # volume
            ])
        
        return ohlcv
        
    except Exception as e:
        return None


# ==========================================
# MACD CALCULATIONS (Fast version)
# ==========================================

def calc_ema(prices: List[float], period: int) -> List[float]:
    """Calculate EMA efficiently"""
    if len(prices) < period:
        return [None] * len(prices)
    
    multiplier = 2 / (period + 1)
    ema = [None] * (period - 1)
    
    # First EMA is SMA
    ema.append(sum(prices[:period]) / period)
    
    # Calculate rest
    for i in range(period, len(prices)):
        ema.append((prices[i] - ema[-1]) * multiplier + ema[-1])
    
    return ema


def analyze_macd_fast(ohlcv: List, config: dict = CONFIG) -> dict:
    """Fast MACD analysis without pandas"""
    if not ohlcv or len(ohlcv) < 50:
        return {'valid': False}
    
    closes = [c[4] for c in ohlcv]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    
    # Calculate MACD
    fast = config['macd']['fast']
    slow = config['macd']['slow']
    signal = config['macd']['signal']
    
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    
    if ema_fast[-1] is None or ema_slow[-1] is None:
        return {'valid': False}
    
    # MACD Line
    macd_line = []
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Signal Line
    macd_values = [m for m in macd_line if m is not None]
    signal_line = calc_ema(macd_values, signal)
    
    if not signal_line or signal_line[-1] is None:
        return {'valid': False}
    
    # Current values
    current_macd = macd_line[-1]
    current_signal = signal_line[-1]
    prev_macd = macd_line[-2] if len(macd_line) > 1 else current_macd
    prev_signal = signal_line[-2] if len(signal_line) > 1 else current_signal
    histogram = current_macd - current_signal
    prev_histogram = prev_macd - prev_signal if prev_signal else histogram
    
    # Zero Line Bias
    bias = 'BULLISH' if current_macd > 0 else 'BEARISH' if current_macd < 0 else 'NEUTRAL'
    
    # Crossover detection
    crossover = None
    crossover_valid = False
    threshold = config['trend_system']['distance_threshold']
    
    if prev_macd <= prev_signal and current_macd > current_signal:
        crossover = 'BULLISH'
        crossover_valid = current_macd > threshold
    elif prev_macd >= prev_signal and current_macd < current_signal:
        crossover = 'BEARISH'
        crossover_valid = current_macd < -threshold
    
    # Histogram pattern
    hist_pattern = None
    if histogram > 0 and prev_histogram < 0:
        hist_pattern = 'FLIP_BULLISH'
    elif histogram < 0 and prev_histogram > 0:
        hist_pattern = 'FLIP_BEARISH'
    
    # Find support/resistance
    recent_high = max(highs[-20:])
    recent_low = min(lows[-20:])
    current_price = closes[-1]
    
    at_support = (current_price - recent_low) / current_price < 0.02
    at_resistance = (recent_high - current_price) / current_price < 0.02
    
    return {
        'valid': True,
        'price': current_price,
        'macd': current_macd,
        'signal': current_signal,
        'histogram': histogram,
        'bias': bias,
        'crossover': crossover,
        'crossover_valid': crossover_valid,
        'hist_pattern': hist_pattern,
        'at_support': at_support,
        'at_resistance': at_resistance,
        'support': recent_low,
        'resistance': recent_high
    }


def full_macd_analysis(symbol: str) -> dict:
    """Complete MACD Money Map analysis for a symbol"""
    result = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'signal': None,
        'signal_type': None,
        'confidence': 0,
        'entry': None,
        'stop_loss': None,
        'take_profit': None,
        'details': {}
    }
    
    # Fetch data for all timeframes
    timeframes = CONFIG['timeframes']
    analyses = {}
    
    for tf_name, tf_value in timeframes.items():
        ohlcv = fetch_ohlcv(symbol, tf_value)
        if ohlcv:
            analyses[tf_name] = analyze_macd_fast(ohlcv)
        else:
            analyses[tf_name] = {'valid': False}
    
    result['details'] = analyses
    
    # Check if all timeframes have valid data
    if not all(a.get('valid', False) for a in analyses.values()):
        return result
    
    trend_analysis = analyses['trend']
    setup_analysis = analyses['setup']
    entry_analysis = analyses['entry']
    
    # SYSTEM 1: Check trend alignment
    trend_bias = trend_analysis['bias']
    setup_bias = setup_analysis['bias']
    
    # SYSTEM 2: Check for crossover or divergence
    has_setup = False
    setup_type = None
    
    if setup_analysis['crossover'] and setup_analysis['crossover_valid']:
        has_setup = True
        setup_type = 'TREND'
    
    # SYSTEM 3: Check entry confirmation
    entry_confirmed = entry_analysis['hist_pattern'] is not None
    
    # Generate signal
    if trend_bias == setup_bias and has_setup:
        if trend_bias == 'BULLISH' and setup_analysis['crossover'] == 'BULLISH':
            if entry_confirmed or entry_analysis['hist_pattern'] == 'FLIP_BULLISH':
                result['signal'] = 'BUY'
                result['signal_type'] = setup_type
                result['confidence'] = 85
                
                # Calculate levels
                result['entry'] = setup_analysis['price']
                result['stop_loss'] = setup_analysis['support']
                risk = result['entry'] - result['stop_loss']
                result['take_profit'] = result['entry'] + (risk * 2)
                
                if setup_analysis['at_support']:
                    result['confidence'] = 95
        
        elif trend_bias == 'BEARISH' and setup_analysis['crossover'] == 'BEARISH':
            if entry_confirmed or entry_analysis['hist_pattern'] == 'FLIP_BEARISH':
                result['signal'] = 'SELL'
                result['signal_type'] = setup_type
                result['confidence'] = 85
                
                result['entry'] = setup_analysis['price']
                result['stop_loss'] = setup_analysis['resistance']
                risk = result['stop_loss'] - result['entry']
                result['take_profit'] = result['entry'] - (risk * 2)
                
                if setup_analysis['at_resistance']:
                    result['confidence'] = 95
    
    return result


# ==========================================
# WEBSOCKET HANDLERS
# ==========================================

async def get_ws_token():
    """Get WebSocket token from KuCoin"""
    r = requests.post("https://api.kucoin.com/api/v1/bullet-public", timeout=10)
    r.raise_for_status()
    data = r.json()['data']
    return data['instanceServers'][0]['endpoint'] + "?token=" + data['token']


async def handle_websocket(ws_id: int, symbols: List[str]):
    """Handle single WebSocket connection"""
    reconnect_delay = 5
    
    while True:
        try:
            print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id}: Connecting for {len(symbols)} coins...")
            
            ws_url = await get_ws_token()
            
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            ) as ws:
                
                # Subscribe in batches
                batch_size = CONFIG['batch_size']
                batches = [symbols[i:i+batch_size] for i in range(0, len(symbols), batch_size)]
                
                for i, batch in enumerate(batches):
                    subscribe_msg = {
                        "id": f"{ws_id}_{int(time.time()*1000)}_{i}",
                        "type": "subscribe",
                        "topic": "/market/ticker:" + ",".join(batch),
                        "privateChannel": False,
                        "response": True
                    }
                    await ws.send(json.dumps(subscribe_msg))
                    await asyncio.sleep(0.3)
                
                print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id}: ✅ Subscribed to {len(symbols)} symbols")
                reconnect_delay = 5  # Reset on successful connection
                
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=CONFIG['ws_timeout'])
                        data = json.loads(msg)
                        
                        if data.get("type") == "message" and "data" in data:
                            now = time.time()
                            ticker_data = data.get("data")
                            topic = data.get("topic", "")
                            
                            if ":" in topic:
                                symbol = topic.split(":")[-1]
                            else:
                                continue
                            
                            if symbol and ticker_data and "price" in ticker_data:
                                price = float(ticker_data['price'])
                                
                                # Update price history
                                with state.lock:
                                    if symbol not in state.price_history:
                                        state.price_history[symbol] = deque(maxlen=1000)
                                    state.price_history[symbol].append((now, price))
                                    state.stats['total_messages'] += 1
                                    state.stats['last_update'] = now
                        
                    except asyncio.TimeoutError:
                        print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id}: Timeout, reconnecting...")
                        break
                    except websockets.exceptions.ConnectionClosed:
                        print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id}: Connection closed")
                        break
        
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id} error: {e}")
        
        print(f"[{time.strftime('%H:%M:%S')}] WS-{ws_id}: Reconnecting in {reconnect_delay}s...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 1.5, 60)


# ==========================================
# SIGNAL SCANNER
# ==========================================

def get_price_change(symbol: str, window_seconds: float) -> Optional[float]:
    """Calculate price change percentage over window"""
    with state.lock:
        if symbol not in state.price_history:
            return None
        
        history = state.price_history[symbol]
        if len(history) < 2:
            return None
        
        now = time.time()
        cutoff = now - window_seconds
        
        # Find oldest price in window
        oldest_price = None
        for ts, price in history:
            if ts >= cutoff:
                oldest_price = price
                break
        
        if oldest_price is None:
            return None
        
        newest_price = history[-1][1]
        return ((newest_price - oldest_price) / oldest_price) * 100


async def signal_scanner():
    """Scan for MACD signals on moving coins"""
    executor = ThreadPoolExecutor(max_workers=5)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] 🔍 Signal scanner started")
    
    while True:
        await asyncio.sleep(CONFIG['scan_interval'])
        
        now = time.time()
        coins_to_analyze = []
        
        # Find coins with significant price movement
        with state.lock:
            for symbol in state.price_history.keys():
                # Check cooldown
                last_check = state.last_analysis.get(symbol, 0)
                if now - last_check < 60:  # Don't re-analyze within 60 seconds
                    continue
                
                change = get_price_change(symbol, CONFIG['window_seconds'])
                if change is not None and abs(change) >= CONFIG['price_change_threshold']:
                    coins_to_analyze.append((symbol, change))
        
        if not coins_to_analyze:
            continue
        
        # Sort by absolute change (most volatile first)
        coins_to_analyze.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Analyze top movers
        for symbol, change in coins_to_analyze[:10]:  # Max 10 at a time
            try:
                state.last_analysis[symbol] = now
                
                # Run analysis in thread pool to not block
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(executor, full_macd_analysis, symbol)
                
                if result['signal']:
                    await handle_signal(result, change)
                    
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")


async def handle_signal(result: dict, price_change: float):
    """Handle a detected signal"""
    symbol = result['symbol']
    signal = result['signal']
    confidence = result['confidence']
    
    # Check cooldown
    now = time.time()
    last_alert = state.active_signals.get(symbol, {}).get('timestamp', 0)
    if now - last_alert < CONFIG['alert_cooldown']:
        return
    
    # Store signal
    state.active_signals[symbol] = {
        'signal': signal,
        'confidence': confidence,
        'timestamp': now,
        'entry': result['entry'],
        'stop_loss': result['stop_loss'],
        'take_profit': result['take_profit']
    }
    state.stats['signals_found'] += 1
    
    # Print alert
    vol = state.volume_24h.get(symbol, 0)
    direction = "🟢 LONG" if signal == 'BUY' else "🔴 SHORT"
    
    print(f"""
{'🚨' * 25}
╔══════════════════════════════════════════════════════════════╗
║              🎯 MACD MONEY MAP SIGNAL ALERT 🎯                ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol:      {symbol:<45} ║
║  Direction:   {direction:<45} ║
║  Confidence:  {str(confidence) + '%':<45} ║
║  Price Move:  {f'{price_change:+.2f}% (last {CONFIG["window_seconds"]}s)':<45} ║
║  24h Volume:  ${vol:,.0f} USDT{' ' * (37 - len(f'{vol:,.0f}'))} ║
╠══════════════════════════════════════════════════════════════╣
║  Entry:       {str(round(result['entry'], 8)) if result['entry'] else 'N/A':<45} ║
║  Stop Loss:   {str(round(result['stop_loss'], 8)) if result['stop_loss'] else 'N/A':<45} ║
║  Take Profit: {str(round(result['take_profit'], 8)) if result['take_profit'] else 'N/A':<45} ║
╠══════════════════════════════════════════════════════════════╣
║  Type:        {result['signal_type'] or 'N/A':<45} ║
║  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<45} ║
╚══════════════════════════════════════════════════════════════╝
{'🚨' * 25}
""")
    
    # Save to database
    save_signal_to_db(result)


def save_signal_to_db(result: dict):
    """Save signal to database"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS macd_signals 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      symbol TEXT,
                      signal TEXT,
                      signal_type TEXT,
                      confidence INTEGER,
                      entry_price REAL,
                      stop_loss REAL,
                      take_profit REAL,
                      trend_bias TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      status TEXT DEFAULT 'ACTIVE')''')
        
        trend_bias = result['details'].get('trend', {}).get('bias', 'N/A')
        
        c.execute('''INSERT INTO macd_signals 
                     (symbol, signal, signal_type, confidence, entry_price, stop_loss, take_profit, trend_bias)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (result['symbol'], result['signal'], result['signal_type'],
                   result['confidence'], result['entry'], result['stop_loss'],
                   result['take_profit'], trend_bias))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB error: {e}")


# ==========================================
# STATUS DISPLAY
# ==========================================

async def status_display():
    """Display scanner status periodically"""
    while True:
        await asyncio.sleep(30)  # Every 30 seconds
        
        with state.lock:
            active_coins = len(state.price_history)
            total_msgs = state.stats['total_messages']
            signals = state.stats['signals_found']
        
        # Find top movers
        movers = []
        for symbol in list(state.price_history.keys())[:100]:
            change = get_price_change(symbol, 60)
            if change is not None:
                movers.append((symbol, change))
        
        movers.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[{time.strftime('%H:%M:%S')}] 📊 STATUS: {active_coins} coins | {total_msgs:,} msgs | {signals} signals")
        
        if movers:
            top3 = movers[:3]
            bottom3 = movers[-3:]
            
            print(f"  📈 Top movers (1min): {', '.join([f'{s} +{c:.2f}%' for s, c in top3])}")
            print(f"  📉 Bottom:           {', '.join([f'{s} {c:.2f}%' for s, c in bottom3])}")


# ==========================================
# MAIN
# ==========================================

async def main():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MACD MONEY MAP - MULTI-SCANNER                       ║
║                                                              ║
║  Scanning ALL KuCoin coins with >50k volume                  ║
║  Using MACD Money Map 3-System Analysis                      ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Get high volume symbols
    print(f"[{time.strftime('%H:%M:%S')}] Fetching high volume symbols (>{CONFIG['min_volume_24h']/1000}k USDT)...")
    symbols = get_high_volume_symbols(CONFIG['min_volume_24h'])
    
    if not symbols:
        print("No symbols found!")
        return
    
    print(f"[{time.strftime('%H:%M:%S')}] Found {len(symbols)} coins with sufficient volume")
    print(f"[{time.strftime('%H:%M:%S')}] Top 10 by volume:")
    for s in symbols[:10]:
        vol = state.volume_24h.get(s, 0)
        print(f"  {s}: ${vol:,.0f}")
    
    state.stats['coins_monitored'] = len(symbols)
    
    # Split symbols across WebSocket connections
    max_per_ws = CONFIG['max_subscriptions_per_ws']
    chunks = [symbols[i:i+max_per_ws] for i in range(0, len(symbols), max_per_ws)]
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting {len(chunks)} WebSocket connections...")
    
    # Create tasks
    tasks = []
    
    # WebSocket tasks
    for i, chunk in enumerate(chunks):
        task = asyncio.create_task(handle_websocket(i, chunk))
        tasks.append(task)
    
    # Signal scanner task
    tasks.append(asyncio.create_task(signal_scanner()))
    
    # Status display task
    tasks.append(asyncio.create_task(status_display()))
    
    # Run all tasks
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Stopped by user")
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{time.strftime('%H:%M:%S')}] Scanner terminated")
