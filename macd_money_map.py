"""
MACD MONEY MAP - The Complete 3-System Trading Strategy
========================================================
Based on the proven MACD methodology:
- System 1: TREND SYSTEM (catches big moves)
- System 2: REVERSAL SYSTEM (catches turning points)  
- System 3: CONFIRMATION SYSTEM (filters bad trades)

Author: Trading Bot
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sqlite3

# ==========================================
# CONFIGURATIE
# ==========================================

CONFIG = {
    # Exchange
    'exchange': 'kucoin',
    
    # Symbols to scan
    'symbols': [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 
        'DOT/USDT', 'LINK/USDT', 'POL/USDT', 'ADA/USDT',
        'XRP/USDT', 'DOGE/USDT', 'ATOM/USDT', 'UNI/USDT',
        'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT'
    ],
    
    # Timeframes (Triple Timeframe Stack: 4x multiplier)
    'timeframes': {
        'trend': '1d',      # Daily = Trend bias
        'setup': '4h',      # 4H = Setup signals
        'entry': '1h'       # 1H = Entry confirmation
    },
    
    # MACD Settings
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    
    # System 1: Trend System
    'trend_system': {
        'distance_threshold': 0.5,    # Minimum distance from zero line
        'wait_candles': 2             # Wait X candles after crossover
    },
    
    # System 2: Reversal System
    'reversal_system': {
        'divergence_lookback': 20,    # Bars to look back for divergence
        'min_divergence_bars': 3      # Minimum bars between peaks/troughs
    },
    
    # Risk Management
    'risk': {
        'risk_reward': 2.0,           # 2R target
        'partial_close': 0.5          # Close 50% at target
    }
}


# ==========================================
# INDICATOR CALCULATIONS
# ==========================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD indicator components
    Returns: DataFrame with macd_line, signal_line, histogram
    """
    df = df.copy()
    
    # MACD Line = Fast EMA - Slow EMA
    df['ema_fast'] = calculate_ema(df['close'], fast)
    df['ema_slow'] = calculate_ema(df['close'], slow)
    df['macd_line'] = df['ema_fast'] - df['ema_slow']
    
    # Signal Line = EMA of MACD Line
    df['signal_line'] = calculate_ema(df['macd_line'], signal)
    
    # Histogram = MACD Line - Signal Line
    df['histogram'] = df['macd_line'] - df['signal_line']
    
    # Previous values for crossover detection
    df['macd_prev'] = df['macd_line'].shift(1)
    df['signal_prev'] = df['signal_line'].shift(1)
    df['hist_prev'] = df['histogram'].shift(1)
    
    return df


def find_swing_points(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Find swing highs and lows for divergence detection"""
    df = df.copy()
    
    df['swing_high'] = False
    df['swing_low'] = False
    
    for i in range(lookback, len(df) - lookback):
        # Swing High: Higher than X bars on both sides
        if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
            df.loc[df.index[i], 'swing_high'] = True
        
        # Swing Low: Lower than X bars on both sides
        if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
            df.loc[df.index[i], 'swing_low'] = True
    
    return df


# ==========================================
# SYSTEM 1: TREND SYSTEM
# ==========================================

class TrendSystem:
    """
    System 1: The Trend System
    - Zero Line Rule: Above 0 = ONLY buys, Below 0 = ONLY sells
    - Distance Rule: Only crossovers far from zero (>0.5 or <-0.5)
    - Wait Rule: Wait 2-3 candles after crossover
    """
    
    def __init__(self, config: dict):
        self.distance_threshold = config['trend_system']['distance_threshold']
        self.wait_candles = config['trend_system']['wait_candles']
    
    def get_zero_line_bias(self, df: pd.DataFrame) -> str:
        """
        Zero Line Foundation: Determine trend bias
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        current_macd = df['macd_line'].iloc[-1]
        
        if current_macd > 0:
            return 'BULLISH'
        elif current_macd < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def detect_crossover(self, df: pd.DataFrame) -> Dict:
        """
        Detect MACD crossovers with Distance Rule
        Returns crossover info if valid, None otherwise
        """
        result = {
            'has_crossover': False,
            'type': None,
            'distance_valid': False,
            'wait_confirmed': False,
            'macd_value': None,
            'candles_since_cross': 0
        }
        
        # Find most recent crossover
        for i in range(len(df) - 1, max(0, len(df) - 20), -1):
            macd_curr = df['macd_line'].iloc[i]
            macd_prev = df['macd_line'].iloc[i-1]
            signal_curr = df['signal_line'].iloc[i]
            signal_prev = df['signal_line'].iloc[i-1]
            
            # Bullish Crossover: MACD crosses above Signal
            if macd_prev <= signal_prev and macd_curr > signal_curr:
                candles_since = len(df) - 1 - i
                result['has_crossover'] = True
                result['type'] = 'BULLISH'
                result['macd_value'] = macd_curr
                result['candles_since_cross'] = candles_since
                result['distance_valid'] = macd_curr > self.distance_threshold
                result['wait_confirmed'] = candles_since >= self.wait_candles
                break
            
            # Bearish Crossover: MACD crosses below Signal
            elif macd_prev >= signal_prev and macd_curr < signal_curr:
                candles_since = len(df) - 1 - i
                result['has_crossover'] = True
                result['type'] = 'BEARISH'
                result['macd_value'] = macd_curr
                result['candles_since_cross'] = candles_since
                result['distance_valid'] = macd_curr < -self.distance_threshold
                result['wait_confirmed'] = candles_since >= self.wait_candles
                break
        
        return result
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Full System 1 analysis"""
        bias = self.get_zero_line_bias(df)
        crossover = self.detect_crossover(df)
        
        # Generate signal only if bias matches crossover
        signal = None
        if crossover['has_crossover'] and crossover['distance_valid']:
            if bias == 'BULLISH' and crossover['type'] == 'BULLISH':
                signal = 'BUY' if crossover['wait_confirmed'] else 'BUY_PENDING'
            elif bias == 'BEARISH' and crossover['type'] == 'BEARISH':
                signal = 'SELL' if crossover['wait_confirmed'] else 'SELL_PENDING'
        
        return {
            'system': 'TREND',
            'bias': bias,
            'crossover': crossover,
            'signal': signal,
            'strength': 'STRONG' if crossover['distance_valid'] else 'WEAK'
        }


# ==========================================
# SYSTEM 2: REVERSAL SYSTEM
# ==========================================

class ReversalSystem:
    """
    System 2: The Reversal System
    - Divergence Detection: Price vs MACD disagreement
    - Histogram Confirmation: Flip, Shrinking Tower, Zero Bounce
    """
    
    def __init__(self, config: dict):
        self.lookback = config['reversal_system']['divergence_lookback']
        self.min_bars = config['reversal_system']['min_divergence_bars']
    
    def detect_divergence(self, df: pd.DataFrame) -> Dict:
        """
        Detect bullish and bearish divergence
        - Bearish: Price higher high + MACD lower high
        - Bullish: Price lower low + MACD higher low
        """
        result = {
            'has_divergence': False,
            'type': None,
            'price_points': [],
            'macd_points': []
        }
        
        df = find_swing_points(df, lookback=5)
        recent_df = df.tail(self.lookback)
        
        # Find swing highs for bearish divergence
        swing_highs = recent_df[recent_df['swing_high'] == True]
        if len(swing_highs) >= 2:
            last_two = swing_highs.tail(2)
            price_1, price_2 = last_two['high'].iloc[0], last_two['high'].iloc[1]
            macd_1, macd_2 = last_two['macd_line'].iloc[0], last_two['macd_line'].iloc[1]
            
            # Bearish Divergence: Higher price high, lower MACD high
            if price_2 > price_1 and macd_2 < macd_1:
                result['has_divergence'] = True
                result['type'] = 'BEARISH'
                result['price_points'] = [price_1, price_2]
                result['macd_points'] = [macd_1, macd_2]
                return result
        
        # Find swing lows for bullish divergence
        swing_lows = recent_df[recent_df['swing_low'] == True]
        if len(swing_lows) >= 2:
            last_two = swing_lows.tail(2)
            price_1, price_2 = last_two['low'].iloc[0], last_two['low'].iloc[1]
            macd_1, macd_2 = last_two['macd_line'].iloc[0], last_two['macd_line'].iloc[1]
            
            # Bullish Divergence: Lower price low, higher MACD low
            if price_2 < price_1 and macd_2 > macd_1:
                result['has_divergence'] = True
                result['type'] = 'BULLISH'
                result['price_points'] = [price_1, price_2]
                result['macd_points'] = [macd_1, macd_2]
                return result
        
        return result
    
    def detect_histogram_pattern(self, df: pd.DataFrame) -> Dict:
        """
        Detect histogram patterns:
        1. The Flip: First opposite color bar after series
        2. Shrinking Tower: Bars getting smaller
        3. Zero Bounce: Histogram bounces off zero
        """
        result = {
            'pattern': None,
            'direction': None,
            'strength': 0
        }
        
        hist = df['histogram'].tail(10).values
        
        if len(hist) < 5:
            return result
        
        # Pattern 1: The Flip
        # First green bar after red bars (bullish) or vice versa
        current = hist[-1]
        prev_bars = hist[-6:-1]
        
        if current > 0 and all(b < 0 for b in prev_bars[-3:]):
            result['pattern'] = 'FLIP'
            result['direction'] = 'BULLISH'
            result['strength'] = abs(current)
            return result
        
        if current < 0 and all(b > 0 for b in prev_bars[-3:]):
            result['pattern'] = 'FLIP'
            result['direction'] = 'BEARISH'
            result['strength'] = abs(current)
            return result
        
        # Pattern 2: Shrinking Tower
        # Bars getting progressively smaller
        abs_hist = [abs(h) for h in hist[-5:]]
        if all(abs_hist[i] > abs_hist[i+1] for i in range(len(abs_hist)-1)):
            result['pattern'] = 'SHRINKING_TOWER'
            result['direction'] = 'BEARISH' if hist[-1] > 0 else 'BULLISH'
            result['strength'] = abs_hist[0] - abs_hist[-1]
            return result
        
        # Pattern 3: Zero Bounce
        # Histogram approaches zero then bounces away
        approaching_zero = abs(hist[-3]) < abs(hist[-4])
        bouncing_away = abs(hist[-1]) > abs(hist[-2]) > abs(hist[-3])
        same_sign = (hist[-1] > 0) == (hist[-3] > 0)
        
        if approaching_zero and bouncing_away and same_sign:
            result['pattern'] = 'ZERO_BOUNCE'
            result['direction'] = 'BULLISH' if hist[-1] > 0 else 'BEARISH'
            result['strength'] = abs(hist[-1])
            return result
        
        return result
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """Full System 2 analysis"""
        divergence = self.detect_divergence(df)
        histogram = self.detect_histogram_pattern(df)
        
        # Signal requires divergence + histogram confirmation
        signal = None
        if divergence['has_divergence']:
            if histogram['pattern'] in ['FLIP', 'SHRINKING_TOWER']:
                if divergence['type'] == histogram['direction']:
                    signal = 'BUY' if divergence['type'] == 'BULLISH' else 'SELL'
        
        return {
            'system': 'REVERSAL',
            'divergence': divergence,
            'histogram': histogram,
            'signal': signal
        }


# ==========================================
# SYSTEM 3: CONFIRMATION SYSTEM
# ==========================================

class ConfirmationSystem:
    """
    System 3: The Confirmation System
    - Triple Timeframe Stack: All 3 TFs must agree
    - Price Action Confirmation: Signals at key levels
    """
    
    def __init__(self, config: dict):
        self.timeframes = config['timeframes']
    
    def check_timeframe_alignment(self, analyses: Dict[str, Dict]) -> Dict:
        """
        Check if all three timeframes agree
        - Daily: Trend bias (zero line)
        - 4H: Setup signal
        - 1H: Entry confirmation (histogram flip)
        """
        result = {
            'aligned': False,
            'direction': None,
            'trend_bias': None,
            'setup_signal': None,
            'entry_confirmed': False
        }
        
        # Get trend bias from daily
        if 'trend' in analyses:
            result['trend_bias'] = analyses['trend'].get('trend', {}).get('bias')
        
        # Get setup signal from 4H
        if 'setup' in analyses:
            trend_signal = analyses['setup'].get('trend', {}).get('signal')
            reversal_signal = analyses['setup'].get('reversal', {}).get('signal')
            result['setup_signal'] = trend_signal or reversal_signal
        
        # Get entry confirmation from 1H histogram
        if 'entry' in analyses:
            hist_pattern = analyses['entry'].get('reversal', {}).get('histogram', {})
            if hist_pattern.get('pattern') == 'FLIP':
                result['entry_confirmed'] = True
                result['entry_direction'] = hist_pattern.get('direction')
        
        # Check alignment
        if result['trend_bias'] and result['setup_signal'] and result['entry_confirmed']:
            # All must point same direction
            if result['trend_bias'] == 'BULLISH' and 'BUY' in str(result['setup_signal']):
                if result.get('entry_direction') == 'BULLISH':
                    result['aligned'] = True
                    result['direction'] = 'BUY'
            
            elif result['trend_bias'] == 'BEARISH' and 'SELL' in str(result['setup_signal']):
                if result.get('entry_direction') == 'BEARISH':
                    result['aligned'] = True
                    result['direction'] = 'SELL'
        
        return result
    
    def find_key_levels(self, df: pd.DataFrame, lookback: int = 50) -> Dict:
        """Find support and resistance levels"""
        recent = df.tail(lookback)
        
        # Simple S/R based on swing points
        swing_highs = recent[recent['high'] == recent['high'].rolling(10, center=True).max()]['high']
        swing_lows = recent[recent['low'] == recent['low'].rolling(10, center=True).min()]['low']
        
        current_price = df['close'].iloc[-1]
        
        # Find nearest support (below price)
        supports = swing_lows[swing_lows < current_price]
        nearest_support = supports.iloc[-1] if len(supports) > 0 else None
        
        # Find nearest resistance (above price)
        resistances = swing_highs[swing_highs > current_price]
        nearest_resistance = resistances.iloc[0] if len(resistances) > 0 else None
        
        return {
            'current_price': current_price,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'at_support': nearest_support and abs(current_price - nearest_support) / current_price < 0.01,
            'at_resistance': nearest_resistance and abs(current_price - nearest_resistance) / current_price < 0.01
        }


# ==========================================
# MAIN SCANNER
# ==========================================

class MACDMoneyMap:
    """
    Complete MACD Money Map Scanner
    Combines all 3 systems for A+ trade setups
    """
    
    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.exchange = getattr(ccxt, config['exchange'])()
        self.trend_system = TrendSystem(config)
        self.reversal_system = ReversalSystem(config)
        self.confirmation_system = ConfirmationSystem(config)
    
    def fetch_data(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from exchange"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate MACD
            df = calculate_macd(
                df, 
                self.config['macd']['fast'],
                self.config['macd']['slow'],
                self.config['macd']['signal']
            )
            
            return df
        except Exception as e:
            print(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Complete analysis of a symbol across all timeframes and systems
        """
        result = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'timeframes': {},
            'final_signal': None,
            'signal_type': None,
            'confidence': 0,
            'entry': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # Fetch data for all timeframes
        dfs = {}
        for tf_name, tf_value in self.config['timeframes'].items():
            df = self.fetch_data(symbol, tf_value)
            if df is not None:
                dfs[tf_name] = df
        
        if len(dfs) < 3:
            return result
        
        # Analyze each timeframe
        analyses = {}
        for tf_name, df in dfs.items():
            analyses[tf_name] = {
                'trend': self.trend_system.analyze(df),
                'reversal': self.reversal_system.analyze(df)
            }
            result['timeframes'][tf_name] = analyses[tf_name]
        
        # System 3: Check confirmation
        alignment = self.confirmation_system.check_timeframe_alignment(analyses)
        key_levels = self.confirmation_system.find_key_levels(dfs['entry'])
        
        result['alignment'] = alignment
        result['key_levels'] = key_levels
        
        # Generate final signal
        if alignment['aligned']:
            result['final_signal'] = alignment['direction']
            result['confidence'] = 90  # High confidence when all systems align
            
            # Calculate entry, SL, TP
            current_price = key_levels['current_price']
            
            if alignment['direction'] == 'BUY':
                result['entry'] = current_price
                result['stop_loss'] = key_levels['nearest_support'] if key_levels['nearest_support'] else current_price * 0.98
                risk = result['entry'] - result['stop_loss']
                result['take_profit'] = result['entry'] + (risk * self.config['risk']['risk_reward'])
                result['signal_type'] = 'TREND' if analyses['setup']['trend']['signal'] else 'REVERSAL'
                
                # Bonus confidence if at support
                if key_levels['at_support']:
                    result['confidence'] = 95
            
            elif alignment['direction'] == 'SELL':
                result['entry'] = current_price
                result['stop_loss'] = key_levels['nearest_resistance'] if key_levels['nearest_resistance'] else current_price * 1.02
                risk = result['stop_loss'] - result['entry']
                result['take_profit'] = result['entry'] - (risk * self.config['risk']['risk_reward'])
                result['signal_type'] = 'TREND' if analyses['setup']['trend']['signal'] else 'REVERSAL'
                
                # Bonus confidence if at resistance
                if key_levels['at_resistance']:
                    result['confidence'] = 95
        
        # Check for pending signals (waiting for confirmation)
        elif analyses['setup']['trend']['signal'] and 'PENDING' in str(analyses['setup']['trend']['signal']):
            result['final_signal'] = 'PENDING'
            result['confidence'] = 50
        
        return result
    
    def scan_all(self) -> List[Dict]:
        """Scan all symbols and return signals"""
        print(f"\n{'='*60}")
        print(f"MACD MONEY MAP SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}\n")
        
        results = []
        
        for symbol in self.config['symbols']:
            print(f"Scanning {symbol}...", end=" ")
            analysis = self.analyze_symbol(symbol)
            results.append(analysis)
            
            if analysis['final_signal'] and analysis['final_signal'] != 'PENDING':
                print(f"✅ {analysis['final_signal']} Signal! (Confidence: {analysis['confidence']}%)")
            elif analysis['final_signal'] == 'PENDING':
                print(f"⏳ Pending signal...")
            else:
                print("No signal")
        
        return results
    
    def get_actionable_signals(self, results: List[Dict]) -> List[Dict]:
        """Filter for actionable signals only"""
        return [r for r in results if r['final_signal'] in ['BUY', 'SELL'] and r['confidence'] >= 70]
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of all signals"""
        actionable = self.get_actionable_signals(results)
        
        print(f"\n{'='*60}")
        print("📊 SCAN SUMMARY")
        print(f"{'='*60}")
        print(f"Total symbols scanned: {len(results)}")
        print(f"Actionable signals: {len(actionable)}")
        
        if actionable:
            print(f"\n🎯 A+ SETUPS:")
            print("-" * 60)
            for signal in actionable:
                print(f"""
Symbol:     {signal['symbol']}
Signal:     {signal['final_signal']} ({signal['signal_type']})
Confidence: {signal['confidence']}%
Entry:      {signal['entry']:.6f}
Stop Loss:  {signal['stop_loss']:.6f}
Target:     {signal['take_profit']:.6f}
Risk/Reward: 1:{self.config['risk']['risk_reward']}
""")
        
        # Show pending signals
        pending = [r for r in results if r['final_signal'] == 'PENDING']
        if pending:
            print(f"\n⏳ PENDING (Wait for confirmation):")
            for p in pending:
                print(f"  - {p['symbol']}")
    
    def save_signals_to_db(self, results: List[Dict]):
        """Save signals to database"""
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        
        # Create table if not exists
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
        
        for result in results:
            if result['final_signal'] in ['BUY', 'SELL']:
                c.execute('''INSERT INTO macd_signals 
                             (symbol, signal, signal_type, confidence, entry_price, stop_loss, take_profit, trend_bias)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                          (result['symbol'], 
                           result['final_signal'],
                           result['signal_type'],
                           result['confidence'],
                           result['entry'],
                           result['stop_loss'],
                           result['take_profit'],
                           result['alignment'].get('trend_bias')))
        
        conn.commit()
        conn.close()
        print("\n💾 Signals saved to database")


# ==========================================
# DAILY ROUTINE
# ==========================================

def morning_scan():
    """
    The 5-Minute Morning Scan Routine
    Step 1: Check the Trend (Daily MACD bias)
    Step 2: Find the Setup (Crossovers or Divergence)
    Step 3: Confirm Everything (All TFs aligned)
    """
    scanner = MACDMoneyMap()
    results = scanner.scan_all()
    scanner.print_summary(results)
    scanner.save_signals_to_db(results)
    
    return scanner.get_actionable_signals(results)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    signals = morning_scan()
