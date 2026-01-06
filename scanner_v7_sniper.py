"""
V7: The Ultimate Sniper - Verbeterde versie met:
- Volume filter (breakout moet met hoog volume)
- Trend filter (EMA 50/200 Golden/Death Cross)
- ATR-based stop-loss en take-profit
- Multi-symbol scanning
- Signaal cooldown (voorkom clustering)
- Vectorized code (sneller)
- Database integratie
- Backtesting statistieken
"""

import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

# ==========================================
# 1. CONFIGURATIE
# ==========================================
@dataclass
class Config:
    # Symbolen om te scannen
    symbols: list = None
    timeframe: str = '4h'
    
    # MACD
    fast_len: int = 12
    slow_len: int = 26
    sig_len: int = 9
    
    # Breakout
    lookback_period: int = 20
    sensitivity: float = 3.0
    
    # RSI
    use_rsi_filter: bool = True
    rsi_threshold: int = 50
    rsi_period: int = 14
    
    # NIEUW: Volume Filter
    use_volume_filter: bool = True
    volume_multiplier: float = 1.5  # Volume moet 1.5x hoger zijn dan gemiddeld
    
    # NIEUW: Trend Filter (EMA)
    use_trend_filter: bool = True
    ema_fast: int = 50
    ema_slow: int = 200
    
    # NIEUW: Trend Mode
    # 'follow' = alleen mee met trend (conservatief)
    # 'reversal' = zoek naar trend omkeringen (agressief)
    # 'both' = negeer trend filter
    trend_mode: str = 'follow'
    
    # NIEUW: ADX Trend Strength Filter
    use_adx_filter: bool = False
    adx_period: int = 14
    adx_threshold: int = 25  # ADX > 25 = sterke trend
    
    # NIEUW: RSI Divergence Detection
    use_divergence: bool = False
    
    # NIEUW: ATR voor SL/TP
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5  # Stop-loss op 1.5x ATR
    atr_tp_multiplier: float = 3.0  # Take-profit op 3x ATR (2:1 R:R)
    
    # NIEUW: Signaal Cooldown
    signal_cooldown: int = 3  # Minimaal 3 candles tussen signalen
    
    # Preset modes
    @classmethod
    def aggressive(cls, symbols: list = None):
        """Agressieve settings - meer signalen, risicovoller"""
        return cls(
            symbols=symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            sensitivity=2.0,
            use_trend_filter=False,
            use_volume_filter=False,
            signal_cooldown=2,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=4.0,
        )
    
    @classmethod
    def conservative(cls, symbols: list = None):
        """Conservatieve settings - minder signalen, veiliger"""
        return cls(
            symbols=symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            sensitivity=3.5,
            use_trend_filter=True,
            use_volume_filter=True,
            trend_mode='follow',
            signal_cooldown=5,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=3.0,
        )
    
    @classmethod
    def reversal_hunter(cls, symbols: list = None):
        """Reversal hunting - zoek naar trend omkeringen"""
        return cls(
            symbols=symbols or ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            sensitivity=2.5,
            use_trend_filter=True,
            trend_mode='reversal',
            use_volume_filter=True,
            volume_multiplier=2.0,  # Hoger volume voor reversals
            signal_cooldown=3,
            rsi_threshold=40,  # Lager voor oversold bounces
        )
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']


# ==========================================
# 2. INDICATOR BEREKENINGEN (VECTORIZED)
# ==========================================
class Indicators:
    @staticmethod
    def ema(source: pd.Series, length: int) -> pd.Series:
        return source.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def sma(source: pd.Series, length: int) -> pd.Series:
        return source.rolling(window=length).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index - meet trend sterkte"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = Indicators.atr(high, low, close, 1)  # True Range
        atr_val = tr.ewm(span=period, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ma_fast = Indicators.ema(close, fast)
        ma_slow = Indicators.ema(close, slow)
        macd_line = ma_fast - ma_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 14) -> tuple:
        """Detecteer bullish/bearish divergence"""
        bullish_div = pd.Series([False] * len(price), index=price.index)
        bearish_div = pd.Series([False] * len(price), index=price.index)
        
        for i in range(lookback, len(price)):
            # Bullish: Price lower low, RSI higher low
            price_window = price.iloc[i-lookback:i+1]
            ind_window = indicator.iloc[i-lookback:i+1]
            
            price_min_idx = price_window.idxmin()
            if price.iloc[i] <= price_window.min() * 1.01:  # Near low
                if indicator.iloc[i] > ind_window.loc[price_min_idx]:
                    bullish_div.iloc[i] = True
            
            # Bearish: Price higher high, RSI lower high  
            if price.iloc[i] >= price_window.max() * 0.99:  # Near high
                price_max_idx = price_window.idxmax()
                if indicator.iloc[i] < ind_window.loc[price_max_idx]:
                    bearish_div.iloc[i] = True
        
        return bullish_div, bearish_div


# ==========================================
# 3. SIGNAAL GENERATOR (VECTORIZED)
# ==========================================
class SignalGenerator:
    def __init__(self, config: Config):
        self.config = config
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereken alle indicatoren vectorized (geen loops!)"""
        cfg = self.config
        
        # MACD
        df['macd'], df['signal'], df['hist'] = Indicators.macd(
            df['close'], cfg.fast_len, cfg.slow_len, cfg.sig_len
        )
        
        # Lagged Volatility Baseline
        df['abs_hist'] = df['hist'].abs()
        df['baseline_noise'] = df['abs_hist'].rolling(window=cfg.lookback_period).mean().shift(1)
        df['baseline_noise'] = df['baseline_noise'].replace(0, 1e-8)
        df['threshold'] = df['baseline_noise'] * cfg.sensitivity
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], cfg.rsi_period)
        
        # ATR voor SL/TP
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], cfg.atr_period)
        
        # Trend EMAs
        df['ema_fast'] = Indicators.ema(df['close'], cfg.ema_fast)
        df['ema_slow'] = Indicators.ema(df['close'], cfg.ema_slow)
        df['uptrend'] = df['ema_fast'] > df['ema_slow']
        df['downtrend'] = df['ema_fast'] < df['ema_slow']
        
        # EMA Cross detectie (voor reversal mode)
        df['ema_cross_up'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
        df['ema_cross_down'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        
        # Prijs relatief tot EMAs
        df['price_above_ema_fast'] = df['close'] > df['ema_fast']
        df['price_above_ema_slow'] = df['close'] > df['ema_slow']
        
        # ADX (trend sterkte)
        if cfg.use_adx_filter:
            df['adx'], df['plus_di'], df['minus_di'] = Indicators.adx(
                df['high'], df['low'], df['close'], cfg.adx_period
            )
            df['strong_trend'] = df['adx'] > cfg.adx_threshold
        else:
            df['adx'] = 0
            df['strong_trend'] = True
        
        # RSI Divergence
        if cfg.use_divergence:
            df['bullish_div'], df['bearish_div'] = Indicators.detect_divergence(
                df['close'], df['rsi']
            )
        else:
            df['bullish_div'] = False
            df['bearish_div'] = False
        
        # Volume
        df['vol_sma'] = df['volume'].rolling(window=20).mean()
        df['high_volume'] = df['volume'] > (df['vol_sma'] * cfg.volume_multiplier)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genereer signalen volledig vectorized"""
        cfg = self.config
        
        # Bereken indicatoren
        df = self.calculate_indicators(df)
        
        # === BUY CONDITIONS (Vectorized) ===
        # Basis: Groene histogram + groeiend
        buy_base = (df['hist'] > 0) & (df['hist'] > df['hist'].shift(1))
        
        # Breakout: Histogram > threshold
        buy_breakout = df['hist'] > df['threshold']
        
        # RSI filter
        buy_rsi = (df['rsi'] > cfg.rsi_threshold) if cfg.use_rsi_filter else True
        
        # Trend filter - afhankelijk van mode
        if cfg.use_trend_filter:
            if cfg.trend_mode == 'follow':
                # Alleen kopen in uptrend
                buy_trend = df['uptrend']
            elif cfg.trend_mode == 'reversal':
                # Kopen bij potentiële trend reversal (downtrend maar prijs breekt uit)
                # Prijs moet boven snelle EMA komen vanuit downtrend
                buy_trend = df['downtrend'] & df['price_above_ema_fast']
            else:  # 'both'
                buy_trend = True
        else:
            buy_trend = True
        
        # Volume filter
        buy_volume = df['high_volume'] if cfg.use_volume_filter else True
        
        # ADX filter (alleen in sterke trends)
        buy_adx = df['strong_trend'] if cfg.use_adx_filter else True
        
        # Divergence bonus
        buy_div = df['bullish_div'] if cfg.use_divergence else True
        
        # Combineer alle filters
        if cfg.use_divergence:
            # Met divergence: basis condities OF divergence
            df['raw_buy'] = (buy_base & buy_breakout & buy_rsi & buy_trend & buy_volume & buy_adx) | \
                           (buy_div & buy_rsi & buy_volume)
        else:
            df['raw_buy'] = buy_base & buy_breakout & buy_rsi & buy_trend & buy_volume & buy_adx
        
        # === SELL CONDITIONS (Vectorized) ===
        sell_base = (df['hist'] < 0) & (df['hist'] < df['hist'].shift(1))
        sell_breakout = df['hist'] < -df['threshold']
        sell_rsi = (df['rsi'] < (100 - cfg.rsi_threshold)) if cfg.use_rsi_filter else True
        
        # Trend filter voor sells
        if cfg.use_trend_filter:
            if cfg.trend_mode == 'follow':
                sell_trend = df['downtrend']
            elif cfg.trend_mode == 'reversal':
                sell_trend = df['uptrend'] & ~df['price_above_ema_fast']
            else:
                sell_trend = True
        else:
            sell_trend = True
            
        sell_volume = df['high_volume'] if cfg.use_volume_filter else True
        sell_adx = df['strong_trend'] if cfg.use_adx_filter else True
        
        df['raw_sell'] = sell_base & sell_breakout & sell_rsi & sell_trend & sell_volume & sell_adx
        
        # === COOLDOWN FILTER ===
        df = self._apply_cooldown(df)
        
        # === BEREKEN SL/TP LEVELS ===
        df['buy_entry'] = np.where(df['buy_signal'], df['close'], np.nan)
        df['buy_sl'] = np.where(df['buy_signal'], 
                                df['close'] - (df['atr'] * cfg.atr_sl_multiplier), np.nan)
        df['buy_tp'] = np.where(df['buy_signal'], 
                                df['close'] + (df['atr'] * cfg.atr_tp_multiplier), np.nan)
        
        df['sell_entry'] = np.where(df['sell_signal'], df['close'], np.nan)
        df['sell_sl'] = np.where(df['sell_signal'], 
                                 df['close'] + (df['atr'] * cfg.atr_sl_multiplier), np.nan)
        df['sell_tp'] = np.where(df['sell_signal'], 
                                 df['close'] - (df['atr'] * cfg.atr_tp_multiplier), np.nan)
        
        return df
    
    def _apply_cooldown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Voorkom clustering van signalen"""
        cooldown = self.config.signal_cooldown
        
        # Buy cooldown
        buy_signals = []
        last_buy = -cooldown - 1
        for i, raw in enumerate(df['raw_buy']):
            if raw and (i - last_buy) > cooldown:
                buy_signals.append(True)
                last_buy = i
            else:
                buy_signals.append(False)
        df['buy_signal'] = buy_signals
        
        # Sell cooldown
        sell_signals = []
        last_sell = -cooldown - 1
        for i, raw in enumerate(df['raw_sell']):
            if raw and (i - last_sell) > cooldown:
                sell_signals.append(True)
                last_sell = i
            else:
                sell_signals.append(False)
        df['sell_signal'] = sell_signals
        
        return df


# ==========================================
# 4. BACKTESTER
# ==========================================
class Backtester:
    @staticmethod
    def calculate_stats(df: pd.DataFrame) -> dict:
        """Bereken backtest statistieken"""
        stats = {
            'total_buy_signals': df['buy_signal'].sum(),
            'total_sell_signals': df['sell_signal'].sum(),
            'wins': 0,
            'losses': 0,
            'total_pnl_pct': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
        }
        
        # Simuleer trades (simpele backtest)
        trades = []
        in_position = False
        entry_price = 0
        entry_sl = 0
        entry_tp = 0
        
        for i in range(len(df)):
            if not in_position and df['buy_signal'].iloc[i]:
                in_position = True
                entry_price = df['close'].iloc[i]
                entry_sl = df['buy_sl'].iloc[i]
                entry_tp = df['buy_tp'].iloc[i]
            
            elif in_position:
                current_price = df['close'].iloc[i]
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                
                # Check TP hit
                if high >= entry_tp:
                    pnl = ((entry_tp - entry_price) / entry_price) * 100
                    trades.append({'pnl': pnl, 'result': 'win'})
                    in_position = False
                # Check SL hit
                elif low <= entry_sl:
                    pnl = ((entry_sl - entry_price) / entry_price) * 100
                    trades.append({'pnl': pnl, 'result': 'loss'})
                    in_position = False
        
        if trades:
            wins = [t for t in trades if t['result'] == 'win']
            losses = [t for t in trades if t['result'] == 'loss']
            
            stats['wins'] = len(wins)
            stats['losses'] = len(losses)
            stats['total_pnl_pct'] = sum(t['pnl'] for t in trades)
            stats['win_rate'] = (len(wins) / len(trades)) * 100 if trades else 0
            stats['avg_win'] = np.mean([t['pnl'] for t in wins]) if wins else 0
            stats['avg_loss'] = np.mean([t['pnl'] for t in losses]) if losses else 0
            
            total_wins = sum(t['pnl'] for t in wins) if wins else 0
            total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 1
            stats['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
            
            # Max drawdown
            equity = [100]
            for t in trades:
                equity.append(equity[-1] * (1 + t['pnl']/100))
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak * 100
                if dd > max_dd:
                    max_dd = dd
            stats['max_drawdown'] = max_dd
        
        return stats


# ==========================================
# 5. DATABASE MANAGER
# ==========================================
class DatabaseManager:
    def __init__(self, db_path: str = 'trading_bot.db'):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS v7_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                timeframe TEXT,
                signal_type TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                rsi REAL,
                macd_hist REAL,
                volume_ratio REAL,
                trend TEXT,
                status TEXT DEFAULT 'ACTIVE',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_signal(self, symbol: str, timeframe: str, signal_type: str,
                    entry: float, sl: float, tp: float, rsi: float,
                    hist: float, vol_ratio: float, trend: str, timestamp: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO v7_signals 
            (timestamp, symbol, timeframe, signal_type, entry_price, stop_loss, 
             take_profit, rsi, macd_hist, volume_ratio, trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, symbol, timeframe, signal_type, entry, sl, tp, 
              rsi, hist, vol_ratio, trend))
        conn.commit()
        conn.close()
    
    def get_recent_signals(self, limit: int = 20) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            f"SELECT * FROM v7_signals ORDER BY created_at DESC LIMIT {limit}", 
            conn
        )
        conn.close()
        return df


# ==========================================
# 6. SCANNER
# ==========================================
class SniperScanner:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.exchange = ccxt.kucoin()
        self.signal_gen = SignalGenerator(self.config)
        self.db = DatabaseManager()
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Haal OHLCV data op"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, self.config.timeframe, limit=500)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def scan_symbol(self, symbol: str, save_to_db: bool = True) -> dict:
        """Scan een enkel symbool"""
        print(f"🔍 Scanning {symbol}...")
        
        df = self.fetch_data(symbol)
        if df is None:
            return {'symbol': symbol, 'error': 'No data'}
        
        # Genereer signalen
        df = self.signal_gen.generate_signals(df)
        
        # Laatste rij voor huidige status
        last = df.iloc[-1]
        
        # Check voor nieuw signaal
        new_signal = None
        if last['buy_signal']:
            new_signal = 'BUY'
            if save_to_db:
                self.db.save_signal(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    signal_type='BUY',
                    entry=last['close'],
                    sl=last['buy_sl'],
                    tp=last['buy_tp'],
                    rsi=last['rsi'],
                    hist=last['hist'],
                    vol_ratio=last['volume'] / last['vol_sma'] if last['vol_sma'] > 0 else 0,
                    trend='UPTREND' if last['uptrend'] else 'DOWNTREND',
                    timestamp=str(last['timestamp'])
                )
        elif last['sell_signal']:
            new_signal = 'SELL'
            if save_to_db:
                self.db.save_signal(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    signal_type='SELL',
                    entry=last['close'],
                    sl=last['sell_sl'],
                    tp=last['sell_tp'],
                    rsi=last['rsi'],
                    hist=last['hist'],
                    vol_ratio=last['volume'] / last['vol_sma'] if last['vol_sma'] > 0 else 0,
                    trend='UPTREND' if last['uptrend'] else 'DOWNTREND',
                    timestamp=str(last['timestamp'])
                )
        
        # Backtest stats
        stats = Backtester.calculate_stats(df)
        
        return {
            'symbol': symbol,
            'price': last['close'],
            'rsi': last['rsi'],
            'macd_hist': last['hist'],
            'trend': 'UPTREND' if last['uptrend'] else 'DOWNTREND',
            'new_signal': new_signal,
            'df': df,
            'stats': stats
        }
    
    def scan_all(self, save_to_db: bool = True) -> list:
        """Scan alle symbolen"""
        results = []
        for symbol in self.config.symbols:
            result = self.scan_symbol(symbol, save_to_db)
            results.append(result)
            if result.get('new_signal'):
                print(f"  🎯 {result['new_signal']} SIGNAL @ {result['price']:.4f}")
            else:
                print(f"  ✓ No signal (RSI: {result.get('rsi', 0):.1f}, Trend: {result.get('trend', 'N/A')})")
        return results
    
    def plot_analysis(self, symbol: str, save_path: str = None) -> str:
        """Maak analyse grafiek"""
        result = self.scan_symbol(symbol, save_to_db=False)
        if 'error' in result:
            return None
        
        df = result['df']
        stats = result['stats']
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
        fig.suptitle(f'{symbol} - V7 Ultimate Sniper Analysis', fontsize=14, fontweight='bold')
        
        # === PLOT 1: PRIJS + SIGNALEN + TREND ===
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['close'], label='Prijs', color='#787b86', linewidth=1)
        ax1.plot(df['timestamp'], df['ema_fast'], label=f'EMA {self.config.ema_fast}', 
                 color='#2196F3', linewidth=1, alpha=0.7)
        ax1.plot(df['timestamp'], df['ema_slow'], label=f'EMA {self.config.ema_slow}', 
                 color='#FF9800', linewidth=1, alpha=0.7)
        
        # Trend fill
        ax1.fill_between(df['timestamp'], df['close'].min(), df['close'].max(),
                        where=df['uptrend'], color='green', alpha=0.05, label='Uptrend')
        ax1.fill_between(df['timestamp'], df['close'].min(), df['close'].max(),
                        where=df['downtrend'], color='red', alpha=0.05, label='Downtrend')
        
        # Signalen
        buy_df = df[df['buy_signal']]
        sell_df = df[df['sell_signal']]
        
        ax1.scatter(buy_df['timestamp'], buy_df['close'] * 0.98, 
                   color='#00E676', marker='^', s=200, label='Buy', zorder=5, edgecolors='black')
        ax1.scatter(sell_df['timestamp'], sell_df['close'] * 1.02, 
                   color='#FF5252', marker='v', s=200, label='Sell', zorder=5, edgecolors='black')
        
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.1)
        
        # === PLOT 2: MACD + THRESHOLD ===
        ax2 = axes[1]
        conditions = [
            (df['hist'] >= 0) & (df['hist'] > df['hist'].shift(1)),
            (df['hist'] >= 0) & (df['hist'] <= df['hist'].shift(1)),
            (df['hist'] < 0) & (df['hist'] > df['hist'].shift(1)), 
            (df['hist'] < 0) & (df['hist'] <= df['hist'].shift(1))
        ]
        colors = ['#26a69a', '#b2dfdb', '#ffcdd2', '#ff5252']
        bar_colors = np.select(conditions, colors, default='#b2dfdb')
        
        ax2.bar(df['timestamp'], df['hist'], color=bar_colors, width=0.1)
        ax2.plot(df['timestamp'], df['threshold'], color='#e91e63', 
                linewidth=1, label='Breakout Threshold')
        ax2.plot(df['timestamp'], -df['threshold'], color='#e91e63', linewidth=1)
        ax2.fill_between(df['timestamp'], -df['threshold'], df['threshold'], 
                        color='#e91e63', alpha=0.05)
        ax2.legend(loc='upper left')
        ax2.set_ylabel('MACD Histogram')
        ax2.grid(True, alpha=0.1)
        
        # === PLOT 3: RSI ===
        ax3 = axes[2]
        ax3.plot(df['timestamp'], df['rsi'], color='#9c27b0', label='RSI')
        ax3.axhline(50, color='gray', linestyle='--', alpha=0.7)
        ax3.axhline(30, color='green', linestyle=':', alpha=0.5)
        ax3.axhline(70, color='red', linestyle=':', alpha=0.5)
        ax3.fill_between(df['timestamp'], 50, df['rsi'], 
                        where=(df['rsi'] >= 50), color='#4CAF50', alpha=0.1)
        ax3.fill_between(df['timestamp'], 50, df['rsi'], 
                        where=(df['rsi'] < 50), color='#f44336', alpha=0.1)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.1)
        
        # === PLOT 4: VOLUME ===
        ax4 = axes[3]
        vol_colors = np.where(df['high_volume'], '#2196F3', '#90CAF9')
        ax4.bar(df['timestamp'], df['volume'], color=vol_colors, width=0.1, alpha=0.7)
        ax4.plot(df['timestamp'], df['vol_sma'] * self.config.volume_multiplier, 
                color='#FF5722', linewidth=1, label=f'Volume Threshold ({self.config.volume_multiplier}x)')
        ax4.set_ylabel('Volume')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.1)
        
        # Stats text box (ASCII only voor betere compatibiliteit)
        stats_text = (
            f"BACKTEST STATS\n"
            f"-----------------\n"
            f"Buy Signals: {stats['total_buy_signals']}\n"
            f"Wins: {stats['wins']} | Losses: {stats['losses']}\n"
            f"Win Rate: {stats['win_rate']:.1f}%\n"
            f"Profit Factor: {stats['profit_factor']:.2f}\n"
            f"Total P&L: {stats['total_pnl_pct']:+.1f}%\n"
            f"Max DD: {stats['max_drawdown']:.1f}%"
        )
        fig.text(0.88, 0.92, stats_text, fontsize=9, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        if save_path is None:
            save_path = f'v7_analysis_{symbol.replace("/", "_")}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path


# ==========================================
# 7. MAIN
# ==========================================
def main():
    import sys
    
    # Parse command line args voor preset selectie
    preset = 'default'
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT']
    
    if len(sys.argv) > 1:
        preset = sys.argv[1].lower()
    
    # Selecteer configuratie op basis van preset
    if preset == 'aggressive':
        config = Config.aggressive(symbols)
        mode_name = "AGGRESSIVE"
    elif preset == 'conservative':
        config = Config.conservative(symbols)
        mode_name = "CONSERVATIVE"
    elif preset == 'reversal':
        config = Config.reversal_hunter(symbols)
        mode_name = "REVERSAL HUNTER"
    else:
        # Default balanced config
        config = Config(
            symbols=symbols,
            timeframe='4h',
            use_rsi_filter=True,
            use_trend_filter=True,
            use_volume_filter=False,  # Standaard uit (veel kleine coins hebben lage volume)
            trend_mode='both',        # Zowel trend als reversal
            sensitivity=2.5,
            signal_cooldown=3,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=3.0,
        )
        mode_name = "BALANCED"
    
    # Scanner
    scanner = SniperScanner(config)
    
    print("=" * 60)
    print(f"  V7 ULTIMATE SNIPER SCANNER - {mode_name} MODE")
    print("=" * 60)
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Sensitivity: {config.sensitivity}x")
    print(f"  Trend Mode: {config.trend_mode}")
    print(f"  Filters: RSI={config.use_rsi_filter}, Trend={config.use_trend_filter}, Vol={config.use_volume_filter}")
    print(f"  Risk/Reward: 1:{config.atr_tp_multiplier/config.atr_sl_multiplier:.1f}")
    print("=" * 60)
    print("  Usage: python scanner_v7_sniper.py [aggressive|conservative|reversal]")
    print("=" * 60)
    
    # Scan alle symbolen
    results = scanner.scan_all(save_to_db=True)
    
    print("\n" + "=" * 60)
    print("  SCAN RESULTS")
    print("=" * 60)
    
    for r in results:
        if 'error' not in r:
            signal_str = f">>> {r['new_signal']} <<<" if r['new_signal'] else "Waiting..."
            print(f"  {r['symbol']:12} | {r['trend']:10} | RSI: {r['rsi']:5.1f} | {signal_str}")
            
            # Genereer chart voor symbolen met signaal
            if r['new_signal']:
                chart_path = scanner.plot_analysis(r['symbol'])
                print(f"               -> Chart: {chart_path}")
    
    # Toon backtest summary
    print("\n" + "=" * 60)
    print("  BACKTEST SUMMARY (historical performance)")
    print("=" * 60)
    
    total_signals = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0
    
    for r in results:
        if 'error' not in r and 'stats' in r:
            s = r['stats']
            total_signals += s['total_buy_signals']
            total_wins += s['wins']
            total_losses += s['losses']
            total_pnl += s['total_pnl_pct']
            
            status = "OK" if s['profit_factor'] >= 1.0 else "--"
            print(f"  {r['symbol']:12} | Signals: {s['total_buy_signals']:3} | "
                  f"WR: {s['win_rate']:5.1f}% | PF: {s['profit_factor']:4.2f} | "
                  f"P&L: {s['total_pnl_pct']:+6.1f}% | {status}")
    
    # Totaal summary
    if total_signals > 0:
        overall_wr = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        print("-" * 60)
        print(f"  {'TOTAL':12} | Signals: {total_signals:3} | "
              f"WR: {overall_wr:5.1f}% | Wins: {total_wins} | Losses: {total_losses} | "
              f"P&L: {total_pnl:+6.1f}%")
    
    # Toon recente signalen uit database
    print("\n" + "=" * 60)
    print("  RECENT SIGNALS (from database)")
    print("=" * 60)
    
    recent = scanner.db.get_recent_signals(10)
    if not recent.empty:
        for _, row in recent.iterrows():
            print(f"  {row['created_at'][:16]} | {row['symbol']:12} | {row['signal_type']:4} | "
                  f"Entry: {row['entry_price']:.4f} | SL: {row['stop_loss']:.4f} | TP: {row['take_profit']:.4f}")
    else:
        print("  No signals in database yet")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    main()
