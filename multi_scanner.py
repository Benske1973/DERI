"""
MULTI-SCANNER SYSTEM V1.0
=========================
Geavanceerd multi-coin scanning systeem met:
- Automatische coin discovery
- Adaptive settings per coin type
- Ranking systeem
- Real-time alerts
- Performance tracking
"""

import ccxt
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import from V7 scanner
from scanner_v7_sniper import Indicators, Config, SignalGenerator, Backtester


# ==========================================
# 1. COIN DISCOVERY & FILTERING
# ==========================================
class CoinDiscovery:
    """Automatisch coins ontdekken en filteren"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def get_usdt_pairs(self, min_volume_24h: float = 100000) -> List[dict]:
        """Haal alle USDT pairs met minimaal volume"""
        pairs = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if '/USDT:' in symbol:  # Skip futures
                    continue
                    
                volume = ticker.get('quoteVolume', 0) or 0
                price = ticker.get('last', 0) or 0
                
                if volume >= min_volume_24h and price > 0:
                    pairs.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': volume,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            # Sorteer op volume
            pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            
        except Exception as e:
            print(f"Error fetching tickers: {e}")
        
        return pairs
    
    def categorize_coins(self, pairs: List[dict]) -> Dict[str, List[str]]:
        """Categoriseer coins op basis van volume/market cap proxy"""
        categories = {
            'large_cap': [],    # Top 20 by volume
            'mid_cap': [],      # 21-100
            'small_cap': [],    # 101-300
            'micro_cap': [],    # 301+
        }
        
        for i, pair in enumerate(pairs):
            symbol = pair['symbol']
            if i < 20:
                categories['large_cap'].append(symbol)
            elif i < 100:
                categories['mid_cap'].append(symbol)
            elif i < 300:
                categories['small_cap'].append(symbol)
            else:
                categories['micro_cap'].append(symbol)
        
        return categories


# ==========================================
# 2. ADAPTIVE CONFIG PER COIN TYPE
# ==========================================
class AdaptiveConfig:
    """Genereer optimale config per coin type"""
    
    @staticmethod
    def for_large_cap(symbols: List[str]) -> Config:
        """Large caps: conservatief, trend-following"""
        return Config(
            symbols=symbols,
            timeframe='4h',
            sensitivity=3.0,          # Hogere threshold
            use_rsi_filter=True,
            rsi_threshold=50,
            use_trend_filter=True,
            trend_mode='follow',      # Alleen met de trend
            use_volume_filter=False,
            signal_cooldown=4,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=3.0,    # 2:1 R:R
        )
    
    @staticmethod
    def for_mid_cap(symbols: List[str]) -> Config:
        """Mid caps: balanced approach"""
        return Config(
            symbols=symbols,
            timeframe='4h',
            sensitivity=2.5,
            use_rsi_filter=True,
            rsi_threshold=45,
            use_trend_filter=True,
            trend_mode='both',
            use_volume_filter=False,
            signal_cooldown=3,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=3.5,
        )
    
    @staticmethod
    def for_small_cap(symbols: List[str]) -> Config:
        """Small caps: reversal hunting, meer risico"""
        return Config(
            symbols=symbols,
            timeframe='4h',
            sensitivity=2.0,          # Lagere threshold
            use_rsi_filter=True,
            rsi_threshold=40,         # Lager voor oversold bounces
            use_trend_filter=True,
            trend_mode='reversal',    # Zoek reversals
            use_volume_filter=False,
            signal_cooldown=2,
            atr_sl_multiplier=2.0,
            atr_tp_multiplier=4.0,    # 2:1 R:R
        )


# ==========================================
# 3. SIGNAL RANKER
# ==========================================
@dataclass
class RankedSignal:
    """Een gerankt trading signaal"""
    symbol: str
    signal_type: str  # BUY or SELL
    price: float
    entry: float
    stop_loss: float
    take_profit: float
    
    # Indicators
    rsi: float
    macd_hist: float
    trend: str
    volume_ratio: float
    
    # Score components
    momentum_score: float = 0.0
    trend_score: float = 0.0
    volume_score: float = 0.0
    risk_reward_score: float = 0.0
    
    # Backtest stats
    historical_wr: float = 0.0
    historical_pf: float = 0.0
    
    # Final score (0-100)
    total_score: float = 0.0
    
    timestamp: str = ""
    category: str = ""


class SignalRanker:
    """Rank signalen op kwaliteit"""
    
    @staticmethod
    def calculate_score(signal: RankedSignal) -> RankedSignal:
        """Bereken totale score voor een signaal"""
        
        # 1. Momentum Score (0-25)
        # RSI sweet spot: 50-70 voor buys
        if signal.signal_type == 'BUY':
            if 50 <= signal.rsi <= 65:
                signal.momentum_score = 25
            elif 40 <= signal.rsi < 50 or 65 < signal.rsi <= 75:
                signal.momentum_score = 15
            else:
                signal.momentum_score = 5
        else:  # SELL
            if 35 <= signal.rsi <= 50:
                signal.momentum_score = 25
            else:
                signal.momentum_score = 10
        
        # 2. Trend Score (0-25)
        if signal.signal_type == 'BUY':
            if signal.trend == 'UPTREND':
                signal.trend_score = 25  # Mee met de trend
            else:
                signal.trend_score = 15  # Reversal play
        else:
            if signal.trend == 'DOWNTREND':
                signal.trend_score = 25
            else:
                signal.trend_score = 15
        
        # 3. Volume Score (0-20)
        if signal.volume_ratio >= 2.0:
            signal.volume_score = 20
        elif signal.volume_ratio >= 1.5:
            signal.volume_score = 15
        elif signal.volume_ratio >= 1.0:
            signal.volume_score = 10
        else:
            signal.volume_score = 5
        
        # 4. Risk/Reward Score (0-15)
        rr = abs(signal.take_profit - signal.entry) / abs(signal.entry - signal.stop_loss) if signal.stop_loss != signal.entry else 0
        if rr >= 3:
            signal.risk_reward_score = 15
        elif rr >= 2:
            signal.risk_reward_score = 10
        else:
            signal.risk_reward_score = 5
        
        # 5. Historical Performance Score (0-15)
        if signal.historical_pf >= 2.0:
            hist_score = 15
        elif signal.historical_pf >= 1.5:
            hist_score = 10
        elif signal.historical_pf >= 1.0:
            hist_score = 5
        else:
            hist_score = 0
        
        # Total Score
        signal.total_score = (
            signal.momentum_score +
            signal.trend_score +
            signal.volume_score +
            signal.risk_reward_score +
            hist_score
        )
        
        return signal


# ==========================================
# 4. MULTI SCANNER ENGINE
# ==========================================
class MultiScanner:
    """Hoofdengine voor multi-coin scanning"""
    
    def __init__(self, exchange_id: str = 'kucoin', db_path: str = 'trading_bot.db'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.discovery = CoinDiscovery(exchange_id)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialiseer database tabellen"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Signals table
        c.execute('''
            CREATE TABLE IF NOT EXISTS scanner_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                category TEXT,
                signal_type TEXT,
                price REAL,
                entry REAL,
                stop_loss REAL,
                take_profit REAL,
                rsi REAL,
                trend TEXT,
                score REAL,
                status TEXT DEFAULT 'ACTIVE',
                result TEXT,
                pnl_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance tracking
        c.execute('''
            CREATE TABLE IF NOT EXISTS scanner_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                category TEXT,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_pnl REAL,
                avg_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Haal OHLCV data op"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            return None
    
    def scan_symbol(self, symbol: str, config: Config, category: str = 'unknown', 
                    lookback_candles: int = 1) -> Optional[RankedSignal]:
        """
        Scan een enkel symbool en return ranked signal als er een is.
        
        lookback_candles: Check de laatste N candles voor signalen (1 = alleen huidige)
        """
        
        df = self.fetch_ohlcv(symbol, config.timeframe)
        if df is None:
            return None
        
        # Genereer signalen
        signal_gen = SignalGenerator(config)
        df = signal_gen.generate_signals(df)
        
        # Check voor signaal in laatste N candles
        signal_found = False
        signal_idx = -1
        
        for i in range(-lookback_candles, 0):
            if df.iloc[i]['buy_signal'] or df.iloc[i]['sell_signal']:
                signal_found = True
                signal_idx = i
                break
        
        if not signal_found:
            return None
        
        last = df.iloc[signal_idx]
        
        # Backtest stats voor historische performance
        stats = Backtester.calculate_stats(df)
        
        # Maak ranked signal
        signal_type = 'BUY' if last['buy_signal'] else 'SELL'
        
        signal = RankedSignal(
            symbol=symbol,
            signal_type=signal_type,
            price=last['close'],
            entry=last['close'],
            stop_loss=last['buy_sl'] if signal_type == 'BUY' else last['sell_sl'],
            take_profit=last['buy_tp'] if signal_type == 'BUY' else last['sell_tp'],
            rsi=last['rsi'],
            macd_hist=last['hist'],
            trend='UPTREND' if last['uptrend'] else 'DOWNTREND',
            volume_ratio=last['volume'] / last['vol_sma'] if last['vol_sma'] > 0 else 1.0,
            historical_wr=stats['win_rate'],
            historical_pf=stats['profit_factor'],
            timestamp=str(last['timestamp']),
            category=category,
        )
        
        # Calculate score
        signal = SignalRanker.calculate_score(signal)
        
        return signal
    
    def scan_category(self, symbols: List[str], config: Config, category: str, 
                      max_workers: int = 5) -> List[RankedSignal]:
        """Scan een hele categorie parallel"""
        signals = []
        
        # Rate limiting - niet te snel
        for i, symbol in enumerate(symbols):
            try:
                signal = self.scan_symbol(symbol, config, category)
                if signal:
                    signals.append(signal)
                    print(f"  [!] {symbol}: {signal.signal_type} signal (Score: {signal.total_score:.0f})")
                
                # Rate limit
                if i % 10 == 0 and i > 0:
                    time.sleep(1)
                    
            except Exception as e:
                continue
        
        return signals
    
    def run_full_scan(self, min_volume: float = 500000, max_coins_per_category: int = 30) -> List[RankedSignal]:
        """Run een volledige scan over alle categorieën"""
        
        print("=" * 70)
        print("  MULTI-SCANNER - DISCOVERING COINS")
        print("=" * 70)
        
        # 1. Discover coins
        pairs = self.discovery.get_usdt_pairs(min_volume_24h=min_volume)
        categories = self.discovery.categorize_coins(pairs)
        
        print(f"  Found {len(pairs)} USDT pairs above ${min_volume:,.0f} volume")
        print(f"  Large Cap: {len(categories['large_cap'])} | Mid Cap: {len(categories['mid_cap'])} | Small Cap: {len(categories['small_cap'])}")
        
        all_signals = []
        
        # 2. Scan per category met aangepaste settings
        scan_configs = [
            ('large_cap', AdaptiveConfig.for_large_cap, categories['large_cap'][:max_coins_per_category]),
            ('mid_cap', AdaptiveConfig.for_mid_cap, categories['mid_cap'][:max_coins_per_category]),
            ('small_cap', AdaptiveConfig.for_small_cap, categories['small_cap'][:max_coins_per_category]),
        ]
        
        for cat_name, config_func, symbols in scan_configs:
            if not symbols:
                continue
                
            print(f"\n  Scanning {cat_name.upper()} ({len(symbols)} coins)...")
            config = config_func(symbols)
            
            signals = self.scan_category(symbols, config, cat_name)
            all_signals.extend(signals)
        
        # 3. Sort by score
        all_signals.sort(key=lambda x: x.total_score, reverse=True)
        
        # 4. Save to database
        self._save_signals(all_signals)
        
        return all_signals
    
    def _save_signals(self, signals: List[RankedSignal]):
        """Sla signalen op in database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for s in signals:
            c.execute('''
                INSERT INTO scanner_signals 
                (timestamp, symbol, category, signal_type, price, entry, stop_loss, 
                 take_profit, rsi, trend, score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
            ''', (s.timestamp, s.symbol, s.category, s.signal_type, s.price, 
                  s.entry, s.stop_loss, s.take_profit, s.rsi, s.trend, s.total_score))
        
        conn.commit()
        conn.close()
    
    def get_top_signals(self, limit: int = 10) -> pd.DataFrame:
        """Haal top signalen uit database"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f'''
            SELECT * FROM scanner_signals 
            WHERE status = 'ACTIVE' 
            ORDER BY score DESC, created_at DESC 
            LIMIT {limit}
        ''', conn)
        conn.close()
        return df
    
    def scan_near_breakouts(self, symbols: List[str], config: Config, 
                            threshold_pct: float = 0.8) -> List[dict]:
        """
        Vind coins die BIJNA een breakout signaal hebben.
        threshold_pct: 0.8 = histogram is 80% van de breakout threshold
        """
        near_breakouts = []
        
        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, config.timeframe)
                if df is None:
                    continue
                
                signal_gen = SignalGenerator(config)
                df = signal_gen.generate_signals(df)
                
                last = df.iloc[-1]
                
                # Check hoe dicht we bij een breakout zijn
                if last['hist'] > 0:
                    breakout_pct = last['hist'] / last['threshold'] if last['threshold'] > 0 else 0
                    
                    if threshold_pct <= breakout_pct < 1.0:
                        near_breakouts.append({
                            'symbol': symbol,
                            'breakout_pct': breakout_pct * 100,
                            'hist': last['hist'],
                            'threshold': last['threshold'],
                            'rsi': last['rsi'],
                            'trend': 'UPTREND' if last['uptrend'] else 'DOWNTREND',
                            'price': last['close'],
                        })
                        
            except Exception:
                continue
        
        # Sorteer op hoe dicht bij breakout
        near_breakouts.sort(key=lambda x: x['breakout_pct'], reverse=True)
        return near_breakouts
    
    def scan_recent_signals(self, symbols: List[str], config: Config, 
                           lookback_candles: int = 6) -> List[RankedSignal]:
        """
        Scan voor signalen in de laatste N candles.
        Handig om recente opportunities te vinden die je gemist hebt.
        """
        signals = []
        
        for symbol in symbols:
            try:
                signal = self.scan_symbol(symbol, config, 'recent', lookback_candles)
                if signal:
                    signals.append(signal)
            except Exception:
                continue
        
        signals.sort(key=lambda x: x.total_score, reverse=True)
        return signals


# ==========================================
# 5. ALERT FORMATTER
# ==========================================
class AlertFormatter:
    """Format alerts voor output"""
    
    @staticmethod
    def format_signal(signal: RankedSignal) -> str:
        """Format een signaal als text alert"""
        
        rr = abs(signal.take_profit - signal.entry) / abs(signal.entry - signal.stop_loss)
        
        alert = f"""
{'='*50}
{'BUY' if signal.signal_type == 'BUY' else 'SELL'} SIGNAL - {signal.symbol}
{'='*50}
Score:      {signal.total_score:.0f}/100 {'*' * int(signal.total_score/10)}
Category:   {signal.category.upper()}
Trend:      {signal.trend}

Entry:      ${signal.entry:.6f}
Stop Loss:  ${signal.stop_loss:.6f} ({abs((signal.stop_loss-signal.entry)/signal.entry*100):.1f}%)
Take Profit:${signal.take_profit:.6f} ({abs((signal.take_profit-signal.entry)/signal.entry*100):.1f}%)
Risk/Reward: 1:{rr:.1f}

RSI:        {signal.rsi:.1f}
Volume:     {signal.volume_ratio:.1f}x average
Hist WR:    {signal.historical_wr:.0f}%
Hist PF:    {signal.historical_pf:.2f}
{'='*50}
"""
        return alert
    
    @staticmethod
    def format_summary(signals: List[RankedSignal]) -> str:
        """Format summary van alle signalen"""
        
        if not signals:
            return "No signals found."
        
        lines = [
            "",
            "=" * 80,
            "  TOP TRADING OPPORTUNITIES (Ranked by Score)",
            "=" * 80,
            f"  {'#':<3} {'SYMBOL':<12} {'TYPE':<5} {'SCORE':<6} {'TREND':<10} {'RSI':<6} {'R:R':<6} {'CAT':<10}",
            "-" * 80,
        ]
        
        for i, s in enumerate(signals[:15], 1):
            rr = abs(s.take_profit - s.entry) / abs(s.entry - s.stop_loss) if s.stop_loss != s.entry else 0
            lines.append(
                f"  {i:<3} {s.symbol:<12} {s.signal_type:<5} {s.total_score:<6.0f} "
                f"{s.trend:<10} {s.rsi:<6.1f} 1:{rr:<4.1f} {s.category:<10}"
            )
        
        lines.append("=" * 80)
        
        # Stats
        buy_signals = [s for s in signals if s.signal_type == 'BUY']
        sell_signals = [s for s in signals if s.signal_type == 'SELL']
        avg_score = sum(s.total_score for s in signals) / len(signals)
        
        lines.append(f"  Total: {len(signals)} signals | BUY: {len(buy_signals)} | SELL: {len(sell_signals)} | Avg Score: {avg_score:.1f}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    import sys
    
    # Parse args
    mode = 'live'        # live, recent, near
    min_volume = 500000  # $500k minimum 24h volume
    max_coins = 30       # Max coins per category
    lookback = 6         # Candles to look back for recent mode
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    if len(sys.argv) > 2:
        try:
            min_volume = float(sys.argv[2])
        except:
            pass
    
    print("\n" + "=" * 70)
    print("  MULTI-SCANNER SYSTEM V1.0")
    print("=" * 70)
    print(f"  Mode: {mode.upper()}")
    print(f"  Min Volume: ${min_volume:,.0f}")
    print("=" * 70)
    print("  Usage: python multi_scanner.py [mode] [min_volume]")
    print("  Modes: live    - Only current candle signals")
    print("         recent  - Signals from last 6 candles (24h)")  
    print("         near    - Coins close to breakout (80%+)")
    print("=" * 70)
    
    # Initialize scanner
    scanner = MultiScanner()
    
    # Discover coins
    pairs = scanner.discovery.get_usdt_pairs(min_volume_24h=min_volume)
    symbols = [p['symbol'] for p in pairs[:100]]  # Top 100 by volume
    
    print(f"\n  Found {len(pairs)} coins, scanning top {len(symbols)}...")
    
    start_time = time.time()
    
    if mode == 'live':
        # Live signals - alleen huidige candle
        signals = scanner.run_full_scan(min_volume=min_volume, max_coins_per_category=max_coins)
        
        print(AlertFormatter.format_summary(signals))
        
        if signals:
            print("\n" + "=" * 70)
            print("  TOP 3 DETAILED SIGNALS")
            print("=" * 70)
            for signal in signals[:3]:
                print(AlertFormatter.format_signal(signal))
                
    elif mode == 'recent':
        # Recent signals - laatste N candles
        print(f"\n  Looking for signals in last {lookback} candles (24h)...")
        
        config = Config(
            symbols=symbols,
            timeframe='4h',
            sensitivity=2.5,
            use_rsi_filter=True,
            rsi_threshold=45,
            use_trend_filter=False,
            use_volume_filter=False,
            signal_cooldown=2,
        )
        
        signals = scanner.scan_recent_signals(symbols, config, lookback_candles=lookback)
        
        print(AlertFormatter.format_summary(signals))
        
        if signals:
            print("\n  RECENT SIGNALS DETAIL:")
            print("-" * 70)
            for s in signals[:10]:
                rr = abs(s.take_profit - s.entry) / abs(s.entry - s.stop_loss) if s.stop_loss != s.entry else 0
                print(f"  {s.symbol:<12} {s.signal_type:<5} Score:{s.total_score:<5.0f} "
                      f"Entry:${s.entry:<10.4f} RSI:{s.rsi:<5.1f} R:R 1:{rr:.1f}")
                      
    elif mode == 'near':
        # Near breakout - coins die bijna signaleren
        print(f"\n  Looking for coins at 70%+ of breakout threshold...")
        
        config = Config(
            symbols=symbols,
            timeframe='4h',
            sensitivity=2.5,
            use_rsi_filter=False,
            use_trend_filter=False,
            use_volume_filter=False,
        )
        
        near_breakouts = scanner.scan_near_breakouts(symbols, config, threshold_pct=0.70)
        
        if near_breakouts:
            print("\n" + "=" * 70)
            print("  COINS NEAR BREAKOUT (potential next signals)")
            print("=" * 70)
            print(f"  {'SYMBOL':<12} {'BREAKOUT%':<10} {'RSI':<8} {'TREND':<12} {'PRICE':<12}")
            print("-" * 70)
            
            for nb in near_breakouts[:20]:
                print(f"  {nb['symbol']:<12} {nb['breakout_pct']:<10.1f} {nb['rsi']:<8.1f} "
                      f"{nb['trend']:<12} ${nb['price']:<12.4f}")
            
            print("-" * 70)
            print(f"  Total: {len(near_breakouts)} coins near breakout")
        else:
            print("\n  No coins currently near breakout threshold.")
        
        signals = []
    
    else:
        print(f"  Unknown mode: {mode}")
        signals = []
    
    elapsed = time.time() - start_time
    print(f"\n  Scan completed in {elapsed:.1f} seconds")
    
    return signals


if __name__ == "__main__":
    main()
