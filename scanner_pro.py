"""
SCANNER PRO V1.0 - Support/Resistance + Momentum
=================================================
- Support/Resistance level detectie
- "Room to Run" analyse - hoeveel ruimte tot weerstand
- Alleen coins die BEWEGEN (momentum filter)
- Live signals only
"""

import ccxt
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import time

# ==========================================
# 1. SUPPORT & RESISTANCE DETECTOR
# ==========================================
class SRDetector:
    """Detecteer Support en Resistance levels"""
    
    @staticmethod
    def find_pivot_points(df: pd.DataFrame, left: int = 5, right: int = 5) -> Tuple[List[float], List[float]]:
        """
        Vind pivot highs en lows (swing points).
        left/right: aantal candles links/rechts die lager/hoger moeten zijn
        """
        highs = df['high'].values
        lows = df['low'].values
        
        resistance_levels = []
        support_levels = []
        
        for i in range(left, len(df) - right):
            # Check voor pivot high (resistance)
            is_pivot_high = True
            for j in range(1, left + 1):
                if highs[i] <= highs[i - j]:
                    is_pivot_high = False
                    break
            for j in range(1, right + 1):
                if highs[i] <= highs[i + j]:
                    is_pivot_high = False
                    break
            
            if is_pivot_high:
                resistance_levels.append(highs[i])
            
            # Check voor pivot low (support)
            is_pivot_low = True
            for j in range(1, left + 1):
                if lows[i] >= lows[i - j]:
                    is_pivot_low = False
                    break
            for j in range(1, right + 1):
                if lows[i] >= lows[i + j]:
                    is_pivot_low = False
                    break
            
            if is_pivot_low:
                support_levels.append(lows[i])
        
        return resistance_levels, support_levels
    
    @staticmethod
    def cluster_levels(levels: List[float], threshold_pct: float = 0.02) -> List[float]:
        """
        Cluster nearby levels samen (binnen threshold_pct van elkaar).
        Returns gemiddelde van elke cluster.
        """
        if not levels:
            return []
        
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            # Als level binnen threshold% van cluster gemiddelde
            cluster_avg = np.mean(current_cluster)
            if abs(level - cluster_avg) / cluster_avg <= threshold_pct:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    @staticmethod
    def find_key_levels(df: pd.DataFrame, num_levels: int = 5) -> Dict:
        """
        Vind de belangrijkste S/R levels.
        Returns dict met support en resistance levels.
        """
        # Vind pivot points met verschillende sensitivities
        all_resistance = []
        all_support = []
        
        for left, right in [(3, 3), (5, 5), (10, 10)]:
            r, s = SRDetector.find_pivot_points(df, left, right)
            all_resistance.extend(r)
            all_support.extend(s)
        
        # Voeg recente highs/lows toe
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        all_resistance.append(recent_high)
        all_support.append(recent_low)
        
        # Cluster levels
        resistance = SRDetector.cluster_levels(all_resistance)
        support = SRDetector.cluster_levels(all_support)
        
        # Filter levels die te ver weg zijn (>50% van huidige prijs)
        current_price = df['close'].iloc[-1]
        resistance = [r for r in resistance if r > current_price and r < current_price * 1.5]
        support = [s for s in support if s < current_price and s > current_price * 0.5]
        
        # Sorteer en neem top N
        resistance = sorted(resistance)[:num_levels]
        support = sorted(support, reverse=True)[:num_levels]
        
        return {
            'resistance': resistance,
            'support': support,
            'current_price': current_price,
        }
    
    @staticmethod
    def calculate_room_to_run(current_price: float, resistance_levels: List[float], 
                              support_levels: List[float]) -> Dict:
        """
        Bereken hoeveel "room to run" er is.
        - Voor LONG: afstand tot eerste resistance
        - Voor SHORT: afstand tot eerste support
        """
        # Eerste resistance boven huidige prijs
        next_resistance = None
        for r in sorted(resistance_levels):
            if r > current_price * 1.005:  # Minimaal 0.5% boven prijs
                next_resistance = r
                break
        
        # Eerste support onder huidige prijs
        next_support = None
        for s in sorted(support_levels, reverse=True):
            if s < current_price * 0.995:  # Minimaal 0.5% onder prijs
                next_support = s
                break
        
        # Bereken percentages
        room_up = ((next_resistance - current_price) / current_price * 100) if next_resistance else 100
        room_down = ((current_price - next_support) / current_price * 100) if next_support else 100
        
        return {
            'next_resistance': next_resistance,
            'next_support': next_support,
            'room_up_pct': room_up,
            'room_down_pct': room_down,
            'bias': 'LONG' if room_up > room_down else 'SHORT',
        }


# ==========================================
# 2. MOMENTUM FILTER
# ==========================================
class MomentumFilter:
    """Filter voor coins die daadwerkelijk bewegen"""
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> Dict:
        """Bereken momentum/volatiliteit metrics"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ATR% - Average True Range als percentage van prijs
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100
        
        # Price change laatste 24h (6 candles van 4h)
        change_24h = ((close.iloc[-1] - close.iloc[-7]) / close.iloc[-7]) * 100 if len(close) > 7 else 0
        
        # Price change laatste 7 dagen (42 candles)
        change_7d = ((close.iloc[-1] - close.iloc[-43]) / close.iloc[-43]) * 100 if len(close) > 43 else 0
        
        # Volume spike - huidige volume vs gemiddelde
        vol_avg = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1
        
        # Volatility score (0-100)
        # Gebaseerd op ATR% - hogere ATR = meer beweging
        volatility_score = min(100, atr_pct * 20)  # 5% ATR = score 100
        
        # Momentum score (0-100)
        # Gebaseerd op recente price changes
        momentum_score = min(100, abs(change_24h) * 5)  # 20% move = score 100
        
        # Combined "Movement Score"
        movement_score = (volatility_score * 0.6) + (momentum_score * 0.4)
        
        return {
            'atr_pct': atr_pct,
            'change_24h': change_24h,
            'change_7d': change_7d,
            'vol_ratio': vol_ratio,
            'volatility_score': volatility_score,
            'momentum_score': momentum_score,
            'movement_score': movement_score,
        }
    
    @staticmethod
    def is_moving(metrics: Dict, min_score: float = 30) -> bool:
        """Check of coin genoeg beweegt"""
        return metrics['movement_score'] >= min_score


# ==========================================
# 3. ENHANCED SIGNAL
# ==========================================
@dataclass
class ProSignal:
    """Enhanced signal met S/R analyse"""
    symbol: str
    signal_type: str  # BUY or SELL
    
    # Price levels
    entry: float
    stop_loss: float
    take_profit: float
    
    # S/R Levels
    next_resistance: float
    next_support: float
    room_up_pct: float
    room_down_pct: float
    
    # Indicators
    rsi: float
    macd_hist: float
    trend: str
    
    # Momentum
    atr_pct: float
    change_24h: float
    movement_score: float
    vol_ratio: float
    
    # Scores
    sr_score: float = 0.0      # S/R quality score
    momentum_score: float = 0.0
    total_score: float = 0.0
    
    # Meta
    timestamp: str = ""
    
    def calculate_scores(self):
        """Bereken alle scores"""
        
        # S/R Score (0-40): Hoeveel room to run?
        if self.signal_type == 'BUY':
            # Voor buys: meer room up = beter
            if self.room_up_pct >= 15:
                self.sr_score = 40
            elif self.room_up_pct >= 10:
                self.sr_score = 30
            elif self.room_up_pct >= 5:
                self.sr_score = 20
            else:
                self.sr_score = 10
        else:
            # Voor sells: meer room down = beter
            if self.room_down_pct >= 15:
                self.sr_score = 40
            elif self.room_down_pct >= 10:
                self.sr_score = 30
            elif self.room_down_pct >= 5:
                self.sr_score = 20
            else:
                self.sr_score = 10
        
        # Momentum Score (0-30): Beweegt de coin?
        self.momentum_score = min(30, self.movement_score * 0.3)
        
        # RSI Score (0-20): Sweet spot?
        if self.signal_type == 'BUY':
            if 45 <= self.rsi <= 65:
                rsi_score = 20
            elif 35 <= self.rsi < 45 or 65 < self.rsi <= 75:
                rsi_score = 10
            else:
                rsi_score = 5
        else:
            if 35 <= self.rsi <= 55:
                rsi_score = 20
            else:
                rsi_score = 10
        
        # Volume Score (0-10)
        vol_score = min(10, self.vol_ratio * 5)
        
        # Total Score
        self.total_score = self.sr_score + self.momentum_score + rsi_score + vol_score


# ==========================================
# 4. PRO SCANNER
# ==========================================
class ProScanner:
    """Hoofdscanner met S/R en momentum analyse"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '4h', limit: int = 500) -> Optional[pd.DataFrame]:
        """Haal OHLCV data op"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def get_moving_coins(self, min_volume: float = 500000, min_movement: float = 30, 
                         max_coins: int = 50) -> List[Dict]:
        """Vind coins die bewegen"""
        
        moving_coins = []
        
        try:
            tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs met volume
            candidates = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT') or '/USDT:' in symbol:
                    continue
                
                volume = ticker.get('quoteVolume', 0) or 0
                if volume >= min_volume:
                    candidates.append({
                        'symbol': symbol,
                        'price': ticker.get('last', 0),
                        'volume': volume,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            # Sorteer op 24h change (absolute value) - meest bewegende eerst
            candidates.sort(key=lambda x: abs(x['change_24h']), reverse=True)
            
            # Neem top movers
            for coin in candidates[:max_coins * 2]:  # Check meer, filter later
                df = self.fetch_ohlcv(coin['symbol'], '4h', limit=100)
                if df is None:
                    continue
                
                metrics = MomentumFilter.calculate_metrics(df)
                
                if MomentumFilter.is_moving(metrics, min_movement):
                    coin['metrics'] = metrics
                    moving_coins.append(coin)
                    
                    if len(moving_coins) >= max_coins:
                        break
                
                time.sleep(0.05)  # Rate limit
                
        except Exception as e:
            print(f"Error: {e}")
        
        return moving_coins
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame = None) -> Optional[Dict]:
        """Volledige analyse van een symbol"""
        
        if df is None:
            df = self.fetch_ohlcv(symbol, '4h', limit=500)
        if df is None:
            return None
        
        # S/R Levels
        sr_levels = SRDetector.find_key_levels(df)
        room = SRDetector.calculate_room_to_run(
            sr_levels['current_price'],
            sr_levels['resistance'],
            sr_levels['support']
        )
        
        # Momentum
        momentum = MomentumFilter.calculate_metrics(df)
        
        # Indicators
        # EMA
        df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_fast'] - df['ema_slow']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['hist'] = df['macd'] - df['signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        # Trend
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        uptrend = df['ema_50'].iloc[-1] > df['ema_200'].iloc[-1]
        
        # Breakout threshold
        df['abs_hist'] = df['hist'].abs()
        df['baseline'] = df['abs_hist'].rolling(20).mean().shift(1)
        threshold = df['baseline'].iloc[-1] * 2.5
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        return {
            'symbol': symbol,
            'price': last['close'],
            'sr_levels': sr_levels,
            'room': room,
            'momentum': momentum,
            'rsi': last['rsi'],
            'hist': last['hist'],
            'prev_hist': prev['hist'],
            'threshold': threshold,
            'trend': 'UPTREND' if uptrend else 'DOWNTREND',
            'vol_ratio': last['volume'] / df['volume'].rolling(20).mean().iloc[-1],
            'timestamp': str(last['timestamp']),
            'df': df,
        }
    
    def check_signal(self, analysis: Dict) -> Optional[ProSignal]:
        """Check of er een signaal is"""
        
        hist = analysis['hist']
        prev_hist = analysis['prev_hist']
        threshold = analysis['threshold']
        rsi = analysis['rsi']
        room = analysis['room']
        momentum = analysis['momentum']
        
        signal_type = None
        
        # BUY Signal
        # Room to run moet groter zijn dan room to support (betere R:R)
        good_rr_setup = room['room_up_pct'] > room['room_down_pct'] * 1.5
        
        if (hist > 0 and 
            hist > prev_hist and 
            hist > threshold and
            rsi > 40 and rsi < 75 and
            room['room_up_pct'] >= 5 and  # Minimaal 5% room
            good_rr_setup):
            signal_type = 'BUY'
        
        # SELL Signal  
        good_rr_short = room['room_down_pct'] > room['room_up_pct'] * 1.5
        
        if (hist < 0 and 
              hist < prev_hist and 
              hist < -threshold and
              rsi < 60 and rsi > 25 and
              room['room_down_pct'] >= 5 and
              good_rr_short):
            signal_type = 'SELL'
        
        if not signal_type:
            return None
        
        # Calculate SL/TP based on S/R
        price = analysis['price']
        atr = momentum['atr_pct'] * price / 100
        
        if signal_type == 'BUY':
            # SL onder support of 1.5x ATR
            sl_sr = room['next_support'] if room['next_support'] else price * 0.95
            sl_atr = price - (atr * 1.5)
            stop_loss = max(sl_sr * 0.995, sl_atr)  # Iets onder support
            
            # TP bij resistance of 2:1 R:R
            risk = price - stop_loss
            tp_rr = price + (risk * 2.5)
            tp_sr = room['next_resistance'] if room['next_resistance'] else price * 1.15
            take_profit = min(tp_sr * 0.995, tp_rr)  # Iets onder resistance
        else:
            # SELL
            sl_sr = room['next_resistance'] if room['next_resistance'] else price * 1.05
            sl_atr = price + (atr * 1.5)
            stop_loss = min(sl_sr * 1.005, sl_atr)
            
            risk = stop_loss - price
            tp_rr = price - (risk * 2.5)
            tp_sr = room['next_support'] if room['next_support'] else price * 0.85
            take_profit = max(tp_sr * 1.005, tp_rr)
        
        signal = ProSignal(
            symbol=analysis['symbol'],
            signal_type=signal_type,
            entry=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            next_resistance=room['next_resistance'] or price * 1.1,
            next_support=room['next_support'] or price * 0.9,
            room_up_pct=room['room_up_pct'],
            room_down_pct=room['room_down_pct'],
            rsi=rsi,
            macd_hist=hist,
            trend=analysis['trend'],
            atr_pct=momentum['atr_pct'],
            change_24h=momentum['change_24h'],
            movement_score=momentum['movement_score'],
            vol_ratio=analysis['vol_ratio'],
            timestamp=analysis['timestamp'],
        )
        
        signal.calculate_scores()
        return signal
    
    def scan_live(self, min_volume: float = 500000, min_movement: float = 25,
                  max_coins: int = 50) -> Tuple[List[ProSignal], List[Dict]]:
        """Scan voor live signalen op bewegende coins"""
        
        signals = []
        watchlist = []  # Coins die bijna signaleren
        
        print(f"\n  Finding coins with movement score >= {min_movement}...")
        moving_coins = self.get_moving_coins(min_volume, min_movement, max_coins)
        print(f"  Found {len(moving_coins)} moving coins")
        
        print(f"\n  Analyzing for signals...")
        
        for coin in moving_coins:
            try:
                analysis = self.analyze_symbol(coin['symbol'])
                if analysis:
                    signal = self.check_signal(analysis)
                    if signal:
                        signals.append(signal)
                        print(f"    [!] {signal.symbol}: {signal.signal_type} (Score: {signal.total_score:.0f})")
                    else:
                        # Check of coin bijna signaleert OF recent breakout had
                        hist = analysis['hist']
                        prev_hist = analysis['prev_hist']
                        threshold = analysis['threshold']
                        room = analysis['room']
                        
                        if hist > 0 and threshold > 0:
                            breakout_pct = (hist / threshold) * 100
                            
                            # Recent breakout: boven threshold maar momentum dalend
                            is_recent_breakout = (hist > threshold and hist < prev_hist and
                                                 room['room_up_pct'] >= 5)
                            
                            if breakout_pct >= 50 or is_recent_breakout:
                                watchlist.append({
                                    'symbol': analysis['symbol'],
                                    'breakout_pct': breakout_pct,
                                    'rsi': analysis['rsi'],
                                    'room_up': analysis['room']['room_up_pct'],
                                    'trend': analysis['trend'],
                                    'change_24h': analysis['momentum']['change_24h'],
                                    'movement': analysis['momentum']['movement_score'],
                                    'price': analysis['price'],
                                    'status': 'RECENT BREAKOUT' if is_recent_breakout else 'APPROACHING',
                                })
            except:
                continue
        
        # Sort
        signals.sort(key=lambda x: x.total_score, reverse=True)
        watchlist.sort(key=lambda x: x['breakout_pct'], reverse=True)
        
        return signals, watchlist


# ==========================================
# 5. OUTPUT FORMATTER
# ==========================================
class ProFormatter:
    """Format output voor PRO scanner"""
    
    @staticmethod
    def format_signal(s: ProSignal) -> str:
        """Format een signaal als gedetailleerde alert"""
        
        risk = abs(s.entry - s.stop_loss)
        reward = abs(s.take_profit - s.entry)
        rr = reward / risk if risk > 0 else 0
        
        sl_pct = abs(s.stop_loss - s.entry) / s.entry * 100
        tp_pct = abs(s.take_profit - s.entry) / s.entry * 100
        
        return f"""
{'='*60}
  {s.signal_type} SIGNAL - {s.symbol}
{'='*60}
  SCORE: {s.total_score:.0f}/100 {'*' * int(s.total_score/10)}
  
  ENTRY:       ${s.entry:<12.6f}
  STOP LOSS:   ${s.stop_loss:<12.6f} ({sl_pct:.1f}%)
  TAKE PROFIT: ${s.take_profit:<12.6f} ({tp_pct:.1f}%)
  RISK/REWARD: 1:{rr:.1f}
  
  --- SUPPORT/RESISTANCE ---
  Next Resistance: ${s.next_resistance:.6f} ({s.room_up_pct:.1f}% away)
  Next Support:    ${s.next_support:.6f} ({s.room_down_pct:.1f}% away)
  Room to Run:     {s.room_up_pct:.1f}% up / {s.room_down_pct:.1f}% down
  
  --- MOMENTUM ---
  Movement Score:  {s.movement_score:.0f}/100
  24h Change:      {s.change_24h:+.1f}%
  ATR:             {s.atr_pct:.2f}%
  Volume:          {s.vol_ratio:.1f}x average
  
  --- INDICATORS ---
  RSI:             {s.rsi:.1f}
  Trend:           {s.trend}
  MACD Hist:       {s.macd_hist:.6f}
{'='*60}
"""
    
    @staticmethod
    def format_summary(signals: List[ProSignal], moving_coins: int = 0) -> str:
        """Format summary tabel"""
        
        lines = [
            "",
            "=" * 90,
            "  LIVE SIGNALS - MOVING COINS ONLY",
            "=" * 90,
        ]
        
        if not signals:
            lines.append("  No live signals at this moment.")
            lines.append("  The scanner only shows signals when breakout conditions are met.")
            lines.append("=" * 90)
            return "\n".join(lines)
        
        lines.append(f"  {'#':<3} {'SYMBOL':<14} {'TYPE':<5} {'SCORE':<6} {'ENTRY':<12} {'ROOM':<8} {'RSI':<6} {'24H':<8} {'TREND':<10}")
        lines.append("-" * 90)
        
        for i, s in enumerate(signals, 1):
            room = s.room_up_pct if s.signal_type == 'BUY' else s.room_down_pct
            lines.append(
                f"  {i:<3} {s.symbol:<14} {s.signal_type:<5} {s.total_score:<6.0f} "
                f"${s.entry:<11.4f} {room:<7.1f}% {s.rsi:<6.1f} {s.change_24h:<+7.1f}% {s.trend:<10}"
            )
        
        lines.append("=" * 90)
        
        # Stats
        buy_signals = [s for s in signals if s.signal_type == 'BUY']
        sell_signals = [s for s in signals if s.signal_type == 'SELL']
        avg_score = sum(s.total_score for s in signals) / len(signals)
        avg_room = sum(s.room_up_pct if s.signal_type == 'BUY' else s.room_down_pct for s in signals) / len(signals)
        
        lines.append(f"  Total: {len(signals)} signals | BUY: {len(buy_signals)} | SELL: {len(sell_signals)}")
        lines.append(f"  Avg Score: {avg_score:.0f} | Avg Room to Run: {avg_room:.1f}%")
        lines.append("=" * 90)
        
        return "\n".join(lines)


# ==========================================
# 6. MAIN
# ==========================================
def main():
    import sys
    
    min_volume = 500000   # $500k min volume
    min_movement = 25     # Minimum movement score
    max_coins = 60        # Max coins to scan
    
    if len(sys.argv) > 1:
        try:
            min_volume = float(sys.argv[1])
        except:
            pass
    if len(sys.argv) > 2:
        try:
            min_movement = float(sys.argv[2])
        except:
            pass
    
    print("\n" + "=" * 70)
    print("  SCANNER PRO - Support/Resistance + Momentum")
    print("=" * 70)
    print(f"  Min Volume:     ${min_volume:,.0f}")
    print(f"  Min Movement:   {min_movement} (score 0-100)")
    print(f"  Mode:           LIVE SIGNALS ONLY")
    print("=" * 70)
    print("  Usage: python scanner_pro.py [min_volume] [min_movement]")
    print("=" * 70)
    
    scanner = ProScanner()
    
    start_time = time.time()
    signals, watchlist = scanner.scan_live(
        min_volume=min_volume,
        min_movement=min_movement,
        max_coins=max_coins
    )
    elapsed = time.time() - start_time
    
    # Print summary
    print(ProFormatter.format_summary(signals))
    
    # Print detailed signals
    if signals:
        print("\n" + "=" * 70)
        print("  DETAILED SIGNAL ANALYSIS")
        print("=" * 70)
        
        for signal in signals[:5]:  # Top 5
            print(ProFormatter.format_signal(signal))
    
    # Print watchlist
    if watchlist:
        print("\n" + "=" * 100)
        print("  WATCHLIST - Moving coins approaching breakout OR recent breakouts")
        print("=" * 100)
        print(f"  {'SYMBOL':<14} {'STATUS':<16} {'BREAKOUT%':<10} {'RSI':<6} {'ROOM UP':<9} {'24H':<8} {'TREND':<10}")
        print("-" * 100)
        
        # Separate recent breakouts from approaching
        recent = [w for w in watchlist if w.get('status') == 'RECENT BREAKOUT']
        approaching = [w for w in watchlist if w.get('status') != 'RECENT BREAKOUT']
        
        # Show recent breakouts first
        for w in recent[:5]:
            print(f"  {w['symbol']:<14} {'>>> BREAKOUT <<<':<16} {w['breakout_pct']:<10.0f} {w['rsi']:<6.1f} "
                  f"{w['room_up']:<8.1f}% {w['change_24h']:<+7.1f}% {w['trend']:<10}")
        
        if recent and approaching:
            print("-" * 100)
        
        for w in approaching[:10]:
            print(f"  {w['symbol']:<14} {'Approaching':<16} {w['breakout_pct']:<10.0f} {w['rsi']:<6.1f} "
                  f"{w['room_up']:<8.1f}% {w['change_24h']:<+7.1f}% {w['trend']:<10}")
        
        print("=" * 100)
        print(f"  {len(recent)} recent breakouts | {len(approaching)} approaching")
    
    print(f"\n  Scan completed in {elapsed:.1f} seconds")
    
    return signals, watchlist


if __name__ == "__main__":
    main()
