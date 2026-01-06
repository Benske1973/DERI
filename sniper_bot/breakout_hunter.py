"""
BREAKOUT HUNTER - Real-time Explosive Mover Scanner
====================================================
Zoekt constant naar coins die:
- Volume explosie hebben (eerste teken!)
- Door resistance breken
- Sterk momentum opbouwen
- Potentie hebben voor 100%+ moves

Run elke 5-15 minuten voor beste resultaten.
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

from core.indicators import Indicators


@dataclass
class BreakoutSignal:
    """Een potentiële breakout"""
    symbol: str
    price: float
    
    # Volume analysis
    volume_spike: float      # Huidige volume vs gemiddeld
    volume_trend: str        # 'EXPLODING', 'RISING', 'NORMAL'
    
    # Price action
    change_1h: float
    change_4h: float  
    change_24h: float
    
    # Technical
    rsi: float
    macd_signal: str         # 'BULLISH_CROSS', 'BULLISH', 'NEUTRAL'
    trend: str               # 'STRONG_UP', 'UP', 'NEUTRAL', 'DOWN'
    
    # Breakout metrics
    distance_from_high_20d: float   # % onder 20-dag high
    distance_from_low_20d: float    # % boven 20-dag low
    breaking_resistance: bool
    
    # Score
    breakout_score: float    # 0-100
    
    # Meta
    timestamp: str
    volume_24h: float


class BreakoutHunter:
    """Scanner voor explosieve breakouts"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def get_all_usdt_pairs(self, min_volume: float = 50000) -> List[Dict]:
        """Get alle USDT pairs met volume"""
        try:
            tickers = self.exchange.fetch_tickers()
            
            pairs = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if '/USDT:' in symbol or '3L' in symbol or '3S' in symbol:
                    continue
                
                vol = ticker.get('quoteVolume', 0) or 0
                price = ticker.get('last', 0) or 0
                change = ticker.get('percentage', 0) or 0
                
                if vol >= min_volume and price > 0:
                    pairs.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': vol,
                        'change_24h': change,
                        'high_24h': ticker.get('high', price),
                        'low_24h': ticker.get('low', price),
                    })
            
            return pairs
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def analyze_breakout(self, symbol: str, ticker_data: Dict) -> Optional[BreakoutSignal]:
        """Analyseer een coin voor breakout potentieel"""
        
        # Fetch different timeframes
        df_1h = self.fetch_ohlcv(symbol, '1h', 50)
        df_4h = self.fetch_ohlcv(symbol, '4h', 50)
        
        if df_1h is None or df_4h is None or len(df_1h) < 20:
            return None
        
        price = ticker_data['price']
        
        # === VOLUME ANALYSIS ===
        vol_current = df_1h['volume'].iloc[-1]
        vol_avg = df_1h['volume'].iloc[-20:-1].mean()
        volume_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Volume trend (laatste 3 uren)
        vol_trend_vals = df_1h['volume'].iloc[-3:].values
        if vol_trend_vals[-1] > vol_trend_vals[0] * 3:
            volume_trend = 'EXPLODING'
        elif vol_trend_vals[-1] > vol_trend_vals[0] * 1.5:
            volume_trend = 'RISING'
        else:
            volume_trend = 'NORMAL'
        
        # === PRICE CHANGES ===
        change_1h = ((price - df_1h['close'].iloc[-2]) / df_1h['close'].iloc[-2]) * 100
        change_4h = ((price - df_4h['close'].iloc[-2]) / df_4h['close'].iloc[-2]) * 100
        change_24h = ticker_data['change_24h']
        
        # === TECHNICAL INDICATORS ===
        df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
        rsi = df_4h['rsi'].iloc[-1]
        
        # MACD
        df_4h['macd'], df_4h['signal'], df_4h['hist'] = Indicators.macd(df_4h['close'])
        
        if df_4h['macd'].iloc[-1] > df_4h['signal'].iloc[-1] and df_4h['macd'].iloc[-2] <= df_4h['signal'].iloc[-2]:
            macd_signal = 'BULLISH_CROSS'
        elif df_4h['hist'].iloc[-1] > 0 and df_4h['hist'].iloc[-1] > df_4h['hist'].iloc[-2]:
            macd_signal = 'BULLISH'
        else:
            macd_signal = 'NEUTRAL'
        
        # Trend
        df_4h['ema20'] = Indicators.ema(df_4h['close'], 20)
        df_4h['ema50'] = Indicators.ema(df_4h['close'], 50)
        
        if price > df_4h['ema20'].iloc[-1] > df_4h['ema50'].iloc[-1]:
            trend = 'STRONG_UP'
        elif price > df_4h['ema20'].iloc[-1]:
            trend = 'UP'
        elif price < df_4h['ema20'].iloc[-1] < df_4h['ema50'].iloc[-1]:
            trend = 'DOWN'
        else:
            trend = 'NEUTRAL'
        
        # === BREAKOUT LEVELS ===
        high_20d = df_4h['high'].iloc[-20:].max()
        low_20d = df_4h['low'].iloc[-20:].min()
        
        distance_from_high = ((high_20d - price) / high_20d) * 100
        distance_from_low = ((price - low_20d) / low_20d) * 100
        
        # Breaking resistance?
        recent_high = df_4h['high'].iloc[-5:-1].max()
        breaking_resistance = price > recent_high * 0.995
        
        # === BREAKOUT SCORE ===
        score = 0
        
        # Volume score (0-30)
        if volume_spike >= 5:
            score += 30
        elif volume_spike >= 3:
            score += 25
        elif volume_spike >= 2:
            score += 20
        elif volume_spike >= 1.5:
            score += 10
        
        # Momentum score (0-25)
        if change_1h >= 10:
            score += 25
        elif change_1h >= 5:
            score += 20
        elif change_1h >= 3:
            score += 15
        elif change_1h >= 1:
            score += 10
        
        # MACD score (0-15)
        if macd_signal == 'BULLISH_CROSS':
            score += 15
        elif macd_signal == 'BULLISH':
            score += 10
        
        # Trend score (0-15)
        if trend == 'STRONG_UP':
            score += 15
        elif trend == 'UP':
            score += 10
        
        # Breakout score (0-15)
        if breaking_resistance:
            score += 15
        elif distance_from_high < 5:
            score += 10
        elif distance_from_high < 10:
            score += 5
        
        return BreakoutSignal(
            symbol=symbol,
            price=price,
            volume_spike=volume_spike,
            volume_trend=volume_trend,
            change_1h=change_1h,
            change_4h=change_4h,
            change_24h=change_24h,
            rsi=rsi,
            macd_signal=macd_signal,
            trend=trend,
            distance_from_high_20d=distance_from_high,
            distance_from_low_20d=distance_from_low,
            breaking_resistance=breaking_resistance,
            breakout_score=score,
            timestamp=str(datetime.now()),
            volume_24h=ticker_data['volume_24h'],
        )
    
    def scan_all(self, min_volume: float = 50000, min_score: float = 40) -> List[BreakoutSignal]:
        """Scan alle coins voor breakouts"""
        
        print(f"Fetching all USDT pairs (min vol: ${min_volume:,.0f})...")
        pairs = self.get_all_usdt_pairs(min_volume)
        print(f"Found {len(pairs)} pairs")
        
        # Pre-filter: alleen coins met beweging
        movers = [p for p in pairs if abs(p['change_24h']) >= 3 or p['volume_24h'] >= 500000]
        print(f"Pre-filtered to {len(movers)} movers")
        
        signals = []
        
        print(f"\nAnalyzing breakout potential...")
        for i, pair in enumerate(movers):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(movers)}")
            
            try:
                signal = self.analyze_breakout(pair['symbol'], pair)
                if signal and signal.breakout_score >= min_score:
                    signals.append(signal)
            except:
                continue
            
            time.sleep(0.05)
        
        # Sort by score
        signals.sort(key=lambda x: x.breakout_score, reverse=True)
        
        return signals
    
    def format_results(self, signals: List[BreakoutSignal]) -> str:
        """Format results voor display"""
        
        lines = [
            "",
            "=" * 100,
            f"  🚀 BREAKOUT HUNTER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 100,
        ]
        
        if not signals:
            lines.append("  No strong breakout signals detected at this moment.")
            lines.append("=" * 100)
            return "\n".join(lines)
        
        # Hot breakouts (score >= 70)
        hot = [s for s in signals if s.breakout_score >= 70]
        if hot:
            lines.append("")
            lines.append("  🔥 HOT BREAKOUTS (Score >= 70)")
            lines.append("-" * 100)
            
            for s in hot[:10]:
                alert = "⚡ BREAKING!" if s.breaking_resistance else ""
                lines.append(f"""
  {s.symbol:<14} Score: {s.breakout_score:.0f}/100 {alert}
  Price: ${s.price:<12.6f} | 1h: {s.change_1h:+.1f}% | 4h: {s.change_4h:+.1f}% | 24h: {s.change_24h:+.1f}%
  Volume: {s.volume_spike:.1f}x ({s.volume_trend}) | RSI: {s.rsi:.0f} | MACD: {s.macd_signal} | Trend: {s.trend}
  Distance from 20d high: {s.distance_from_high_20d:.1f}%
""")
        
        # Warming up (score 50-70)
        warming = [s for s in signals if 50 <= s.breakout_score < 70]
        if warming:
            lines.append("")
            lines.append("  📈 WARMING UP (Score 50-70)")
            lines.append("-" * 100)
            lines.append(f"  {'SYMBOL':<14} {'SCORE':<8} {'1H':<10} {'4H':<10} {'24H':<10} {'VOL':<12} {'TREND':<12}")
            lines.append("-" * 100)
            
            for s in warming[:15]:
                lines.append(f"  {s.symbol:<14} {s.breakout_score:<8.0f} {s.change_1h:<+10.1f} "
                           f"{s.change_4h:<+10.1f} {s.change_24h:<+10.1f} {s.volume_spike:<12.1f}x {s.trend:<12}")
        
        # Watching (score 40-50)
        watching = [s for s in signals if 40 <= s.breakout_score < 50]
        if watching:
            lines.append("")
            lines.append(f"  👀 WATCHING ({len(watching)} coins with score 40-50)")
            top_watch = watching[:10]
            lines.append(f"  {', '.join(s.symbol for s in top_watch)}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"  Total signals: {len(signals)} | Hot: {len(hot)} | Warming: {len(warming)} | Watching: {len(watching)}")
        lines.append("=" * 100)
        
        return "\n".join(lines)
    
    def find_volume_explosions(self, min_volume: float = 100000) -> List[Dict]:
        """Vind coins met plotselinge volume explosies - vroege indicator!"""
        
        pairs = self.get_all_usdt_pairs(min_volume)
        
        explosions = []
        
        for pair in pairs[:200]:  # Top 200 by volume
            try:
                df = self.fetch_ohlcv(pair['symbol'], '15m', 20)  # 15 min candles
                if df is None or len(df) < 10:
                    continue
                
                # Check volume explosion
                vol_current = df['volume'].iloc[-1]
                vol_avg = df['volume'].iloc[-10:-1].mean()
                vol_spike = vol_current / vol_avg if vol_avg > 0 else 1
                
                # Price change last 15 min
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                if vol_spike >= 3 and price_change > 0:
                    explosions.append({
                        'symbol': pair['symbol'],
                        'vol_spike': vol_spike,
                        'price_change_15m': price_change,
                        'price': pair['price'],
                        'change_24h': pair['change_24h'],
                    })
                
                time.sleep(0.02)
            except:
                continue
        
        explosions.sort(key=lambda x: x['vol_spike'], reverse=True)
        return explosions


def main():
    """Run breakout hunter"""
    
    print("\n" + "=" * 70)
    print("  INITIALIZING BREAKOUT HUNTER...")
    print("=" * 70)
    
    hunter = BreakoutHunter()
    
    # Full scan
    signals = hunter.scan_all(min_volume=50000, min_score=40)
    print(hunter.format_results(signals))
    
    # Volume explosions
    print("\n" + "=" * 100)
    print("  💥 VOLUME EXPLOSIONS (Real-time 15min)")
    print("=" * 100)
    
    explosions = hunter.find_volume_explosions(min_volume=100000)
    
    if explosions:
        print(f"\n  {'SYMBOL':<14} {'VOL SPIKE':<12} {'15M CHANGE':<12} {'24H':<10} {'PRICE':<14}")
        print("-" * 70)
        for e in explosions[:20]:
            print(f"  {e['symbol']:<14} {e['vol_spike']:<12.1f}x {e['price_change_15m']:<+12.2f}% "
                  f"{e['change_24h']:<+10.1f}% ${e['price']:<14.6f}")
    else:
        print("  No volume explosions detected right now.")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
