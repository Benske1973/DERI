"""
OPTIMIZED BREAKOUT HUNTER
==========================
Gebaseerd op analyse van 142 mega movers (50%+ gains).

KEY INSIGHTS:
- 51% had oversold RSI (<40) VOOR de move
- Average days to peak: 6.8 dagen
- 35% waren bottom reversals
- Volume spike komt vaak NA de start (niet voor)

STRATEGIE:
1. Zoek oversold coins (RSI < 40) met toenemend volume
2. Detecteer vroege momentum shift (histogram flip)
3. Prioriteer bottom reversals
4. Focus op coins met liquiditeit
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import time

from core.indicators import Indicators


@dataclass
class HunterSignal:
    """Optimized breakout signal"""
    symbol: str
    price: float
    
    # Core metrics
    rsi: float
    is_oversold: bool
    recovery_started: bool
    
    # Volume
    volume_spike: float
    volume_trend: str  # 'BUILDING', 'SPIKE', 'NORMAL'
    
    # Momentum
    macd_hist: float
    hist_change: float      # Change vs previous
    momentum_flip: bool     # Histogram flipped positive
    
    # Price action
    change_24h: float
    change_7d: float
    distance_from_low_7d: float  # How far above 7d low
    
    # Scores
    setup_score: float      # 0-100
    potential: str          # 'HIGH', 'MEDIUM', 'WATCH'
    
    # Signals
    signal_type: str        # 'BOTTOM_REVERSAL', 'MOMENTUM_SHIFT', 'EARLY_BREAKOUT'
    reasons: List[str]
    
    volume_24h: float
    timestamp: str


class OptimizedHunter:
    """Geoptimaliseerde breakout hunter"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def get_pairs(self, min_volume: float = 50000) -> List[Dict]:
        """Get all pairs met ticker data"""
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
                    })
            
            return pairs
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit // 2:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def analyze_coin(self, symbol: str, ticker: Dict) -> Optional[HunterSignal]:
        """Analyseer coin met geoptimaliseerde criteria"""
        
        df_4h = self.fetch_ohlcv(symbol, '4h', 60)  # 10 dagen
        df_1h = self.fetch_ohlcv(symbol, '1h', 48)  # 2 dagen
        
        if df_4h is None or df_1h is None or len(df_4h) < 30:
            return None
        
        price = ticker['price']
        
        # === RSI ANALYSIS (KEY INDICATOR) ===
        df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
        rsi = df_4h['rsi'].iloc[-1]
        rsi_prev = df_4h['rsi'].iloc[-2]
        
        is_oversold = rsi < 40
        was_oversold = df_4h['rsi'].iloc[-5:].min() < 35
        recovery_started = was_oversold and rsi > rsi_prev and rsi > 30
        
        # === VOLUME ANALYSIS ===
        vol_current = df_1h['volume'].iloc[-3:].mean()  # Last 3 hours
        vol_avg = df_1h['volume'].iloc[-24:-3].mean()   # Previous day
        volume_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Volume trend
        vol_early = df_1h['volume'].iloc[-24:-12].mean()
        vol_recent = df_1h['volume'].iloc[-12:].mean()
        vol_trend_ratio = vol_recent / vol_early if vol_early > 0 else 1
        
        if volume_spike >= 3:
            volume_trend = 'SPIKE'
        elif vol_trend_ratio > 1.5:
            volume_trend = 'BUILDING'
        else:
            volume_trend = 'NORMAL'
        
        # === MACD ANALYSIS ===
        df_4h['macd'], df_4h['signal'], df_4h['hist'] = Indicators.macd(df_4h['close'])
        hist = df_4h['hist'].iloc[-1]
        hist_prev = df_4h['hist'].iloc[-2]
        hist_prev2 = df_4h['hist'].iloc[-3]
        
        hist_change = hist - hist_prev
        momentum_flip = hist > 0 and hist_prev <= 0  # Just crossed positive
        
        # === PRICE ACTION ===
        low_7d = df_4h['low'].iloc[-42:].min()  # 7 days in 4h candles
        high_7d = df_4h['high'].iloc[-42:].max()
        
        distance_from_low = ((price - low_7d) / low_7d) * 100 if low_7d > 0 else 0
        
        # 7 day change
        price_7d_ago = df_4h['close'].iloc[-42] if len(df_4h) >= 42 else df_4h['close'].iloc[0]
        change_7d = ((price - price_7d_ago) / price_7d_ago) * 100 if price_7d_ago > 0 else 0
        
        # === SCORING BASED ON MEGA MOVER PATTERNS ===
        score = 0
        reasons = []
        signal_type = None
        
        # 1. Oversold RSI recovery (51% of mega movers had this!)
        if is_oversold:
            score += 30
            reasons.append(f"Oversold RSI: {rsi:.0f}")
            if recovery_started:
                score += 20
                reasons.append("Recovery starting!")
                signal_type = 'BOTTOM_REVERSAL'
        elif was_oversold and rsi > 40:
            score += 20
            reasons.append("Recent oversold bounce")
            signal_type = 'BOTTOM_REVERSAL'
        
        # 2. MACD momentum shift
        if momentum_flip:
            score += 25
            reasons.append("MACD histogram flipped positive!")
            if signal_type is None:
                signal_type = 'MOMENTUM_SHIFT'
        elif hist > 0 and hist > hist_prev and hist_prev > hist_prev2:
            score += 15
            reasons.append("MACD momentum building")
        
        # 3. Volume confirmation
        if volume_trend == 'SPIKE':
            score += 20
            reasons.append(f"Volume spike: {volume_spike:.1f}x")
        elif volume_trend == 'BUILDING':
            score += 10
            reasons.append("Volume building")
        
        # 4. Price position (early in move = more upside)
        if distance_from_low < 20:  # Within 20% of 7d low
            score += 10
            reasons.append(f"Near bottom: {distance_from_low:.0f}% from 7d low")
        
        # 5. Negative 7d change (oversold bounce setup)
        if change_7d < -20 and rsi < 45:
            score += 10
            reasons.append(f"7d drop: {change_7d:.0f}% - bounce potential")
        
        # Determine potential
        if score >= 60:
            potential = 'HIGH'
        elif score >= 40:
            potential = 'MEDIUM'
        elif score >= 25:
            potential = 'WATCH'
        else:
            return None
        
        if signal_type is None:
            if distance_from_low < 15:
                signal_type = 'EARLY_BREAKOUT'
            else:
                signal_type = 'MOMENTUM_SHIFT'
        
        return HunterSignal(
            symbol=symbol,
            price=price,
            rsi=rsi,
            is_oversold=is_oversold,
            recovery_started=recovery_started,
            volume_spike=volume_spike,
            volume_trend=volume_trend,
            macd_hist=hist,
            hist_change=hist_change,
            momentum_flip=momentum_flip,
            change_24h=ticker['change_24h'],
            change_7d=change_7d,
            distance_from_low_7d=distance_from_low,
            setup_score=score,
            potential=potential,
            signal_type=signal_type,
            reasons=reasons,
            volume_24h=ticker['volume_24h'],
            timestamp=str(datetime.now()),
        )
    
    def hunt(self, min_volume: float = 50000, min_score: float = 25) -> List[HunterSignal]:
        """Hunt for breakout candidates"""
        
        print(f"\n🎯 Fetching pairs (min vol: ${min_volume:,.0f})...")
        pairs = self.get_pairs(min_volume)
        print(f"Found {len(pairs)} pairs")
        
        # Quick pre-filter
        candidates = [p for p in pairs if p['change_24h'] < 50]  # Not already pumped
        print(f"Analyzing {len(candidates)} candidates...")
        
        signals = []
        
        for i, pair in enumerate(candidates):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(candidates)}")
            
            try:
                signal = self.analyze_coin(pair['symbol'], pair)
                if signal and signal.setup_score >= min_score:
                    signals.append(signal)
            except:
                continue
            
            time.sleep(0.03)
        
        # Sort by score
        signals.sort(key=lambda x: x.setup_score, reverse=True)
        
        return signals
    
    def format_results(self, signals: List[HunterSignal]) -> str:
        """Format results"""
        
        lines = [
            "",
            "=" * 120,
            f"  🎯 OPTIMIZED BREAKOUT HUNTER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "  Based on analysis of 142 mega movers (50%+ gains)",
            "=" * 120,
        ]
        
        if not signals:
            lines.append("  No strong signals at this moment.")
            return "\n".join(lines)
        
        # Group by potential
        high = [s for s in signals if s.potential == 'HIGH']
        medium = [s for s in signals if s.potential == 'MEDIUM']
        watch = [s for s in signals if s.potential == 'WATCH']
        
        if high:
            lines.append("")
            lines.append("  🔥 HIGH POTENTIAL - Act on these!")
            lines.append("-" * 120)
            
            for s in high:
                recovery = "⬆️ RECOVERING" if s.recovery_started else ""
                flip = "🔄 MACD FLIP!" if s.momentum_flip else ""
                
                lines.append(f"""
  {s.symbol:<14} | Score: {s.setup_score:.0f}/100 | {s.signal_type} {recovery} {flip}
  ├─ Price: ${s.price:.6f} | RSI: {s.rsi:.0f} | 24h: {s.change_24h:+.1f}% | 7d: {s.change_7d:+.1f}%
  ├─ Volume: {s.volume_spike:.1f}x ({s.volume_trend}) | Distance from 7d low: {s.distance_from_low_7d:.0f}%
  └─ Signals: {' | '.join(s.reasons)}
""")
        
        if medium:
            lines.append("")
            lines.append("  📊 MEDIUM POTENTIAL - Watch closely")
            lines.append("-" * 120)
            lines.append(f"  {'SYMBOL':<14} {'SCORE':<8} {'RSI':<8} {'VOL':<10} {'24H':<10} {'7D':<10} {'FROM LOW':<10} {'TYPE':<20}")
            lines.append("-" * 120)
            
            for s in medium[:15]:
                lines.append(f"  {s.symbol:<14} {s.setup_score:<8.0f} {s.rsi:<8.0f} "
                           f"{s.volume_spike:<10.1f}x {s.change_24h:<+10.1f} {s.change_7d:<+10.1f} "
                           f"{s.distance_from_low_7d:<10.0f}% {s.signal_type:<20}")
        
        if watch:
            lines.append("")
            lines.append(f"  👀 WATCHLIST ({len(watch)} coins with score 25-40)")
            top_watch = [s.symbol for s in watch[:15]]
            lines.append(f"  {', '.join(top_watch)}")
        
        # Summary stats
        oversold_count = len([s for s in signals if s.is_oversold])
        recovering_count = len([s for s in signals if s.recovery_started])
        flip_count = len([s for s in signals if s.momentum_flip])
        
        lines.append("")
        lines.append("=" * 120)
        lines.append(f"  📈 SUMMARY")
        lines.append(f"  Total signals: {len(signals)} | High: {len(high)} | Medium: {len(medium)} | Watch: {len(watch)}")
        lines.append(f"  Oversold coins: {oversold_count} | Active recoveries: {recovering_count} | MACD flips: {flip_count}")
        lines.append("=" * 120)
        
        return "\n".join(lines)


def main():
    """Run optimized hunter"""
    
    print("\n" + "=" * 70)
    print("  OPTIMIZED BREAKOUT HUNTER")
    print("  Using patterns from 142 analyzed mega movers")
    print("=" * 70)
    
    hunter = OptimizedHunter()
    signals = hunter.hunt(min_volume=50000, min_score=25)
    print(hunter.format_results(signals))


if __name__ == "__main__":
    main()
