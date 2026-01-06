"""
EARLY BREAKOUT DETECTOR - Vind coins VOOR de 100%+ move
========================================================
Detecteert:
1. Accumulation patterns (langzame stijging + toenemend volume)
2. Squeeze setups (lage volatiliteit voor explosie)
3. First breakout candle (voordat rest volgt)
4. Smart money inflow (grote orders)

Dit zijn de patronen die vaak voorafgaan aan 100-1000% moves.
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
class EarlySignal:
    """Early breakout signal"""
    symbol: str
    price: float
    
    # Pattern type
    pattern: str  # 'ACCUMULATION', 'SQUEEZE', 'FIRST_BREAKOUT', 'VOLUME_DIVERGENCE'
    
    # Technical setup
    setup_score: float       # 0-100
    risk_reward: float       # Potential R:R
    
    # Metrics
    volatility_squeeze: float    # How compressed is volatility (lower = better)
    volume_buildup: float        # Volume trend
    price_structure: str         # 'HIGHER_LOWS', 'CONSOLIDATING', 'BASING'
    
    # Entry/Exit
    entry_price: float
    stop_loss: float
    target_1: float   # Conservative
    target_2: float   # Aggressive
    
    # Context
    days_in_pattern: int
    change_24h: float
    rsi: float
    
    volume_24h: float
    timestamp: str


class EarlyDetector:
    """Vroege detectie van breakout setups"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def get_pairs(self, min_volume: float = 100000) -> List[Dict]:
        """Get active pairs"""
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
                
                if vol >= min_volume and price > 0:
                    pairs.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': vol,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            return pairs
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def detect_squeeze(self, df: pd.DataFrame) -> Dict:
        """Detect volatility squeeze - vaak voorloper van explosieve move"""
        
        # Bollinger Bands width
        upper, lower, sma = Indicators.bollinger_bands(df['close'], 20, 2)
        bb_width = (upper - lower) / sma
        
        # ATR as % of price
        atr = Indicators.atr(df, 14)
        atr_pct = (atr / df['close']) * 100
        
        # Compare current to recent average
        current_bb = bb_width.iloc[-1]
        avg_bb = bb_width.iloc[-20:-1].mean()
        bb_squeeze = current_bb / avg_bb if avg_bb > 0 else 1
        
        current_atr = atr_pct.iloc[-1]
        avg_atr = atr_pct.iloc[-20:-1].mean()
        atr_squeeze = current_atr / avg_atr if avg_atr > 0 else 1
        
        # Squeeze score (lower = tighter squeeze)
        squeeze_score = (bb_squeeze + atr_squeeze) / 2
        
        return {
            'squeeze_score': squeeze_score,
            'bb_width': current_bb,
            'atr_pct': current_atr,
            'is_squeezed': squeeze_score < 0.7,  # 30%+ compression
        }
    
    def detect_accumulation(self, df: pd.DataFrame) -> Dict:
        """Detect accumulation pattern - higher lows + volume buildup"""
        
        # Check for higher lows (laatste 10 candles)
        lows = df['low'].iloc[-10:].values
        higher_lows_count = 0
        for i in range(1, len(lows)):
            if lows[i] > lows[i-1] * 0.99:  # 1% tolerance
                higher_lows_count += 1
        
        # Volume trend
        vol_start = df['volume'].iloc[-10:-7].mean()
        vol_end = df['volume'].iloc[-3:].mean()
        volume_trend = vol_end / vol_start if vol_start > 0 else 1
        
        # Price range compression
        range_start = df['high'].iloc[-10:-7].max() - df['low'].iloc[-10:-7].min()
        range_end = df['high'].iloc[-3:].max() - df['low'].iloc[-3:].min()
        range_compression = range_end / range_start if range_start > 0 else 1
        
        return {
            'higher_lows': higher_lows_count >= 6,
            'higher_lows_count': higher_lows_count,
            'volume_trend': volume_trend,
            'volume_building': volume_trend > 1.3,
            'range_compression': range_compression,
            'consolidating': range_compression < 0.7,
        }
    
    def detect_first_breakout(self, df: pd.DataFrame) -> Dict:
        """Detect first breakout candle - voordat de rest volgt"""
        
        # Recent resistance (20 candle high, excluding last 3)
        resistance = df['high'].iloc[-20:-3].max()
        current_high = df['high'].iloc[-1]
        prev_close = df['close'].iloc[-2]
        
        # First break above resistance
        first_break = current_high > resistance and prev_close <= resistance
        
        # Volume confirmation
        vol_current = df['volume'].iloc[-1]
        vol_avg = df['volume'].iloc[-20:-1].mean()
        vol_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Strong candle?
        candle_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        candle_range = df['high'].iloc[-1] - df['low'].iloc[-1]
        body_ratio = candle_body / candle_range if candle_range > 0 else 0
        
        bullish_candle = df['close'].iloc[-1] > df['open'].iloc[-1]
        
        return {
            'first_break': first_break,
            'resistance': resistance,
            'break_amount': ((current_high - resistance) / resistance) * 100 if resistance > 0 else 0,
            'volume_spike': vol_spike,
            'volume_confirmed': vol_spike > 2,
            'strong_candle': body_ratio > 0.6 and bullish_candle,
        }
    
    def detect_volume_divergence(self, df: pd.DataFrame) -> Dict:
        """Detect volume divergence - volume stijgt terwijl prijs stabiel"""
        
        # Price change laatste 10 candles
        price_change = ((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) * 100
        
        # Volume change
        vol_start = df['volume'].iloc[-10:-7].mean()
        vol_end = df['volume'].iloc[-3:].mean()
        vol_change = ((vol_end - vol_start) / vol_start) * 100 if vol_start > 0 else 0
        
        # Divergence: volume stijgt significant, prijs relatief stabiel
        has_divergence = vol_change > 50 and abs(price_change) < 10
        
        return {
            'price_change': price_change,
            'volume_change': vol_change,
            'has_divergence': has_divergence,
        }
    
    def analyze_setup(self, symbol: str, ticker: Dict) -> Optional[EarlySignal]:
        """Analyseer een coin voor early breakout setup"""
        
        df = self.fetch_ohlcv(symbol, '4h', 60)  # 10 dagen 4h data
        if df is None or len(df) < 40:
            return None
        
        price = ticker['price']
        
        # Run all detections
        squeeze = self.detect_squeeze(df)
        accum = self.detect_accumulation(df)
        first_break = self.detect_first_breakout(df)
        vol_div = self.detect_volume_divergence(df)
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], 14)
        rsi = df['rsi'].iloc[-1]
        
        # ATR for stops
        atr = Indicators.atr(df, 14).iloc[-1]
        atr_pct = (atr / price) * 100
        
        # Determine pattern type and score
        pattern = None
        score = 0
        
        # First breakout is highest priority
        if first_break['first_break'] and first_break['volume_confirmed']:
            pattern = 'FIRST_BREAKOUT'
            score = 80
            if first_break['strong_candle']:
                score += 10
            if rsi < 70:
                score += 10
        
        # Squeeze ready to pop
        elif squeeze['is_squeezed'] and accum['volume_building']:
            pattern = 'SQUEEZE_READY'
            score = 70
            if accum['higher_lows']:
                score += 15
            if rsi > 50 and rsi < 65:
                score += 15
        
        # Accumulation pattern
        elif accum['higher_lows'] and accum['volume_building']:
            pattern = 'ACCUMULATION'
            score = 65
            if squeeze['squeeze_score'] < 0.8:
                score += 10
            if rsi > 45 and rsi < 60:
                score += 10
        
        # Volume divergence
        elif vol_div['has_divergence']:
            pattern = 'VOLUME_DIVERGENCE'
            score = 55
            if accum['consolidating']:
                score += 10
        
        # No clear pattern
        else:
            return None
        
        # Minimum score filter
        if score < 60:
            return None
        
        # Calculate entry/exit levels
        entry = price
        stop_loss = price - (2 * atr)
        
        # Conservative target: 2:1 R:R
        risk = entry - stop_loss
        target_1 = entry + (2 * risk)
        
        # Aggressive target: 5:1 R:R (voor 100%+ moves)
        target_2 = entry + (5 * risk)
        
        # Risk/Reward
        potential_gain = ((target_1 - entry) / entry) * 100
        potential_loss = ((entry - stop_loss) / entry) * 100
        risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
        
        # Price structure
        if accum['higher_lows']:
            structure = 'HIGHER_LOWS'
        elif accum['consolidating']:
            structure = 'CONSOLIDATING'
        else:
            structure = 'BASING'
        
        return EarlySignal(
            symbol=symbol,
            price=price,
            pattern=pattern,
            setup_score=score,
            risk_reward=risk_reward,
            volatility_squeeze=squeeze['squeeze_score'],
            volume_buildup=accum['volume_trend'],
            price_structure=structure,
            entry_price=entry,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            days_in_pattern=10,  # Simplified
            change_24h=ticker['change_24h'],
            rsi=rsi,
            volume_24h=ticker['volume_24h'],
            timestamp=str(datetime.now()),
        )
    
    def scan_all(self, min_volume: float = 100000, min_score: float = 60) -> List[EarlySignal]:
        """Scan alle coins voor early setups"""
        
        print(f"\nFetching pairs (min vol: ${min_volume:,.0f})...")
        pairs = self.get_pairs(min_volume)
        print(f"Found {len(pairs)} pairs")
        
        # Pre-filter: Skip coins die al te veel gestegen zijn (miss de boot)
        # Focus op coins die nog niet exploded zijn
        candidates = [p for p in pairs if p['change_24h'] < 30 and p['change_24h'] > -20]
        print(f"Filtered to {len(candidates)} candidates (not yet exploded)")
        
        signals = []
        
        print("\nAnalyzing early setups...")
        for i, pair in enumerate(candidates):
            if (i + 1) % 30 == 0:
                print(f"  Progress: {i+1}/{len(candidates)}")
            
            try:
                signal = self.analyze_setup(pair['symbol'], pair)
                if signal and signal.setup_score >= min_score:
                    signals.append(signal)
            except:
                continue
            
            time.sleep(0.05)
        
        # Sort by score
        signals.sort(key=lambda x: x.setup_score, reverse=True)
        
        return signals
    
    def format_results(self, signals: List[EarlySignal]) -> str:
        """Format results"""
        
        lines = [
            "",
            "=" * 110,
            f"  🎯 EARLY BREAKOUT DETECTOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "=" * 110,
        ]
        
        if not signals:
            lines.append("  No strong early setups detected.")
            lines.append("=" * 110)
            return "\n".join(lines)
        
        # Group by pattern type
        for pattern_type in ['FIRST_BREAKOUT', 'SQUEEZE_READY', 'ACCUMULATION', 'VOLUME_DIVERGENCE']:
            pattern_signals = [s for s in signals if s.pattern == pattern_type]
            
            if not pattern_signals:
                continue
            
            emoji = {
                'FIRST_BREAKOUT': '🚀',
                'SQUEEZE_READY': '💥', 
                'ACCUMULATION': '📈',
                'VOLUME_DIVERGENCE': '🔍',
            }.get(pattern_type, '')
            
            lines.append("")
            lines.append(f"  {emoji} {pattern_type.replace('_', ' ')} ({len(pattern_signals)} signals)")
            lines.append("-" * 110)
            
            for s in pattern_signals[:8]:
                gain_1 = ((s.target_1 - s.entry_price) / s.entry_price) * 100
                gain_2 = ((s.target_2 - s.entry_price) / s.entry_price) * 100
                loss = ((s.entry_price - s.stop_loss) / s.entry_price) * 100
                
                lines.append(f"""
  {s.symbol:<14} Score: {s.setup_score:.0f}/100 | R:R {s.risk_reward:.1f}:1
  ├─ Price: ${s.price:.6f} | RSI: {s.rsi:.0f} | 24h: {s.change_24h:+.1f}%
  ├─ Structure: {s.price_structure} | Vol Buildup: {s.volume_buildup:.1f}x | Squeeze: {s.volatility_squeeze:.2f}
  └─ Entry: ${s.entry_price:.6f} | SL: ${s.stop_loss:.6f} (-{loss:.1f}%) | TP1: +{gain_1:.0f}% | TP2: +{gain_2:.0f}%
""")
        
        lines.append("=" * 110)
        
        # Summary
        by_pattern = {}
        for s in signals:
            by_pattern[s.pattern] = by_pattern.get(s.pattern, 0) + 1
        
        summary_parts = [f"{k}: {v}" for k, v in sorted(by_pattern.items())]
        lines.append(f"  Total: {len(signals)} setups | " + " | ".join(summary_parts))
        lines.append("=" * 110)
        
        return "\n".join(lines)


def main():
    """Run early detector"""
    
    print("\n" + "=" * 70)
    print("  INITIALIZING EARLY BREAKOUT DETECTOR...")
    print("  Looking for coins BEFORE the 100%+ move!")
    print("=" * 70)
    
    detector = EarlyDetector()
    
    signals = detector.scan_all(min_volume=100000, min_score=60)
    print(detector.format_results(signals))


if __name__ == "__main__":
    main()
