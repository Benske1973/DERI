"""
MEGA MOVER ANALYZER
====================
Analyseert coins die 100%+ moves hebben gemaakt.
Zoekt naar patronen die we vroeg kunnen detecteren.

Dit helpt ons begrijpen:
- Welke technische signalen komen VOOR de grote move?
- Wat is het ideale instapmoment?
- Hoe lang duren deze moves?
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
class MegaMove:
    """Een grote prijsbeweging"""
    symbol: str
    
    # Move stats
    start_price: float
    peak_price: float
    current_price: float
    gain_to_peak: float       # % stijging tot piek
    current_gain: float       # Huidige % vs start
    
    # Timing
    move_start_date: str
    peak_date: str
    days_to_peak: int
    
    # Pre-move indicators (wat we konden zien VOOR de move)
    pre_volume_spike: float   # Volume spike net voor de move
    pre_rsi: float           # RSI net voor de move
    pre_macd_signal: str     # MACD status voor de move
    pre_squeeze: float       # Volatility squeeze voor de move
    
    # Pattern
    breakout_type: str       # 'RESISTANCE_BREAK', 'BOTTOM_REVERSAL', 'CONTINUATION'
    
    volume_24h: float


class MegaMoverAnalyzer:
    """Analyseert mega movers"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
    
    def get_all_pairs(self, min_volume: float = 10000) -> List[Dict]:
        """Get alle pairs"""
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
            if not ohlcv or len(ohlcv) < limit // 2:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def analyze_mega_move(self, symbol: str, ticker: Dict, lookback_days: int = 30) -> Optional[MegaMove]:
        """Analyseer of een coin een mega move heeft gemaakt"""
        
        # Fetch daily data
        df = self.fetch_ohlcv(symbol, '1d', lookback_days)
        if df is None or len(df) < 14:
            return None
        
        # Find the lowest point and highest point
        low_idx = df['low'].idxmin()
        high_idx = df['high'].idxmax()
        
        lowest = df.loc[low_idx, 'low']
        highest = df.loc[high_idx, 'high']
        
        # Check if there was a significant move (at least 50%)
        if highest <= lowest * 1.5:
            return None
        
        # Check if the move was upward (low before high)
        if high_idx <= low_idx:
            return None  # This was a drop, not a pump
        
        current_price = ticker['price']
        gain_to_peak = ((highest - lowest) / lowest) * 100
        current_gain = ((current_price - lowest) / lowest) * 100
        
        # Calculate days to peak
        start_date = df.loc[low_idx, 'timestamp']
        peak_date = df.loc[high_idx, 'timestamp']
        days_to_peak = (peak_date - start_date).days
        
        # Get 4h data for more detailed analysis
        df_4h = self.fetch_ohlcv(symbol, '4h', 100)
        if df_4h is None:
            return None
        
        # Find the candle just before the move started
        # We look for the period around the low point
        pre_move_idx = max(0, len(df_4h) - (lookback_days - low_idx) * 6)  # Approximate
        
        if pre_move_idx >= len(df_4h) - 10:
            pre_move_idx = len(df_4h) // 2  # Fallback
        
        # Calculate indicators at pre-move point
        df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
        df_4h['macd'], df_4h['signal'], df_4h['hist'] = Indicators.macd(df_4h['close'])
        
        # Bollinger squeeze
        upper, lower, sma = Indicators.bollinger_bands(df_4h['close'], 20, 2)
        bb_width = (upper - lower) / sma
        avg_bb = bb_width.rolling(20).mean()
        squeeze = bb_width / avg_bb
        
        # Pre-move values (safe index access)
        safe_idx = min(pre_move_idx, len(df_4h) - 5)
        
        pre_rsi = df_4h['rsi'].iloc[safe_idx] if not pd.isna(df_4h['rsi'].iloc[safe_idx]) else 50
        pre_hist = df_4h['hist'].iloc[safe_idx] if not pd.isna(df_4h['hist'].iloc[safe_idx]) else 0
        prev_hist = df_4h['hist'].iloc[safe_idx - 1] if safe_idx > 0 and not pd.isna(df_4h['hist'].iloc[safe_idx - 1]) else 0
        pre_squeeze = squeeze.iloc[safe_idx] if not pd.isna(squeeze.iloc[safe_idx]) else 1
        
        # Volume at move start
        vol_at_move = df_4h['volume'].iloc[safe_idx:safe_idx+3].mean()
        vol_before = df_4h['volume'].iloc[max(0, safe_idx-10):safe_idx].mean()
        pre_volume_spike = vol_at_move / vol_before if vol_before > 0 else 1
        
        # MACD signal
        if pre_hist > 0 and pre_hist > prev_hist:
            pre_macd_signal = 'BULLISH_CROSS' if prev_hist <= 0 else 'BULLISH'
        else:
            pre_macd_signal = 'NEUTRAL'
        
        # Determine breakout type
        price_before = df_4h['close'].iloc[safe_idx-5:safe_idx].mean() if safe_idx > 5 else lowest
        if pre_rsi < 35:
            breakout_type = 'BOTTOM_REVERSAL'
        elif lowest < price_before * 0.95:
            breakout_type = 'CONTINUATION'
        else:
            breakout_type = 'RESISTANCE_BREAK'
        
        return MegaMove(
            symbol=symbol,
            start_price=lowest,
            peak_price=highest,
            current_price=current_price,
            gain_to_peak=gain_to_peak,
            current_gain=current_gain,
            move_start_date=start_date.strftime('%Y-%m-%d'),
            peak_date=peak_date.strftime('%Y-%m-%d'),
            days_to_peak=days_to_peak,
            pre_volume_spike=pre_volume_spike,
            pre_rsi=pre_rsi,
            pre_macd_signal=pre_macd_signal,
            pre_squeeze=pre_squeeze,
            breakout_type=breakout_type,
            volume_24h=ticker['volume_24h'],
        )
    
    def find_mega_movers(self, min_gain: float = 50, lookback_days: int = 14) -> List[MegaMove]:
        """Vind alle coins met mega moves"""
        
        print(f"\nSearching for coins with {min_gain}%+ moves in last {lookback_days} days...")
        
        pairs = self.get_all_pairs(min_volume=50000)
        print(f"Found {len(pairs)} pairs to analyze")
        
        mega_movers = []
        
        for i, pair in enumerate(pairs):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(pairs)}")
            
            try:
                move = self.analyze_mega_move(pair['symbol'], pair, lookback_days)
                if move and move.gain_to_peak >= min_gain:
                    mega_movers.append(move)
            except:
                continue
            
            time.sleep(0.05)
        
        # Sort by gain
        mega_movers.sort(key=lambda x: x.gain_to_peak, reverse=True)
        
        return mega_movers
    
    def analyze_patterns(self, movers: List[MegaMove]) -> Dict:
        """Analyseer gemeenschappelijke patronen in mega movers"""
        
        if not movers:
            return {}
        
        patterns = {
            'total_movers': len(movers),
            'avg_gain': np.mean([m.gain_to_peak for m in movers]),
            'max_gain': max(m.gain_to_peak for m in movers),
            'avg_days_to_peak': np.mean([m.days_to_peak for m in movers]),
            
            # Pre-move indicators
            'avg_pre_volume_spike': np.mean([m.pre_volume_spike for m in movers]),
            'avg_pre_rsi': np.mean([m.pre_rsi for m in movers]),
            'pct_with_volume_spike': len([m for m in movers if m.pre_volume_spike > 1.5]) / len(movers) * 100,
            'pct_with_oversold_rsi': len([m for m in movers if m.pre_rsi < 40]) / len(movers) * 100,
            'pct_with_squeeze': len([m for m in movers if m.pre_squeeze < 0.8]) / len(movers) * 100,
            
            # Breakout types
            'breakout_types': {},
            'macd_signals': {},
        }
        
        for m in movers:
            patterns['breakout_types'][m.breakout_type] = patterns['breakout_types'].get(m.breakout_type, 0) + 1
            patterns['macd_signals'][m.pre_macd_signal] = patterns['macd_signals'].get(m.pre_macd_signal, 0) + 1
        
        return patterns
    
    def format_results(self, movers: List[MegaMove], patterns: Dict) -> str:
        """Format results"""
        
        lines = [
            "",
            "=" * 120,
            f"  🚀 MEGA MOVER ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 120,
        ]
        
        if not movers:
            lines.append("  No mega movers found in the specified period.")
            return "\n".join(lines)
        
        # Top movers
        lines.append("")
        lines.append("  📊 TOP MEGA MOVERS (by peak gain)")
        lines.append("-" * 120)
        lines.append(f"  {'SYMBOL':<14} {'PEAK GAIN':<12} {'CURRENT':<12} {'DAYS':<8} "
                    f"{'PRE-VOL':<10} {'PRE-RSI':<10} {'MACD':<15} {'TYPE':<20}")
        lines.append("-" * 120)
        
        for m in movers[:30]:
            status = "🟢" if m.current_gain >= m.gain_to_peak * 0.8 else "🟡" if m.current_gain > 0 else "🔴"
            lines.append(f"  {m.symbol:<14} {m.gain_to_peak:<+12.1f}% {m.current_gain:<+12.1f}% {m.days_to_peak:<8} "
                        f"{m.pre_volume_spike:<10.1f}x {m.pre_rsi:<10.1f} {m.pre_macd_signal:<15} {m.breakout_type:<20} {status}")
        
        # Pattern analysis
        lines.append("")
        lines.append("=" * 120)
        lines.append("  📈 PATTERN ANALYSIS - What signals came BEFORE the big moves?")
        lines.append("=" * 120)
        
        lines.append(f"""
  Total mega movers analyzed: {patterns['total_movers']}
  
  GAINS:
  ├─ Average gain to peak: {patterns['avg_gain']:.1f}%
  ├─ Maximum gain: {patterns['max_gain']:.1f}%
  └─ Average days to peak: {patterns['avg_days_to_peak']:.1f}
  
  PRE-MOVE SIGNALS (what we could have detected):
  ├─ {patterns['pct_with_volume_spike']:.0f}% had volume spike (>1.5x) at move start
  ├─ {patterns['pct_with_oversold_rsi']:.0f}% had oversold RSI (<40) before move
  ├─ {patterns['pct_with_squeeze']:.0f}% had volatility squeeze before move
  └─ Average pre-move volume spike: {patterns['avg_pre_volume_spike']:.1f}x
  
  BREAKOUT TYPES:
""")
        
        for btype, count in sorted(patterns['breakout_types'].items(), key=lambda x: -x[1]):
            pct = count / patterns['total_movers'] * 100
            lines.append(f"  ├─ {btype}: {count} ({pct:.0f}%)")
        
        lines.append("")
        lines.append("  MACD SIGNALS AT START:")
        for sig, count in sorted(patterns['macd_signals'].items(), key=lambda x: -x[1]):
            pct = count / patterns['total_movers'] * 100
            lines.append(f"  ├─ {sig}: {count} ({pct:.0f}%)")
        
        # Key insights
        lines.append("")
        lines.append("=" * 120)
        lines.append("  💡 KEY INSIGHTS FOR EARLY DETECTION")
        lines.append("=" * 120)
        
        insights = []
        
        if patterns['pct_with_volume_spike'] > 60:
            insights.append("✓ Volume spike is a strong predictor - monitor for 1.5x+ volume increase")
        
        if patterns['pct_with_oversold_rsi'] > 40:
            insights.append("✓ Oversold RSI often precedes big moves - watch RSI < 40 with volume")
        
        if patterns['pct_with_squeeze'] > 50:
            insights.append("✓ Volatility squeeze often precedes explosions - look for compressed Bollinger Bands")
        
        if patterns['breakout_types'].get('BOTTOM_REVERSAL', 0) > patterns['total_movers'] * 0.3:
            insights.append("✓ Bottom reversals are common - look for oversold bounces with volume")
        
        if patterns['macd_signals'].get('BULLISH_CROSS', 0) > patterns['total_movers'] * 0.3:
            insights.append("✓ MACD bullish cross often signals the start - watch for histogram flip")
        
        if patterns['avg_days_to_peak'] < 10:
            insights.append(f"✓ Moves are fast! Average {patterns['avg_days_to_peak']:.1f} days to peak")
        
        for insight in insights:
            lines.append(f"  {insight}")
        
        lines.append("")
        lines.append("=" * 120)
        
        return "\n".join(lines)


def main():
    """Run mega mover analysis"""
    
    print("\n" + "=" * 70)
    print("  MEGA MOVER ANALYZER")
    print("  Analyzing coins with big moves to find patterns...")
    print("=" * 70)
    
    analyzer = MegaMoverAnalyzer()
    
    # Find movers with 50%+ gain in last 14 days
    movers = analyzer.find_mega_movers(min_gain=50, lookback_days=14)
    
    # Analyze patterns
    patterns = analyzer.analyze_patterns(movers)
    
    # Print results
    print(analyzer.format_results(movers, patterns))
    
    return movers


if __name__ == "__main__":
    main()
