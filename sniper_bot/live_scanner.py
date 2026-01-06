"""
LIVE SWING SCANNER
==================
Real-time monitoring van top coins voor breakout entries.
Toont:
- Huidige status van elke coin
- Actieve signalen
- Potentiële setups (bijna breakout)
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

from core.data import DataFetcher
from strategies.swing_trader import SwingTrader
from core.indicators import Indicators
import pandas as pd
from datetime import datetime


class LiveScanner:
    """Live scanner voor swing trade opportunities"""
    
    def __init__(self):
        self.fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
        self.trader = SwingTrader()
    
    def scan_symbol(self, symbol: str) -> dict:
        """Scan een symbol voor trading opportunities"""
        
        df = self.fetcher.fetch(symbol, '4h', days=30, use_cache=False)
        if df.empty:
            return {'symbol': symbol, 'error': 'No data'}
        
        # Calculate indicators
        df = self.trader.calculate_indicators(df)
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check for signal
        entry = self.trader.check_entry(df, len(df) - 1)
        
        # Calculate breakout progress
        if last['hist'] > 0:
            breakout_pct = (last['hist'] / last['breakout_threshold']) * 100
        else:
            breakout_pct = 0
        
        # Trend status
        if last['uptrend']:
            trend = 'UPTREND'
        elif last['downtrend']:
            trend = 'DOWNTREND'
        else:
            trend = 'NEUTRAL'
        
        # Momentum
        hist_growing = last['hist'] > prev['hist']
        rsi_rising = last['rsi'] > prev['rsi']
        
        return {
            'symbol': symbol,
            'price': last['close'],
            'change_24h': ((last['close'] - df.iloc[-7]['close']) / df.iloc[-7]['close']) * 100,
            
            # Signal
            'has_signal': entry is not None,
            'signal_type': entry['direction'] if entry else None,
            'entry_price': entry['entry'] if entry else None,
            'stop_loss': entry['stop'] if entry else None,
            
            # Indicators
            'rsi': last['rsi'],
            'rsi_rising': rsi_rising,
            'trend': trend,
            'atr_pct': last['atr_pct'],
            
            # Breakout status
            'breakout_pct': breakout_pct,
            'hist_growing': hist_growing,
            'volume_ok': last['high_volume'],
            
            # Timestamp
            'candle_time': last['timestamp'],
        }
    
    def scan_all(self) -> list:
        """Scan alle top coins"""
        results = []
        
        for symbol in SwingTrader.TOP_COINS:
            result = self.scan_symbol(symbol)
            results.append(result)
        
        return results
    
    def format_results(self, results: list) -> str:
        """Format results als readable output"""
        
        lines = [
            "",
            "=" * 90,
            f"  LIVE SWING SCANNER - {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC",
            "=" * 90,
            "",
        ]
        
        # Active signals
        signals = [r for r in results if r.get('has_signal')]
        if signals:
            lines.append("  🎯 ACTIVE SIGNALS")
            lines.append("-" * 90)
            
            for s in signals:
                risk_pct = abs(s['entry_price'] - s['stop_loss']) / s['entry_price'] * 100
                lines.append(f"""
  {s['signal_type']} - {s['symbol']}
  Entry:     ${s['entry_price']:.6f}
  Stop Loss: ${s['stop_loss']:.6f} ({risk_pct:.1f}% risk)
  RSI:       {s['rsi']:.1f}
  Trend:     {s['trend']}
""")
            lines.append("-" * 90)
        else:
            lines.append("  ⏳ No active signals - monitoring for breakouts...")
            lines.append("")
        
        # Near breakout (approaching signals)
        near = [r for r in results if not r.get('has_signal') and r.get('breakout_pct', 0) >= 60]
        if near:
            lines.append("")
            lines.append("  📊 APPROACHING BREAKOUT (60%+)")
            lines.append("-" * 90)
            lines.append(f"  {'SYMBOL':<12} {'BREAKOUT%':<10} {'RSI':<8} {'TREND':<12} {'24H':<10} {'STATUS'}")
            lines.append("-" * 90)
            
            for r in sorted(near, key=lambda x: x['breakout_pct'], reverse=True):
                momentum = "↑" if r['hist_growing'] and r['rsi_rising'] else "→" if r['hist_growing'] else "↓"
                vol = "VOL✓" if r['volume_ok'] else ""
                lines.append(f"  {r['symbol']:<12} {r['breakout_pct']:<10.0f} {r['rsi']:<8.1f} "
                            f"{r['trend']:<12} {r['change_24h']:<+9.1f}% {momentum} {vol}")
        
        # All coins status
        lines.append("")
        lines.append("  📈 ALL COINS STATUS")
        lines.append("-" * 90)
        lines.append(f"  {'SYMBOL':<12} {'PRICE':<14} {'RSI':<8} {'TREND':<12} {'ATR%':<8} {'BREAKOUT%':<10}")
        lines.append("-" * 90)
        
        for r in sorted(results, key=lambda x: x.get('breakout_pct', 0), reverse=True):
            if 'error' in r:
                lines.append(f"  {r['symbol']:<12} ERROR")
                continue
            
            lines.append(f"  {r['symbol']:<12} ${r['price']:<13.4f} {r['rsi']:<8.1f} "
                        f"{r['trend']:<12} {r['atr_pct']:<8.1f} {r['breakout_pct']:<10.0f}")
        
        lines.append("=" * 90)
        lines.append("")
        lines.append("  Strategy: Swing Trader | WR: 56.9% | PF: 2.49 | Validated on 1 year data")
        lines.append("=" * 90)
        
        return "\n".join(lines)


def main():
    """Run live scanner"""
    print("\n" + "=" * 70)
    print("  INITIALIZING LIVE SWING SCANNER...")
    print("=" * 70)
    
    scanner = LiveScanner()
    
    print("  Scanning coins...")
    results = scanner.scan_all()
    
    print(scanner.format_results(results))


if __name__ == "__main__":
    main()
