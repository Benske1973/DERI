"""
BREAKOUT SCANNER DASHBOARD
===========================
All-in-one scanner die alle methodes combineert.

MODES:
- hunt    : Optimized hunt (oversold + momentum)
- active  : Active breakouts happening NOW
- volume  : Volume explosions (15min)
- mega    : Recent mega movers
- all     : Run all scans

Usage: python3 scanner_dashboard.py [mode]
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


class Scanner:
    """Combined scanner"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
        self.tickers = {}
    
    def refresh_tickers(self, min_volume: float = 50000) -> List[Dict]:
        """Refresh all tickers"""
        try:
            self.tickers = self.exchange.fetch_tickers()
            pairs = []
            
            for symbol, ticker in self.tickers.items():
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
        """Fetch OHLCV"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit // 2:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    # === HUNT MODE ===
    def hunt_breakouts(self, pairs: List[Dict]) -> List[Dict]:
        """Optimized breakout hunting (oversold + momentum)"""
        signals = []
        
        for pair in pairs:
            try:
                df = self.fetch_ohlcv(pair['symbol'], '4h', 50)
                if df is None or len(df) < 30:
                    continue
                
                price = pair['price']
                
                # RSI
                df['rsi'] = Indicators.rsi(df['close'], 14)
                rsi = df['rsi'].iloc[-1]
                rsi_prev = df['rsi'].iloc[-2]
                
                # MACD
                df['macd'], df['signal'], df['hist'] = Indicators.macd(df['close'])
                hist = df['hist'].iloc[-1]
                hist_prev = df['hist'].iloc[-2]
                
                # Price levels
                low_7d = df['low'].iloc[-42:].min()
                distance_from_low = ((price - low_7d) / low_7d) * 100
                
                # Volume
                vol_current = df['volume'].iloc[-1]
                vol_avg = df['volume'].iloc[-10:-1].mean()
                vol_spike = vol_current / vol_avg if vol_avg > 0 else 1
                
                # Scoring
                score = 0
                reasons = []
                
                # Oversold RSI (key signal!)
                if rsi < 35:
                    score += 40
                    reasons.append(f"Very oversold RSI: {rsi:.0f}")
                elif rsi < 40:
                    score += 30
                    reasons.append(f"Oversold RSI: {rsi:.0f}")
                
                # Recovery
                if rsi < 45 and rsi > rsi_prev and df['rsi'].iloc[-5:].min() < 35:
                    score += 20
                    reasons.append("Recovery starting!")
                
                # MACD flip
                if hist > 0 and hist_prev <= 0:
                    score += 25
                    reasons.append("MACD flip!")
                
                # Volume
                if vol_spike > 2:
                    score += 15
                    reasons.append(f"Vol: {vol_spike:.1f}x")
                
                # Near bottom
                if distance_from_low < 10:
                    score += 10
                    reasons.append(f"{distance_from_low:.0f}% from 7d low")
                
                if score >= 40:
                    signals.append({
                        'symbol': pair['symbol'],
                        'price': price,
                        'score': score,
                        'rsi': rsi,
                        'vol_spike': vol_spike,
                        'change_24h': pair['change_24h'],
                        'distance_from_low': distance_from_low,
                        'reasons': reasons,
                    })
            except:
                continue
            time.sleep(0.02)
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:20]
    
    # === ACTIVE MODE ===
    def find_active_breakouts(self, pairs: List[Dict]) -> List[Dict]:
        """Find breakouts happening RIGHT NOW"""
        signals = []
        
        # Pre-filter to movers
        movers = [p for p in pairs if p['change_24h'] >= 3 or p['volume_24h'] > 500000]
        
        for pair in movers:
            try:
                df = self.fetch_ohlcv(pair['symbol'], '1h', 30)
                df_4h = self.fetch_ohlcv(pair['symbol'], '4h', 30)
                
                if df is None or df_4h is None:
                    continue
                
                price = pair['price']
                
                # Volume spike
                vol_current = df['volume'].iloc[-1]
                vol_avg = df['volume'].iloc[-15:-1].mean()
                vol_spike = vol_current / vol_avg if vol_avg > 0 else 1
                
                # Price change 1h
                change_1h = ((price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                # Breaking resistance?
                high_recent = df_4h['high'].iloc[-10:-1].max()
                breaking = price > high_recent
                
                # RSI
                df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
                rsi = df_4h['rsi'].iloc[-1]
                
                # Active breakout criteria
                if vol_spike >= 3 and change_1h >= 2:
                    score = 80
                    if breaking:
                        score += 15
                    
                    signals.append({
                        'symbol': pair['symbol'],
                        'price': price,
                        'score': score,
                        'vol_spike': vol_spike,
                        'change_1h': change_1h,
                        'change_24h': pair['change_24h'],
                        'rsi': rsi,
                        'breaking': breaking,
                    })
            except:
                continue
            time.sleep(0.02)
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:15]
    
    # === VOLUME MODE ===
    def find_volume_explosions(self, pairs: List[Dict]) -> List[Dict]:
        """Find 15-minute volume explosions"""
        signals = []
        
        for pair in pairs[:200]:  # Top 200 by volume
            try:
                df = self.fetch_ohlcv(pair['symbol'], '15m', 20)
                if df is None or len(df) < 10:
                    continue
                
                vol_current = df['volume'].iloc[-1]
                vol_avg = df['volume'].iloc[-10:-1].mean()
                vol_spike = vol_current / vol_avg if vol_avg > 0 else 1
                
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                if vol_spike >= 3:
                    signals.append({
                        'symbol': pair['symbol'],
                        'price': pair['price'],
                        'vol_spike': vol_spike,
                        'change_15m': price_change,
                        'change_24h': pair['change_24h'],
                        'bullish': price_change > 0,
                    })
            except:
                continue
            time.sleep(0.02)
        
        signals.sort(key=lambda x: x['vol_spike'], reverse=True)
        return signals[:15]
    
    def print_header(self, title: str):
        """Print section header"""
        print("\n" + "=" * 100)
        print(f"  {title}")
        print("=" * 100)
    
    def run_hunt(self):
        """Run hunt mode"""
        self.print_header("🎯 BREAKOUT HUNT - Oversold coins with momentum potential")
        
        pairs = self.refresh_tickers(50000)
        print(f"Analyzing {len(pairs)} pairs...")
        
        signals = self.hunt_breakouts([p for p in pairs if p['change_24h'] < 30])
        
        if not signals:
            print("  No signals found.")
            return
        
        print("\n  TOP HUNT SIGNALS:")
        print("-" * 100)
        
        for s in signals:
            indicator = "🔥" if s['score'] >= 60 else "📈"
            print(f"\n  {indicator} {s['symbol']:<14} | Score: {s['score']}/100")
            print(f"     Price: ${s['price']:.6f} | RSI: {s['rsi']:.0f} | Vol: {s['vol_spike']:.1f}x | "
                  f"24h: {s['change_24h']:+.1f}% | From low: {s['distance_from_low']:.0f}%")
            print(f"     → {' | '.join(s['reasons'])}")
    
    def run_active(self):
        """Run active breakouts mode"""
        self.print_header("🚀 ACTIVE BREAKOUTS - Happening NOW")
        
        pairs = self.refresh_tickers(50000)
        print(f"Scanning {len(pairs)} pairs...")
        
        signals = self.find_active_breakouts(pairs)
        
        if not signals:
            print("  No active breakouts at this moment.")
            return
        
        print("\n  ACTIVE BREAKOUTS:")
        print("-" * 100)
        print(f"  {'SYMBOL':<14} {'SCORE':<8} {'1H':<10} {'24H':<10} {'VOLUME':<12} {'RSI':<8} {'BREAKING'}")
        print("-" * 100)
        
        for s in signals:
            breaking = "✓ YES" if s['breaking'] else ""
            print(f"  {s['symbol']:<14} {s['score']:<8} {s['change_1h']:<+10.1f}% "
                  f"{s['change_24h']:<+10.1f}% {s['vol_spike']:<12.1f}x {s['rsi']:<8.0f} {breaking}")
    
    def run_volume(self):
        """Run volume explosion mode"""
        self.print_header("💥 VOLUME EXPLOSIONS - 15min timeframe")
        
        pairs = self.refresh_tickers(100000)
        print(f"Scanning {min(200, len(pairs))} pairs...")
        
        signals = self.find_volume_explosions(pairs)
        
        if not signals:
            print("  No volume explosions detected.")
            return
        
        print("\n  VOLUME EXPLOSIONS:")
        print("-" * 100)
        print(f"  {'SYMBOL':<14} {'VOL SPIKE':<12} {'15M':<12} {'24H':<10} {'DIRECTION'}")
        print("-" * 100)
        
        for s in signals:
            direction = "🟢 BULLISH" if s['bullish'] else "🔴 BEARISH"
            print(f"  {s['symbol']:<14} {s['vol_spike']:<12.1f}x {s['change_15m']:<+12.2f}% "
                  f"{s['change_24h']:<+10.1f}% {direction}")
    
    def run_all(self):
        """Run all scans"""
        print("\n" + "=" * 100)
        print(f"  📊 FULL MARKET SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 100)
        
        self.run_active()
        self.run_hunt()
        self.run_volume()
        
        self.print_header("✅ SCAN COMPLETE")


def main():
    """Main entry point"""
    
    mode = 'all'
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    
    scanner = Scanner()
    
    if mode == 'hunt':
        scanner.run_hunt()
    elif mode == 'active':
        scanner.run_active()
    elif mode == 'volume':
        scanner.run_volume()
    elif mode == 'all':
        scanner.run_all()
    else:
        print(f"Unknown mode: {mode}")
        print("Available modes: hunt, active, volume, all")


if __name__ == "__main__":
    main()
