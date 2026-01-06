"""
CONTINUOUS BREAKOUT SCANNER
============================
Combineert alle detectie methodes en draait continu.
- Scant elke X minuten
- Categoriseert signalen op urgentie
- Print alleen NIEUWE signalen

Run met: python3 continuous_scanner.py [interval_minutes]
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import time

from core.indicators import Indicators


@dataclass
class Alert:
    """Trading alert"""
    symbol: str
    price: float
    alert_type: str    # 'BREAKOUT', 'VOLUME_EXPLOSION', 'MOMENTUM', 'SETUP'
    urgency: str       # 'HIGH', 'MEDIUM', 'LOW'
    score: float
    change_1h: float
    change_24h: float
    volume_spike: float
    rsi: float
    reason: str
    timestamp: datetime


class ContinuousScanner:
    """Continuous breakout scanner"""
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
        self.seen_alerts: Set[str] = set()  # Track already alerted
        self.last_prices: Dict[str, float] = {}
    
    def get_active_pairs(self, min_volume: float = 50000) -> List[Dict]:
        """Get all active pairs with tickers"""
        try:
            tickers = self.exchange.fetch_tickers()
            pairs = []
            
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if '/USDT:' in symbol or '3L' in symbol or '3S' in symbol or 'UP/' in symbol or 'DOWN/' in symbol:
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
            print(f"Error fetching tickers: {e}")
            return []
    
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < limit // 2:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def analyze_coin(self, symbol: str, ticker: Dict) -> Optional[Alert]:
        """Full analysis of a coin"""
        
        # Fetch 1h and 4h data
        df_1h = self.fetch_ohlcv(symbol, '1h', 30)
        df_4h = self.fetch_ohlcv(symbol, '4h', 30)
        
        if df_1h is None or df_4h is None:
            return None
        
        price = ticker['price']
        
        # === VOLUME ANALYSIS ===
        vol_current = df_1h['volume'].iloc[-1]
        vol_avg = df_1h['volume'].iloc[-15:-1].mean()
        volume_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # === PRICE CHANGES ===
        change_1h = ((price - df_1h['close'].iloc[-2]) / df_1h['close'].iloc[-2]) * 100 if df_1h['close'].iloc[-2] > 0 else 0
        change_4h = ((price - df_4h['close'].iloc[-2]) / df_4h['close'].iloc[-2]) * 100 if df_4h['close'].iloc[-2] > 0 else 0
        change_24h = ticker['change_24h']
        
        # === TECHNICAL INDICATORS (4h) ===
        df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
        rsi = df_4h['rsi'].iloc[-1]
        
        df_4h['macd'], df_4h['signal'], df_4h['hist'] = Indicators.macd(df_4h['close'])
        hist = df_4h['hist'].iloc[-1]
        prev_hist = df_4h['hist'].iloc[-2]
        
        df_4h['ema20'] = Indicators.ema(df_4h['close'], 20)
        above_ema = price > df_4h['ema20'].iloc[-1]
        
        # === BREAKOUT DETECTION ===
        # Recent highs
        high_10 = df_4h['high'].iloc[-10:-1].max()
        high_20 = df_4h['high'].iloc[-20:-1].max()
        
        breaking_10 = price > high_10
        breaking_20 = price > high_20
        distance_from_high = ((high_20 - price) / high_20) * 100 if high_20 > 0 else 0
        
        # === DETERMINE ALERT TYPE ===
        alert_type = None
        urgency = None
        score = 0
        reasons = []
        
        # HIGH URGENCY: Active breakout met volume
        if volume_spike >= 3 and change_1h >= 3 and breaking_10:
            alert_type = 'BREAKOUT'
            urgency = 'HIGH'
            score = 85
            reasons.append(f"Volume {volume_spike:.1f}x")
            reasons.append(f"+{change_1h:.1f}% in 1h")
            if breaking_20:
                reasons.append("Breaking 20-period high!")
                score += 10
        
        # HIGH: Volume explosion
        elif volume_spike >= 5:
            alert_type = 'VOLUME_EXPLOSION'
            urgency = 'HIGH'
            score = 75
            reasons.append(f"Volume {volume_spike:.1f}x surge!")
            if change_1h > 0:
                reasons.append(f"+{change_1h:.1f}% price")
        
        # MEDIUM: Strong momentum building
        elif hist > 0 and hist > prev_hist * 1.5 and rsi > 50 and rsi < 75 and volume_spike >= 1.5:
            alert_type = 'MOMENTUM'
            urgency = 'MEDIUM'
            score = 60
            reasons.append("MACD momentum accelerating")
            if above_ema:
                reasons.append("Above EMA20")
                score += 10
            if distance_from_high < 5:
                reasons.append(f"Near breakout ({distance_from_high:.1f}% from high)")
                score += 10
        
        # MEDIUM: Near breakout with setup
        elif distance_from_high < 3 and volume_spike >= 1.5 and rsi > 50 and rsi < 70:
            alert_type = 'SETUP'
            urgency = 'MEDIUM'
            score = 55
            reasons.append(f"Only {distance_from_high:.1f}% from breakout")
            reasons.append(f"Volume building: {volume_spike:.1f}x")
        
        if alert_type is None:
            return None
        
        return Alert(
            symbol=symbol,
            price=price,
            alert_type=alert_type,
            urgency=urgency,
            score=score,
            change_1h=change_1h,
            change_24h=change_24h,
            volume_spike=volume_spike,
            rsi=rsi,
            reason=" | ".join(reasons),
            timestamp=datetime.now(),
        )
    
    def scan_once(self, min_volume: float = 50000) -> List[Alert]:
        """Run one scan cycle"""
        
        pairs = self.get_active_pairs(min_volume)
        
        # Pre-filter: Focus on coins with activity
        active = [p for p in pairs if abs(p['change_24h']) >= 2 or p['volume_24h'] >= 500000]
        
        alerts = []
        
        for pair in active:
            try:
                alert = self.analyze_coin(pair['symbol'], pair)
                if alert:
                    # Check if we've already seen this alert recently
                    alert_key = f"{alert.symbol}_{alert.alert_type}"
                    if alert_key not in self.seen_alerts:
                        alerts.append(alert)
                        self.seen_alerts.add(alert_key)
            except:
                continue
            
            time.sleep(0.03)
        
        # Sort by score
        alerts.sort(key=lambda x: x.score, reverse=True)
        
        return alerts
    
    def format_alert(self, alert: Alert) -> str:
        """Format single alert"""
        
        if alert.urgency == 'HIGH':
            prefix = "🚨 HIGH"
        elif alert.urgency == 'MEDIUM':
            prefix = "📢 MEDIUM"
        else:
            prefix = "👀 LOW"
        
        return f"""
  {prefix} | {alert.alert_type} | {alert.symbol}
  Price: ${alert.price:.6f} | 1h: {alert.change_1h:+.1f}% | 24h: {alert.change_24h:+.1f}%
  Volume: {alert.volume_spike:.1f}x | RSI: {alert.rsi:.0f} | Score: {alert.score}/100
  → {alert.reason}
"""
    
    def run_continuous(self, interval_minutes: int = 5, min_volume: float = 50000):
        """Run continuous scanning"""
        
        print("\n" + "=" * 80)
        print(f"  CONTINUOUS BREAKOUT SCANNER")
        print(f"  Scanning every {interval_minutes} minutes | Min volume: ${min_volume:,.0f}")
        print("=" * 80)
        
        cycle = 0
        
        while True:
            cycle += 1
            scan_time = datetime.now().strftime('%H:%M:%S')
            
            print(f"\n{'─'*80}")
            print(f"  Scan #{cycle} - {scan_time}")
            print(f"{'─'*80}")
            
            try:
                alerts = self.scan_once(min_volume)
                
                if alerts:
                    # Separate by urgency
                    high = [a for a in alerts if a.urgency == 'HIGH']
                    medium = [a for a in alerts if a.urgency == 'MEDIUM']
                    
                    if high:
                        print("\n  🔥 HIGH URGENCY ALERTS:")
                        print("-" * 80)
                        for a in high:
                            print(self.format_alert(a))
                    
                    if medium:
                        print("\n  📊 MEDIUM URGENCY:")
                        print("-" * 80)
                        for a in medium[:10]:
                            print(self.format_alert(a))
                    
                    print(f"\n  Total new alerts: {len(alerts)} (High: {len(high)}, Medium: {len(medium)})")
                else:
                    print("  No new signals detected.")
                
                # Clean old alerts every 10 cycles (to re-alert if conditions persist)
                if cycle % 10 == 0:
                    self.seen_alerts.clear()
                    print("  (Cleared alert memory - will re-alert on persistent signals)")
                
            except Exception as e:
                print(f"  Error during scan: {e}")
            
            print(f"\n  Next scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)


def single_scan():
    """Run a single scan"""
    
    print("\n" + "=" * 80)
    print("  BREAKOUT SCANNER - Single Scan")
    print("=" * 80)
    
    scanner = ContinuousScanner()
    alerts = scanner.scan_once(min_volume=50000)
    
    if not alerts:
        print("\n  No strong signals at this moment.")
        return
    
    # Group by urgency
    high = [a for a in alerts if a.urgency == 'HIGH']
    medium = [a for a in alerts if a.urgency == 'MEDIUM']
    
    if high:
        print("\n  🚨 HIGH URGENCY - ACT NOW")
        print("=" * 80)
        for a in high:
            print(scanner.format_alert(a))
    
    if medium:
        print("\n  📊 MEDIUM URGENCY - WATCH CLOSELY")
        print("=" * 80)
        for a in medium:
            print(scanner.format_alert(a))
    
    print("\n" + "=" * 80)
    print(f"  Summary: {len(high)} HIGH, {len(medium)} MEDIUM alerts")
    print("=" * 80)


def main():
    """Main entry point"""
    
    # Check for continuous mode
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
            scanner = ContinuousScanner()
            scanner.run_continuous(interval_minutes=interval)
        except ValueError:
            print("Usage: python3 continuous_scanner.py [interval_minutes]")
            print("       Omit interval for single scan")
    else:
        single_scan()


if __name__ == "__main__":
    main()
