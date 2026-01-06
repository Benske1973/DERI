"""
🚀 SUPERTRADER V2 - Ultimate Breakout Trading System
=====================================================
Combines ALL learnings from mega mover analysis.
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
class Signal:
    symbol: str
    price: float
    signal_type: str  # BREAKOUT, BOUNCE, ACCUMULATION, MOMENTUM
    score: float
    
    # Entry/Exit
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    
    # Risk metrics
    risk_pct: float
    reward_pct: float
    rr_ratio: float
    
    # Technical
    rsi: float
    macd_status: str
    volume_spike: float
    trend: str
    
    # Reason
    reasons: List[str]


class SuperTraderV2:
    """Improved SuperTrader"""
    
    def __init__(self):
        self.exchange = ccxt.kucoin()
        self.exchange.load_markets()
    
    def fetch_data(self, symbol: str, timeframe: str = '4h', limit: int = 60) -> Optional[pd.DataFrame]:
        """Fetch OHLCV with error handling"""
        try:
            time.sleep(0.05)  # Rate limit
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 30:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except:
            return None
    
    def get_pairs(self, min_volume: float = 100000) -> List[Dict]:
        """Get active trading pairs"""
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
                        'volume': vol,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            return sorted(pairs, key=lambda x: x['volume'], reverse=True)
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def analyze(self, symbol: str, ticker: Dict) -> Optional[Signal]:
        """Analyze a coin for trading signals"""
        
        df = self.fetch_data(symbol, '4h', 60)
        if df is None:
            return None
        
        price = ticker['price']
        
        # === INDICATORS ===
        # RSI
        rsi_series = Indicators.rsi(df['close'], 14)
        rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50
        rsi_prev = float(rsi_series.iloc[-2]) if not pd.isna(rsi_series.iloc[-2]) else 50
        
        # MACD
        macd, signal_line, hist = Indicators.macd(df['close'])
        curr_hist = float(hist.iloc[-1]) if not pd.isna(hist.iloc[-1]) else 0
        prev_hist = float(hist.iloc[-2]) if not pd.isna(hist.iloc[-2]) else 0
        
        macd_rising = curr_hist > prev_hist
        macd_positive = curr_hist > 0
        macd_cross = curr_hist > 0 and prev_hist <= 0
        
        if macd_cross:
            macd_status = "CROSS"
        elif macd_positive and macd_rising:
            macd_status = "BULLISH"
        elif macd_rising:
            macd_status = "RISING"
        else:
            macd_status = "FALLING"
        
        # ATR
        atr_series = Indicators.atr(df['high'], df['low'], df['close'], 14)
        atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else price * 0.03
        
        # EMAs
        ema20 = Indicators.ema(df['close'], 20)
        ema50 = Indicators.ema(df['close'], 50)
        
        above_ema20 = price > float(ema20.iloc[-1])
        above_ema50 = price > float(ema50.iloc[-1])
        ema_aligned = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        
        trend_score = sum([above_ema20, above_ema50, ema_aligned])
        trend = "STRONG" if trend_score == 3 else "UP" if trend_score >= 2 else "WEAK" if trend_score == 1 else "DOWN"
        
        # Volume
        vol_current = float(df['volume'].iloc[-3:].mean())
        vol_avg = float(df['volume'].iloc[-20:-3].mean())
        volume_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Volume trend (for accumulation)
        vol_old = float(df['volume'].iloc[-30:-15].mean()) if len(df) >= 30 else vol_avg
        vol_new = float(df['volume'].iloc[-15:].mean())
        vol_growth = vol_new / vol_old if vol_old > 0 else 1
        
        # Price levels
        high_20 = float(df['high'].iloc[-20:-1].max())
        low_20 = float(df['low'].iloc[-20:].min())
        
        dist_from_high = ((high_20 - price) / price) * 100 if price > 0 else 0
        dist_from_low = ((price - low_20) / low_20) * 100 if low_20 > 0 else 0
        
        breaking = price > high_20
        near_break = dist_from_high < 5
        
        # === SIGNAL DETECTION ===
        signal_type = None
        score = 0
        reasons = []
        
        # BOUNCE - Oversold reversal
        if rsi < 40:
            signal_type = "BOUNCE"
            if rsi < 25:
                score += 40
                reasons.append(f"RSI {rsi:.0f} extreme oversold")
            elif rsi < 30:
                score += 30
                reasons.append(f"RSI {rsi:.0f} very oversold")
            else:
                score += 20
                reasons.append(f"RSI {rsi:.0f} oversold")
            
            if rsi > rsi_prev:
                score += 15
                reasons.append("Recovery starting")
            
            if macd_rising:
                score += 15
                reasons.append("MACD turning")
            
            if volume_spike > 2:
                score += 10
                reasons.append(f"Volume {volume_spike:.1f}x")
        
        # BREAKOUT - Breaking resistance
        elif breaking or (near_break and macd_rising) or macd_cross:
            signal_type = "BREAKOUT"
            
            if breaking:
                score += 30
                reasons.append("Breaking resistance!")
            elif near_break:
                score += 20
                reasons.append(f"Near breakout ({dist_from_high:.1f}%)")
            
            if macd_cross:
                score += 25
                reasons.append("MACD cross!")
            elif macd_status == "BULLISH":
                score += 15
                reasons.append("MACD bullish")
            
            if volume_spike > 2:
                score += 20
                reasons.append(f"Volume {volume_spike:.1f}x")
            elif volume_spike > 1.5:
                score += 10
                reasons.append(f"Volume building")
            
            if trend_score >= 2:
                score += 10
                reasons.append("Trend aligned")
        
        # ACCUMULATION - Volume building, price stable
        elif vol_growth > 1.5 and abs(ticker['change_24h']) < 10:
            signal_type = "ACCUMULATION"
            
            if vol_growth > 3:
                score += 30
                reasons.append(f"Strong accumulation {vol_growth:.1f}x")
            elif vol_growth > 2:
                score += 20
                reasons.append(f"Accumulation {vol_growth:.1f}x")
            else:
                score += 10
                reasons.append(f"Volume building {vol_growth:.1f}x")
            
            if near_break:
                score += 20
                reasons.append(f"Near breakout")
            
            if trend_score >= 2:
                score += 15
                reasons.append("Uptrend")
        
        # MOMENTUM - Strong trend continuation
        elif macd_positive and macd_rising and trend_score >= 2:
            signal_type = "MOMENTUM"
            
            score += 20
            reasons.append("Momentum building")
            
            if trend == "STRONG":
                score += 15
                reasons.append("Strong trend")
            
            if volume_spike > 1.5:
                score += 10
                reasons.append(f"Volume {volume_spike:.1f}x")
            
            if 55 < rsi < 70:
                score += 10
                reasons.append(f"RSI {rsi:.0f} healthy")
        
        # No signal
        if signal_type is None or score < 30:
            return None
        
        # === ENTRY/EXIT LEVELS ===
        entry = price
        
        if signal_type == "BOUNCE":
            stop_loss = low_20 * 0.98
        else:
            stop_loss = price - (2 * atr)
        
        risk = entry - stop_loss
        risk_pct = (risk / entry) * 100 if entry > 0 else 5
        
        tp1 = entry + (risk * 2)    # 2:1 R:R
        tp2 = entry + (risk * 4)    # 4:1 R:R
        tp3 = entry + (risk * 8)    # 8:1 R:R (moon)
        
        reward_pct = ((tp2 - entry) / entry) * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        return Signal(
            symbol=symbol,
            price=price,
            signal_type=signal_type,
            score=score,
            entry=entry,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            rr_ratio=rr_ratio,
            rsi=rsi,
            macd_status=macd_status,
            volume_spike=volume_spike,
            trend=trend,
            reasons=reasons,
        )
    
    def hunt(self, max_coins: int = 200) -> List[Signal]:
        """Hunt for signals"""
        
        print("\n" + "=" * 100)
        print("  🚀 SUPERTRADER V2 - Hunting...")
        print("=" * 100)
        
        pairs = self.get_pairs(min_volume=100000)
        print(f"  Found {len(pairs)} pairs, scanning top {min(len(pairs), max_coins)}...")
        
        signals = []
        scanned = 0
        
        for pair in pairs[:max_coins]:
            scanned += 1
            if scanned % 50 == 0:
                print(f"  Progress: {scanned}/{min(len(pairs), max_coins)}")
            
            signal = self.analyze(pair['symbol'], pair)
            if signal:
                signals.append(signal)
        
        signals.sort(key=lambda x: x.score, reverse=True)
        
        print(f"  Found {len(signals)} signals!")
        return signals
    
    def display(self, signals: List[Signal]):
        """Display signals"""
        
        if not signals:
            print("\n  No signals found.")
            return
        
        print("\n" + "=" * 120)
        print(f"  🎯 SUPERTRADER SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 120)
        
        # Group by type
        by_type = {}
        for s in signals:
            if s.signal_type not in by_type:
                by_type[s.signal_type] = []
            by_type[s.signal_type].append(s)
        
        # Best signals (score >= 50)
        best = [s for s in signals if s.score >= 50]
        if best:
            print("\n  🔥 TOP SIGNALS (Score 50+)")
            print("-" * 120)
            for s in best[:10]:
                self._print_signal(s)
        
        # By type
        for sig_type in ["BREAKOUT", "BOUNCE", "ACCUMULATION", "MOMENTUM"]:
            type_sigs = by_type.get(sig_type, [])
            remaining = [s for s in type_sigs if s.score < 50]
            if remaining:
                print(f"\n  {sig_type} SIGNALS ({len(type_sigs)} total)")
                print("-" * 120)
                for s in remaining[:5]:
                    self._print_signal(s)
        
        # Summary
        print("\n" + "=" * 120)
        print(f"  📊 SUMMARY: {len(signals)} signals | Top: {len(best)}")
        for t, sigs in by_type.items():
            print(f"     {t}: {len(sigs)}")
        print("=" * 120)
    
    def _print_signal(self, s: Signal):
        """Print single signal"""
        emoji = {"BREAKOUT": "🚀", "BOUNCE": "📉", "ACCUMULATION": "📊", "MOMENTUM": "📈"}
        strength = "🔥" if s.score >= 70 else "⚡" if s.score >= 50 else "👀"
        
        print(f"""
  {emoji.get(s.signal_type, '')} {s.symbol:<14} | {s.signal_type:<12} | Score: {s.score:.0f} {strength}
  ├─ Entry: ${s.entry:.6f} | SL: ${s.stop_loss:.6f} (-{s.risk_pct:.1f}%)
  ├─ TP1: ${s.tp1:.6f} | TP2: ${s.tp2:.6f} | TP3: ${s.tp3:.6f}
  ├─ R:R: {s.rr_ratio:.1f}:1 | RSI: {s.rsi:.0f} | MACD: {s.macd_status} | Vol: {s.volume_spike:.1f}x | Trend: {s.trend}
  └─ {' | '.join(s.reasons)}
""")
    
    def trade_plan(self, s: Signal, capital: float = 10000) -> str:
        """Generate trade plan"""
        
        risk_amount = capital * 0.02  # 2% risk
        position_size = risk_amount / (s.risk_pct / 100)
        position_size = min(position_size, capital * 0.2)  # Max 20%
        
        return f"""
╔{'═' * 78}╗
║  TRADE PLAN: {s.symbol:<63}║
╠{'═' * 78}╣
║  Signal: {s.signal_type:<12} | Score: {s.score:.0f}/100 | R:R: {s.rr_ratio:.1f}:1{' ' * 28}║
╠{'═' * 78}╣
║  ENTRY: ${s.entry:<12.6f}                                                     ║
║  Position: ${position_size:<10.2f} (2% risk = ${risk_amount:.2f}){' ' * 30}║
╠{'═' * 78}╣
║  EXIT STRATEGY:                                                              ║
║  ├─ Stop Loss: ${s.stop_loss:<12.6f} (-{s.risk_pct:.1f}%){' ' * 38}║
║  ├─ TP1 (50%): ${s.tp1:<12.6f} (+{((s.tp1/s.entry)-1)*100:.1f}%){' ' * 37}║
║  ├─ TP2 (30%): ${s.tp2:<12.6f} (+{((s.tp2/s.entry)-1)*100:.1f}%){' ' * 36}║
║  └─ TP3 (20%): ${s.tp3:<12.6f} (+{((s.tp3/s.entry)-1)*100:.1f}%){' ' * 35}║
╠{'═' * 78}╣
║  TECHNICAL: RSI {s.rsi:.0f} | MACD {s.macd_status:<8} | Vol {s.volume_spike:.1f}x | Trend {s.trend:<8}{' ' * 9}║
╚{'═' * 78}╝
"""


def main():
    trader = SuperTraderV2()
    signals = trader.hunt(max_coins=200)
    trader.display(signals)
    
    # Trade plans for top 3
    best = [s for s in signals if s.score >= 50][:3]
    if best:
        print("\n" + "=" * 80)
        print("  📋 TRADE PLANS")
        print("=" * 80)
        for s in best:
            print(trader.trade_plan(s))
    
    return trader, signals


if __name__ == "__main__":
    main()
