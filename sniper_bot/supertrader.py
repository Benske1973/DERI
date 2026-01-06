"""
🚀 SUPERTRADER - Ultimate Breakout Trading System
==================================================
Combineert ALLE learnings:
- Mega mover patterns (51% had oversold RSI)
- Accumulation detection (volume divergence)
- Momentum breakouts (MACD + volume)
- Bounce plays (oversold reversals)
- Smart position sizing & risk management

STRATEGY MODES:
1. HUNTER   - Zoekt naar nieuwe setups
2. SNIPER   - Wacht op perfecte entry
3. EXECUTOR - Managed open posities

PROVEN PATTERNS (from 142 mega movers):
- 51% had RSI < 40 before move
- Average 6.8 days to peak
- Volume spike at/after move start
- 35% bottom reversals, 35% resistance breaks
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time
import json

from core.indicators import Indicators


class SignalType(Enum):
    BREAKOUT = "BREAKOUT"
    BOUNCE = "BOUNCE"
    ACCUMULATION = "ACCUMULATION"
    MOMENTUM = "MOMENTUM"


class SignalStrength(Enum):
    STRONG = "STRONG"      # Score 80+
    MEDIUM = "MEDIUM"      # Score 60-80
    WEAK = "WEAK"          # Score 40-60


@dataclass
class TradingSignal:
    """Complete trading signal with entry/exit"""
    symbol: str
    signal_type: SignalType
    strength: SignalStrength
    score: float
    
    # Entry
    entry_price: float
    entry_reason: str
    
    # Exit levels
    stop_loss: float
    take_profit_1: float  # Conservative
    take_profit_2: float  # Aggressive
    take_profit_3: float  # Moon
    
    # Risk/Reward
    risk_pct: float
    reward_pct: float
    rr_ratio: float
    
    # Technical data
    rsi: float
    macd_hist: float
    volume_spike: float
    trend_score: int
    
    # Meta
    timestamp: str = field(default_factory=lambda: str(datetime.now()))
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'type': self.signal_type.value,
            'strength': self.strength.value,
            'score': self.score,
            'entry': self.entry_price,
            'sl': self.stop_loss,
            'tp1': self.take_profit_1,
            'tp2': self.take_profit_2,
            'tp3': self.take_profit_3,
            'rr': self.rr_ratio,
            'rsi': self.rsi,
            'timestamp': self.timestamp,
        }


@dataclass
class Position:
    """Open position tracking"""
    symbol: str
    entry_price: float
    entry_time: str
    size: float  # In quote currency
    stop_loss: float
    take_profit: float
    current_price: float = 0
    pnl_pct: float = 0
    highest_price: float = 0
    trailing_stop: float = 0
    status: str = "OPEN"


class SuperTrader:
    """The Ultimate Trading System"""
    
    # Configuration
    CONFIG = {
        # Risk Management
        'max_risk_per_trade': 0.02,      # 2% risk per trade
        'max_positions': 5,               # Max concurrent positions
        'portfolio_size': 10000,          # Starting capital
        
        # Entry Criteria
        'min_score_strong': 80,
        'min_score_medium': 60,
        'min_score_entry': 70,            # Minimum for actual entry
        
        # Exit Strategy
        'default_sl_atr': 2.0,            # Stop loss in ATR
        'tp1_rr': 2.0,                    # Take profit 1: 2:1 R:R
        'tp2_rr': 4.0,                    # Take profit 2: 4:1 R:R
        'tp3_rr': 8.0,                    # Take profit 3: 8:1 R:R (moon)
        
        # Trailing Stop
        'trail_activation': 1.5,          # Activate after 1.5x risk profit
        'trail_distance': 0.8,            # Trail at 0.8x ATR
        
        # Filters
        'min_volume_24h': 100000,
        'min_rsi_bounce': 35,
        'max_rsi_breakout': 75,
    }
    
    def __init__(self, exchange_id: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.exchange.load_markets()
        self.positions: List[Position] = []
        self.signals: List[TradingSignal] = []
        self.trade_history: List[dict] = []
    
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
    
    def get_active_pairs(self, min_volume: float = None) -> List[Dict]:
        """Get all active USDT pairs"""
        min_vol = min_volume or self.CONFIG['min_volume_24h']
        
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
                
                if vol >= min_vol and price > 0:
                    pairs.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': vol,
                        'change_24h': ticker.get('percentage', 0) or 0,
                    })
            
            return pairs
        except Exception as e:
            print(f"Error fetching pairs: {e}")
            return []
    
    def analyze_coin(self, symbol: str, ticker: Dict) -> Optional[TradingSignal]:
        """Complete analysis of a single coin"""
        
        df_4h = self.fetch_ohlcv(symbol, '4h', 60)
        df_1h = self.fetch_ohlcv(symbol, '1h', 48)
        
        if df_4h is None or df_1h is None or len(df_4h) < 40:
            return None
        
        price = ticker['price']
        
        # === INDICATORS ===
        # RSI
        df_4h['rsi'] = Indicators.rsi(df_4h['close'], 14)
        rsi = df_4h['rsi'].iloc[-1]
        rsi_prev = df_4h['rsi'].iloc[-2]
        
        # MACD
        df_4h['macd'], df_4h['signal'], df_4h['hist'] = Indicators.macd(df_4h['close'])
        hist = df_4h['hist'].iloc[-1]
        hist_prev = df_4h['hist'].iloc[-2]
        hist_prev2 = df_4h['hist'].iloc[-3]
        
        macd_cross = hist > 0 and hist_prev <= 0
        macd_rising = hist > hist_prev
        macd_accelerating = hist > hist_prev > hist_prev2
        
        # ATR
        atr = Indicators.atr(df_4h, 14).iloc[-1]
        atr_pct = (atr / price) * 100
        
        # EMAs
        df_4h['ema20'] = Indicators.ema(df_4h['close'], 20)
        df_4h['ema50'] = Indicators.ema(df_4h['close'], 50)
        
        above_ema20 = price > df_4h['ema20'].iloc[-1]
        above_ema50 = price > df_4h['ema50'].iloc[-1]
        ema20_above_50 = df_4h['ema20'].iloc[-1] > df_4h['ema50'].iloc[-1]
        
        trend_score = sum([above_ema20, above_ema50, ema20_above_50])
        
        # Volume
        vol_current = df_1h['volume'].iloc[-3:].mean()
        vol_avg = df_1h['volume'].iloc[-24:-3].mean()
        volume_spike = vol_current / vol_avg if vol_avg > 0 else 1
        
        # Volume trend (accumulation detection)
        vol_early = df_4h['volume'].iloc[-20:-10].mean()
        vol_recent = df_4h['volume'].iloc[-10:].mean()
        vol_trend = vol_recent / vol_early if vol_early > 0 else 1
        
        # Price levels
        high_20 = df_4h['high'].iloc[-20:-1].max()
        low_20 = df_4h['low'].iloc[-20:].min()
        low_7d = df_4h['low'].iloc[-42:].min()
        
        dist_from_high = ((high_20 - price) / price) * 100
        dist_from_low = ((price - low_20) / low_20) * 100 if low_20 > 0 else 0
        
        breaking_resistance = price > high_20
        near_breakout = dist_from_high < 5
        
        # === SIGNAL DETECTION ===
        signal_type = None
        score = 0
        reasons = []
        
        # 1. BOUNCE SIGNAL (oversold reversal)
        if rsi < 45:
            signal_type = SignalType.BOUNCE
            
            # RSI scoring
            if rsi < 25:
                score += 40
                reasons.append(f"Extreme oversold RSI {rsi:.0f}")
            elif rsi < 30:
                score += 35
                reasons.append(f"Very oversold RSI {rsi:.0f}")
            else:
                score += 25
                reasons.append(f"Oversold RSI {rsi:.0f}")
            
            # Recovery detection
            if rsi > rsi_prev:
                score += 15
                reasons.append("Recovery starting")
            
            # MACD turning
            if macd_rising and hist < 0:
                score += 20
                reasons.append("MACD turning up")
            
            # Volume
            if volume_spike > 2:
                score += 15
                reasons.append(f"Volume {volume_spike:.1f}x")
            
            # Near support
            if dist_from_low < 10:
                score += 10
                reasons.append("Near support")
        
        # 2. BREAKOUT SIGNAL
        elif breaking_resistance or near_breakout or (macd_cross and volume_spike > 1.5):
            signal_type = SignalType.BREAKOUT
            
            if breaking_resistance:
                score += 30
                reasons.append("Breaking resistance!")
            else:
                score += 20
                reasons.append(f"Near breakout ({dist_from_high:.1f}%)")
            
            # MACD
            if macd_cross:
                score += 25
                reasons.append("MACD cross!")
            elif macd_accelerating:
                score += 15
                reasons.append("MACD accelerating")
            
            # Volume confirmation
            if volume_spike > 3:
                score += 25
                reasons.append(f"Strong volume {volume_spike:.1f}x")
            elif volume_spike > 1.5:
                score += 15
                reasons.append(f"Volume {volume_spike:.1f}x")
            
            # Trend alignment
            if trend_score >= 2:
                score += 15
                reasons.append("Trend aligned")
            
            # RSI sweet spot
            if 50 < rsi < 70:
                score += 10
                reasons.append(f"RSI {rsi:.0f} in range")
        
        # 3. ACCUMULATION SIGNAL
        elif vol_trend > 1.3 and abs(ticker['change_24h']) < 15:
            signal_type = SignalType.ACCUMULATION
            
            if vol_trend > 3:
                score += 35
                reasons.append(f"Strong accumulation {vol_trend:.1f}x")
            elif vol_trend > 2:
                score += 25
                reasons.append(f"Accumulation {vol_trend:.1f}x")
            else:
                score += 15
                reasons.append(f"Volume building {vol_trend:.1f}x")
            
            # Near breakout
            if near_breakout:
                score += 25
                reasons.append(f"Near breakout ({dist_from_high:.1f}%)")
            
            # Trend
            if trend_score >= 2:
                score += 20
                reasons.append("Uptrend")
            
            # RSI
            if 45 < rsi < 65:
                score += 15
                reasons.append(f"RSI {rsi:.0f} neutral")
        
        # 4. MOMENTUM SIGNAL
        elif macd_rising and hist > 0 and rsi > 50:
            signal_type = SignalType.MOMENTUM
            
            if macd_accelerating:
                score += 25
                reasons.append("Momentum accelerating")
            else:
                score += 15
                reasons.append("Momentum building")
            
            # Trend
            if trend_score == 3:
                score += 25
                reasons.append("Strong uptrend")
            elif trend_score >= 2:
                score += 15
                reasons.append("Uptrend")
            
            # Volume
            if volume_spike > 1.5:
                score += 15
                reasons.append(f"Volume {volume_spike:.1f}x")
            
            # RSI
            if 55 < rsi < 70:
                score += 10
                reasons.append(f"RSI {rsi:.0f}")
        
        # No valid signal
        if signal_type is None or score < 30:
            return None
        
        # === CALCULATE ENTRY/EXIT ===
        entry_price = price
        
        # Stop loss based on signal type
        if signal_type == SignalType.BOUNCE:
            stop_loss = low_20 * 0.98  # Below recent low
        else:
            stop_loss = price - (self.CONFIG['default_sl_atr'] * atr)
        
        risk = entry_price - stop_loss
        risk_pct = (risk / entry_price) * 100
        
        # Take profit levels
        tp1 = entry_price + (risk * self.CONFIG['tp1_rr'])
        tp2 = entry_price + (risk * self.CONFIG['tp2_rr'])
        tp3 = entry_price + (risk * self.CONFIG['tp3_rr'])
        
        reward_pct = ((tp2 - entry_price) / entry_price) * 100
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
        
        # Determine strength
        if score >= self.CONFIG['min_score_strong']:
            strength = SignalStrength.STRONG
        elif score >= self.CONFIG['min_score_medium']:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK
        
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            score=score,
            entry_price=entry_price,
            entry_reason=" | ".join(reasons),
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_pct=risk_pct,
            reward_pct=reward_pct,
            rr_ratio=rr_ratio,
            rsi=rsi,
            macd_hist=hist,
            volume_spike=volume_spike,
            trend_score=trend_score,
        )
    
    def hunt(self, max_coins: int = 200) -> List[TradingSignal]:
        """Hunt for trading signals"""
        
        print("\n" + "=" * 80)
        print("  🎯 SUPERTRADER - Hunting for signals...")
        print("=" * 80)
        
        pairs = self.get_active_pairs()
        print(f"  Scanning {min(len(pairs), max_coins)} pairs...")
        
        signals = []
        
        for pair in pairs[:max_coins]:
            try:
                signal = self.analyze_coin(pair['symbol'], pair)
                if signal:
                    signals.append(signal)
            except:
                continue
            time.sleep(0.03)
        
        # Sort by score
        signals.sort(key=lambda x: x.score, reverse=True)
        self.signals = signals
        
        return signals
    
    def get_best_entries(self, signals: List[TradingSignal] = None) -> List[TradingSignal]:
        """Get signals that meet entry criteria"""
        sigs = signals or self.signals
        return [s for s in sigs if s.score >= self.CONFIG['min_score_entry']]
    
    def format_signal(self, signal: TradingSignal) -> str:
        """Format a single signal for display"""
        
        type_emoji = {
            SignalType.BREAKOUT: "🚀",
            SignalType.BOUNCE: "📉",
            SignalType.ACCUMULATION: "📊",
            SignalType.MOMENTUM: "📈",
        }
        
        strength_color = {
            SignalStrength.STRONG: "🔥",
            SignalStrength.MEDIUM: "⚡",
            SignalStrength.WEAK: "👀",
        }
        
        return f"""
  {type_emoji[signal.signal_type]} {signal.symbol:<14} | {signal.signal_type.value:<12} | Score: {signal.score:.0f}/100 {strength_color[signal.strength]}
  ├─ Entry: ${signal.entry_price:.6f}
  ├─ Stop Loss: ${signal.stop_loss:.6f} ({signal.risk_pct:.1f}% risk)
  ├─ TP1: ${signal.take_profit_1:.6f} | TP2: ${signal.take_profit_2:.6f} | TP3: ${signal.take_profit_3:.6f}
  ├─ R:R Ratio: {signal.rr_ratio:.1f}:1 | RSI: {signal.rsi:.0f} | Vol: {signal.volume_spike:.1f}x
  └─ Reason: {signal.entry_reason}
"""
    
    def display_signals(self, signals: List[TradingSignal] = None):
        """Display all signals in formatted output"""
        
        sigs = signals or self.signals
        
        if not sigs:
            print("\n  No signals found.")
            return
        
        # Group by type
        by_type = {}
        for s in sigs:
            if s.signal_type not in by_type:
                by_type[s.signal_type] = []
            by_type[s.signal_type].append(s)
        
        print("\n" + "=" * 100)
        print(f"  🚀 SUPERTRADER SIGNALS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 100)
        
        # Best entries first
        best = self.get_best_entries(sigs)
        if best:
            print("\n  🎯 READY TO TRADE (Score 70+):")
            print("-" * 100)
            for s in best[:10]:
                print(self.format_signal(s))
        
        # All signals by type
        for sig_type in [SignalType.BREAKOUT, SignalType.BOUNCE, SignalType.ACCUMULATION, SignalType.MOMENTUM]:
            type_sigs = by_type.get(sig_type, [])
            if type_sigs:
                print(f"\n  {sig_type.value} SIGNALS ({len(type_sigs)}):")
                print("-" * 100)
                for s in type_sigs[:5]:
                    if s not in best:
                        print(self.format_signal(s))
        
        # Summary
        print("\n" + "=" * 100)
        print(f"  📊 SUMMARY")
        print(f"  Total signals: {len(sigs)} | Ready to trade: {len(best)}")
        print(f"  Breakouts: {len(by_type.get(SignalType.BREAKOUT, []))} | "
              f"Bounces: {len(by_type.get(SignalType.BOUNCE, []))} | "
              f"Accumulation: {len(by_type.get(SignalType.ACCUMULATION, []))} | "
              f"Momentum: {len(by_type.get(SignalType.MOMENTUM, []))}")
        print("=" * 100)
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk"""
        
        portfolio = self.CONFIG['portfolio_size']
        max_risk = self.CONFIG['max_risk_per_trade']
        
        risk_amount = portfolio * max_risk
        position_size = risk_amount / (signal.risk_pct / 100)
        
        # Cap at 20% of portfolio
        max_position = portfolio * 0.20
        return min(position_size, max_position)
    
    def generate_trade_plan(self, signal: TradingSignal) -> str:
        """Generate complete trade plan"""
        
        position_size = self.calculate_position_size(signal)
        
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TRADE PLAN: {signal.symbol}
╠══════════════════════════════════════════════════════════════════════════════╣
║
║  SIGNAL: {signal.signal_type.value} | Strength: {signal.strength.value} | Score: {signal.score:.0f}/100
║  
║  ENTRY:
║  ├─ Price: ${signal.entry_price:.6f}
║  ├─ Position Size: ${position_size:.2f}
║  └─ Reason: {signal.entry_reason}
║
║  EXIT STRATEGY:
║  ├─ Stop Loss: ${signal.stop_loss:.6f} (-{signal.risk_pct:.1f}%)
║  ├─ TP1 (50%): ${signal.take_profit_1:.6f} (+{((signal.take_profit_1/signal.entry_price)-1)*100:.1f}%)
║  ├─ TP2 (30%): ${signal.take_profit_2:.6f} (+{((signal.take_profit_2/signal.entry_price)-1)*100:.1f}%)
║  └─ TP3 (20%): ${signal.take_profit_3:.6f} (+{((signal.take_profit_3/signal.entry_price)-1)*100:.1f}%)
║
║  RISK/REWARD:
║  ├─ Risk: ${position_size * signal.risk_pct / 100:.2f} ({signal.risk_pct:.1f}%)
║  ├─ Reward (TP2): ${position_size * ((signal.take_profit_2/signal.entry_price)-1):.2f}
║  └─ R:R Ratio: {signal.rr_ratio:.1f}:1
║
║  TECHNICAL:
║  ├─ RSI: {signal.rsi:.0f}
║  ├─ Volume: {signal.volume_spike:.1f}x average
║  └─ Trend Score: {signal.trend_score}/3
║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def run_continuous(self, interval_minutes: int = 15):
        """Run continuous scanning"""
        
        print("\n" + "=" * 80)
        print("  🚀 SUPERTRADER - Continuous Mode")
        print(f"  Scanning every {interval_minutes} minutes")
        print("=" * 80)
        
        cycle = 0
        
        while True:
            cycle += 1
            print(f"\n  Cycle {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            
            try:
                signals = self.hunt(max_coins=150)
                best = self.get_best_entries(signals)
                
                if best:
                    print(f"\n  🎯 {len(best)} TRADE-READY SIGNALS:")
                    for s in best[:5]:
                        print(f"    • {s.symbol} - {s.signal_type.value} - Score {s.score:.0f}")
                else:
                    print("  No signals meeting entry criteria.")
                
            except Exception as e:
                print(f"  Error: {e}")
            
            print(f"\n  Next scan in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)


def main():
    """Main entry point"""
    
    trader = SuperTrader()
    
    # Hunt for signals
    signals = trader.hunt(max_coins=200)
    
    # Display all signals
    trader.display_signals()
    
    # Generate trade plans for best signals
    best = trader.get_best_entries()
    
    if best:
        print("\n" + "=" * 80)
        print("  📋 TRADE PLANS FOR TOP SIGNALS")
        print("=" * 80)
        
        for signal in best[:3]:
            print(trader.generate_trade_plan(signal))
    
    return trader, signals


if __name__ == "__main__":
    main()
