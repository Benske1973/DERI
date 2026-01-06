"""
🚀 SUPERTRADER OPTIMIZED
========================
Geoptimaliseerd op basis van backtest resultaten.

VERBETERINGEN:
1. Strengere entry criteria (score >= 70)
2. Trailing stop loss
3. Momentum filter
4. Volume confirmation required
5. Trend filter
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
import time

from core.indicators import Indicators
from core.data import DataFetcher


@dataclass
class Trade:
    symbol: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    signal_type: str
    pnl_pct: float
    exit_reason: str
    bars_held: int


class SuperTraderOptimized:
    """Optimized SuperTrader with strict criteria"""
    
    # Optimized parameters
    PARAMS = {
        # Entry
        'min_score': 70,              # Even higher threshold
        'min_rsi_bounce': 30,         # Must be really oversold
        'max_rsi_breakout': 68,       # Not overbought
        'min_volume_spike': 1.8,      # Stronger volume confirmation
        'require_trend': True,        # Must have trend alignment
        
        # Exit
        'atr_stop': 1.2,              # Even tighter stop (reduce loss size)
        'atr_trail_activate': 0.5,    # Start trailing earlier (after 0.5 ATR profit)
        'atr_trail_distance': 0.6,    # Tighter trail
        'max_bars': 20,               # Shorter max hold (faster exit)
        
        # Risk
        'risk_per_trade': 0.02,
        'target_rr': 2.0,             # 2:1 R:R target
    }
    
    def __init__(self):
        self.fetcher = DataFetcher()
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate optimized signals"""
        
        # Indicators
        df['rsi'] = Indicators.rsi(df['close'], 14)
        df['rsi_prev'] = df['rsi'].shift(1)
        
        df['macd'], df['signal_line'], df['hist'] = Indicators.macd(df['close'])
        df['hist_prev'] = df['hist'].shift(1)
        df['hist_prev2'] = df['hist'].shift(2)
        df['macd_rising'] = df['hist'] > df['hist_prev']
        df['macd_accel'] = (df['hist'] > df['hist_prev']) & (df['hist_prev'] > df['hist_prev2'])
        df['macd_cross'] = (df['hist'] > 0) & (df['hist_prev'] <= 0)
        
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], 14)
        
        df['ema20'] = Indicators.ema(df['close'], 20)
        df['ema50'] = Indicators.ema(df['close'], 50)
        df['trend_up'] = (df['close'] > df['ema20']) & (df['ema20'] > df['ema50'])
        df['trend_down'] = (df['close'] < df['ema20']) & (df['ema20'] < df['ema50'])
        
        df['high_20'] = df['high'].rolling(20).max().shift(1)
        df['low_20'] = df['low'].rolling(20).min()
        
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / df['vol_ma']
        
        # Signals
        df['signal'] = 'NONE'
        df['score'] = 0
        
        for i in range(35, len(df)):
            rsi = df['rsi'].iloc[i]
            rsi_prev = df['rsi_prev'].iloc[i]
            hist = df['hist'].iloc[i]
            macd_rising = df['macd_rising'].iloc[i]
            macd_accel = df['macd_accel'].iloc[i]
            macd_cross = df['macd_cross'].iloc[i]
            price = df['close'].iloc[i]
            high_20 = df['high_20'].iloc[i]
            vol_spike = df['vol_spike'].iloc[i]
            trend_up = df['trend_up'].iloc[i]
            
            if pd.isna(rsi) or pd.isna(high_20) or pd.isna(vol_spike):
                continue
            
            signal = 'NONE'
            score = 0
            
            # BOUNCE - Stricter criteria
            if rsi < self.PARAMS['min_rsi_bounce']:
                # Must have volume and MACD turning
                if vol_spike >= self.PARAMS['min_volume_spike'] and macd_rising:
                    signal = 'BOUNCE'
                    score = 45
                    
                    if rsi < 25:
                        score += 25
                    if rsi > rsi_prev:  # Recovery
                        score += 20
                    if macd_cross:
                        score += 20
                    if vol_spike > 2.5:
                        score += 15
            
            # BREAKOUT - Must have strong confirmation
            elif price > high_20 and macd_rising and vol_spike >= self.PARAMS['min_volume_spike']:
                if rsi < self.PARAMS['max_rsi_breakout'] and trend_up:  # Not overbought AND trend up
                    signal = 'BREAKOUT'
                    score = 45
                    
                    if macd_cross:
                        score += 30
                    elif macd_accel:
                        score += 20
                    elif hist > 0:
                        score += 10
                    
                    if vol_spike > 2.5:
                        score += 20
                    elif vol_spike > 2:
                        score += 10
                    
                    # Extra: strong candle
                    candle_body = abs(price - df['open'].iloc[i])
                    candle_range = df['high'].iloc[i] - df['low'].iloc[i]
                    if candle_range > 0 and candle_body / candle_range > 0.7:
                        score += 10
            
            if score >= self.PARAMS['min_score']:
                df.loc[df.index[i], 'signal'] = signal
                df.loc[df.index[i], 'score'] = score
        
        return df
    
    def backtest_symbol(self, symbol: str, days: int = 365) -> List[Trade]:
        """Backtest with trailing stop"""
        
        df = self.fetcher.fetch(symbol, '4h', days)
        if df is None or len(df) < 100:
            return []
        
        df = self.generate_signals(df)
        
        trades = []
        in_trade = False
        entry_price = 0
        entry_time = None
        entry_idx = 0
        entry_signal = ''
        entry_atr = 0
        stop_loss = 0
        highest_since_entry = 0
        trailing_activated = False
        
        for i in range(35, len(df)):
            row = df.iloc[i]
            price = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            
            if pd.isna(atr) or atr == 0:
                continue
            
            if in_trade:
                bars_held = i - entry_idx
                
                # Update highest
                if high > highest_since_entry:
                    highest_since_entry = high
                
                # Trailing stop activation
                if not trailing_activated:
                    profit_atr = (highest_since_entry - entry_price) / atr
                    if profit_atr >= self.PARAMS['atr_trail_activate']:
                        trailing_activated = True
                        stop_loss = highest_since_entry - (self.PARAMS['atr_trail_distance'] * atr)
                
                # Update trailing stop
                if trailing_activated:
                    new_stop = highest_since_entry - (self.PARAMS['atr_trail_distance'] * atr)
                    stop_loss = max(stop_loss, new_stop)
                
                # Take profit level (2.5:1 R:R)
                risk = entry_price - (entry_price - (self.PARAMS['atr_stop'] * entry_atr))
                take_profit = entry_price + (risk * 2.5)
                
                # Check exit conditions
                exit_reason = None
                exit_price = None
                
                # Take profit hit
                if high >= take_profit:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = take_profit
                
                # Stop loss
                elif low <= stop_loss:
                    exit_reason = 'STOP_LOSS' if not trailing_activated else 'TRAILING_STOP'
                    exit_price = stop_loss
                
                # Time stop
                elif bars_held >= self.PARAMS['max_bars']:
                    exit_reason = 'TIME_STOP'
                    exit_price = price
                
                if exit_reason:
                    pnl = ((exit_price - entry_price) / entry_price) * 100
                    trades.append(Trade(
                        symbol=symbol,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=str(row['timestamp']),
                        exit_price=exit_price,
                        signal_type=entry_signal,
                        pnl_pct=pnl,
                        exit_reason=exit_reason,
                        bars_held=bars_held,
                    ))
                    in_trade = False
            
            # Entry
            if not in_trade and row['signal'] != 'NONE':
                in_trade = True
                entry_price = price
                entry_time = str(row['timestamp'])
                entry_idx = i
                entry_signal = row['signal']
                entry_atr = atr
                stop_loss = price - (self.PARAMS['atr_stop'] * atr)
                highest_since_entry = price
                trailing_activated = False
        
        # Close open trade
        if in_trade:
            final_price = df['close'].iloc[-1]
            pnl = ((final_price - entry_price) / entry_price) * 100
            trades.append(Trade(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=str(df['timestamp'].iloc[-1]),
                exit_price=final_price,
                signal_type=entry_signal,
                pnl_pct=pnl,
                exit_reason='END_OF_DATA',
                bars_held=len(df) - entry_idx,
            ))
        
        return trades
    
    def run_backtest(self, symbols: List[str], days: int = 365) -> dict:
        """Run full backtest"""
        
        print("\n" + "=" * 80)
        print("  🧪 SUPERTRADER OPTIMIZED BACKTEST")
        print("=" * 80)
        print(f"  Symbols: {len(symbols)} | Period: {days} days")
        print(f"  Min Score: {self.PARAMS['min_score']} | ATR Stop: {self.PARAMS['atr_stop']}")
        print()
        
        all_trades = []
        
        for i, symbol in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] {symbol}...", end=" ")
            trades = self.backtest_symbol(symbol, days)
            all_trades.extend(trades)
            
            if trades:
                wins = len([t for t in trades if t.pnl_pct > 0])
                print(f"{len(trades)} trades, {wins} wins")
            else:
                print("No trades")
        
        if not all_trades:
            print("\n  No trades found.")
            return {}
        
        # Statistics
        wins = [t for t in all_trades if t.pnl_pct > 0]
        losses = [t for t in all_trades if t.pnl_pct <= 0]
        
        total_pnl = sum(t.pnl_pct for t in all_trades)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        
        win_rate = len(wins) / len(all_trades) * 100
        
        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # By exit reason
        by_exit = {}
        for t in all_trades:
            if t.exit_reason not in by_exit:
                by_exit[t.exit_reason] = []
            by_exit[t.exit_reason].append(t)
        
        # By signal type
        by_type = {}
        for t in all_trades:
            if t.signal_type not in by_type:
                by_type[t.signal_type] = []
            by_type[t.signal_type].append(t)
        
        return {
            'total_trades': len(all_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_bars': np.mean([t.bars_held for t in all_trades]),
            'by_exit': {
                k: {
                    'count': len(v),
                    'win_rate': len([t for t in v if t.pnl_pct > 0]) / len(v) * 100 if v else 0,
                    'pnl': sum(t.pnl_pct for t in v),
                } for k, v in by_exit.items()
            },
            'by_type': {
                k: {
                    'trades': len(v),
                    'win_rate': len([t for t in v if t.pnl_pct > 0]) / len(v) * 100 if v else 0,
                    'pnl': sum(t.pnl_pct for t in v),
                } for k, v in by_type.items()
            }
        }
    
    def print_results(self, results: dict):
        """Print results"""
        
        if not results:
            return
        
        win_ok = "✓" if results['win_rate'] >= 50 else "✗"
        pf_ok = "✓" if results['profit_factor'] >= 1.5 else "✗"
        
        targets_met = results['win_rate'] >= 50 and results['profit_factor'] >= 1.5
        
        print("\n" + "=" * 80)
        print("  📊 OPTIMIZED BACKTEST RESULTS")
        print("=" * 80)
        print(f"""
  Total Trades:    {results['total_trades']}
  Wins/Losses:     {results['wins']} / {results['losses']}
  Avg Bars Held:   {results['avg_bars']:.1f}
  
  Win Rate:        {results['win_rate']:.1f}% {win_ok} (target: >=50%)
  Profit Factor:   {results['profit_factor']:.2f} {pf_ok} (target: >=1.5)
  
  Total P&L:       {results['total_pnl']:+.1f}%
  Avg Win:         {results['avg_win']:+.1f}%
  Avg Loss:        {results['avg_loss']:.1f}%
""")
        
        print("  BY SIGNAL TYPE:")
        for sig_type, stats in results['by_type'].items():
            print(f"    {sig_type}: {stats['trades']} trades | WR: {stats['win_rate']:.1f}% | P&L: {stats['pnl']:+.1f}%")
        
        print("\n  BY EXIT REASON:")
        for reason, stats in results['by_exit'].items():
            print(f"    {reason}: {stats['count']} trades | WR: {stats['win_rate']:.1f}% | P&L: {stats['pnl']:+.1f}%")
        
        print("\n" + "=" * 80)
        if targets_met:
            print("  ✅ ALL TARGETS MET!")
        else:
            print("  ❌ Needs more optimization")
        print("=" * 80)


def main():
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
        'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT',
        'LINK/USDT', 'AVAX/USDT', 'DOT/USDT',
        'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT',
        'INJ/USDT', 'RENDER/USDT', 'PEPE/USDT', 'WLD/USDT',
    ]
    
    bt = SuperTraderOptimized()
    results = bt.run_backtest(symbols, days=180)
    bt.print_results(results)
    
    return bt, results


if __name__ == "__main__":
    main()
