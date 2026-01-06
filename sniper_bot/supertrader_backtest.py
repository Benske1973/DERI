"""
🧪 SUPERTRADER BACKTEST
=======================
Test de SuperTrader strategie op historische data.
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple
import os

from core.indicators import Indicators
from core.data import DataFetcher


@dataclass
class BacktestTrade:
    symbol: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    signal_type: str
    pnl_pct: float
    exit_reason: str


class SuperTraderBacktest:
    """Backtest the SuperTrader strategy"""
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.trades: List[BacktestTrade] = []
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for entire dataframe"""
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], 14)
        
        # MACD
        df['macd'], df['signal_line'], df['hist'] = Indicators.macd(df['close'])
        df['hist_prev'] = df['hist'].shift(1)
        df['macd_rising'] = df['hist'] > df['hist_prev']
        df['macd_cross'] = (df['hist'] > 0) & (df['hist_prev'] <= 0)
        
        # ATR
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], 14)
        
        # EMAs
        df['ema20'] = Indicators.ema(df['close'], 20)
        df['ema50'] = Indicators.ema(df['close'], 50)
        df['above_ema20'] = df['close'] > df['ema20']
        df['above_ema50'] = df['close'] > df['ema50']
        df['ema_aligned'] = df['ema20'] > df['ema50']
        df['trend_score'] = df['above_ema20'].astype(int) + df['above_ema50'].astype(int) + df['ema_aligned'].astype(int)
        
        # Rolling highs/lows
        df['high_20'] = df['high'].rolling(20).max().shift(1)
        df['low_20'] = df['low'].rolling(20).min()
        
        # Volume
        df['vol_avg'] = df['volume'].rolling(20).mean()
        df['vol_spike'] = df['volume'] / df['vol_avg']
        
        # Signals
        df['signal'] = 'NONE'
        df['score'] = 0
        
        for i in range(30, len(df)):
            rsi = df['rsi'].iloc[i]
            hist = df['hist'].iloc[i]
            hist_prev = df['hist_prev'].iloc[i]
            macd_rising = df['macd_rising'].iloc[i]
            macd_cross = df['macd_cross'].iloc[i]
            price = df['close'].iloc[i]
            high_20 = df['high_20'].iloc[i]
            vol_spike = df['vol_spike'].iloc[i]
            trend_score = df['trend_score'].iloc[i]
            
            # Skip if NaN
            if pd.isna(rsi) or pd.isna(high_20):
                continue
            
            signal = 'NONE'
            score = 0
            
            # BOUNCE
            if rsi < 40:
                signal = 'BOUNCE'
                if rsi < 25:
                    score = 40
                elif rsi < 30:
                    score = 30
                else:
                    score = 20
                
                if i > 0 and df['rsi'].iloc[i] > df['rsi'].iloc[i-1]:
                    score += 15
                if macd_rising:
                    score += 15
                if vol_spike > 2:
                    score += 10
            
            # BREAKOUT
            elif price > high_20 or macd_cross or (((high_20 - price) / price) * 100 < 5 and macd_rising):
                signal = 'BREAKOUT'
                
                if price > high_20:
                    score = 30
                else:
                    score = 20
                
                if macd_cross:
                    score += 25
                elif hist > 0 and macd_rising:
                    score += 15
                
                if vol_spike > 2:
                    score += 20
                elif vol_spike > 1.5:
                    score += 10
                
                if trend_score >= 2:
                    score += 10
            
            # Only take signals with score >= 50
            if score >= 50:
                df.loc[df.index[i], 'signal'] = signal
                df.loc[df.index[i], 'score'] = score
        
        return df
    
    def backtest_symbol(self, symbol: str, days: int = 365) -> List[BacktestTrade]:
        """Backtest a single symbol"""
        
        df = self.fetcher.fetch(symbol, '4h', days)
        if df is None or len(df) < 100:
            return []
        
        df = self.generate_signals(df)
        
        trades = []
        in_trade = False
        entry_price = 0
        entry_time = None
        entry_signal = ''
        stop_loss = 0
        take_profit = 0
        
        for i in range(30, len(df)):
            row = df.iloc[i]
            price = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            
            if pd.isna(atr):
                continue
            
            # Check for exit
            if in_trade:
                # Stop loss hit
                if low <= stop_loss:
                    pnl = ((stop_loss - entry_price) / entry_price) * 100
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=str(row['timestamp']),
                        exit_price=stop_loss,
                        signal_type=entry_signal,
                        pnl_pct=pnl,
                        exit_reason='STOP_LOSS'
                    ))
                    in_trade = False
                
                # Take profit hit
                elif high >= take_profit:
                    pnl = ((take_profit - entry_price) / entry_price) * 100
                    trades.append(BacktestTrade(
                        symbol=symbol,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=str(row['timestamp']),
                        exit_price=take_profit,
                        signal_type=entry_signal,
                        pnl_pct=pnl,
                        exit_reason='TAKE_PROFIT'
                    ))
                    in_trade = False
            
            # Check for entry
            if not in_trade and row['signal'] != 'NONE' and row['score'] >= 50:
                in_trade = True
                entry_price = price
                entry_time = str(row['timestamp'])
                entry_signal = row['signal']
                
                # Stop loss and take profit
                if entry_signal == 'BOUNCE':
                    stop_loss = row['low_20'] * 0.98
                else:
                    stop_loss = price - (2 * atr)
                
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 3)  # 3:1 R:R
        
        # Close any open trade at end
        if in_trade:
            final_price = df['close'].iloc[-1]
            pnl = ((final_price - entry_price) / entry_price) * 100
            trades.append(BacktestTrade(
                symbol=symbol,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=str(df['timestamp'].iloc[-1]),
                exit_price=final_price,
                signal_type=entry_signal,
                pnl_pct=pnl,
                exit_reason='END_OF_DATA'
            ))
        
        return trades
    
    def run_backtest(self, symbols: List[str], days: int = 365) -> dict:
        """Run full backtest"""
        
        print("\n" + "=" * 80)
        print("  🧪 SUPERTRADER BACKTEST")
        print("=" * 80)
        print(f"  Symbols: {len(symbols)} | Period: {days} days")
        print()
        
        all_trades = []
        
        for i, symbol in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] Backtesting {symbol}...")
            trades = self.backtest_symbol(symbol, days)
            all_trades.extend(trades)
            print(f"    Found {len(trades)} trades")
        
        self.trades = all_trades
        
        if not all_trades:
            print("\n  No trades found.")
            return {}
        
        # Calculate statistics
        wins = [t for t in all_trades if t.pnl_pct > 0]
        losses = [t for t in all_trades if t.pnl_pct <= 0]
        
        total_pnl = sum(t.pnl_pct for t in all_trades)
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        
        win_rate = len(wins) / len(all_trades) * 100
        
        gross_profit = sum(t.pnl_pct for t in wins)
        gross_loss = abs(sum(t.pnl_pct for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # By signal type
        by_type = {}
        for t in all_trades:
            if t.signal_type not in by_type:
                by_type[t.signal_type] = []
            by_type[t.signal_type].append(t)
        
        results = {
            'total_trades': len(all_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'by_type': {
                k: {
                    'trades': len(v),
                    'win_rate': len([t for t in v if t.pnl_pct > 0]) / len(v) * 100 if v else 0,
                    'total_pnl': sum(t.pnl_pct for t in v),
                } for k, v in by_type.items()
            }
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print backtest results"""
        
        if not results:
            return
        
        win_ok = "✓" if results['win_rate'] >= 50 else "✗"
        pf_ok = "✓" if results['profit_factor'] >= 1.5 else "✗"
        
        print("\n" + "=" * 80)
        print("  📊 BACKTEST RESULTS")
        print("=" * 80)
        print(f"""
  Total Trades:   {results['total_trades']}
  Wins/Losses:    {results['wins']} / {results['losses']}
  
  Win Rate:       {results['win_rate']:.1f}% {win_ok} (target: >=50%)
  Profit Factor:  {results['profit_factor']:.2f} {pf_ok} (target: >=1.5)
  
  Total P&L:      {results['total_pnl']:+.1f}%
  Avg Win:        {results['avg_win']:+.1f}%
  Avg Loss:       {results['avg_loss']:.1f}%
  
  BY SIGNAL TYPE:
""")
        
        for sig_type, stats in results['by_type'].items():
            print(f"  {sig_type}:")
            print(f"    Trades: {stats['trades']} | Win Rate: {stats['win_rate']:.1f}% | P&L: {stats['total_pnl']:+.1f}%")
        
        print()
        print("=" * 80)
        
        targets_met = results['win_rate'] >= 50 and results['profit_factor'] >= 1.5
        if targets_met:
            print("  ✅ ALL TARGETS MET - Strategy is validated!")
        else:
            print("  ❌ Targets not met - needs optimization")
        print("=" * 80)


def main():
    """Run backtest"""
    
    # Test symbols (mix of different types)
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT',
        'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'SHIB/USDT',
        'LINK/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT',
        'NEAR/USDT', 'APT/USDT', 'ARB/USDT', 'OP/USDT',
        'INJ/USDT', 'RENDER/USDT', 'PEPE/USDT', 'WLD/USDT',
    ]
    
    bt = SuperTraderBacktest()
    results = bt.run_backtest(symbols, days=180)
    bt.print_results(results)
    
    return bt, results


if __name__ == "__main__":
    main()
