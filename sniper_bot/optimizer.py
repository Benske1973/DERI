"""
Strategy Optimizer - Test parameters and validate across multiple coins
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Any
import time

from core.data import DataFetcher
from core.backtest import Backtester, BacktestResult
from strategies.momentum_breakout import MomentumBreakout


class Optimizer:
    """Optimize strategy parameters"""
    
    def __init__(self):
        self.fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
        self.backtester = Backtester()
    
    def test_params(self, symbols: List[str], params: Dict, 
                    timeframe: str = '4h', days: int = 365) -> Dict:
        """Test parameters across multiple symbols"""
        
        all_trades = []
        symbol_results = []
        
        for symbol in symbols:
            df = self.fetcher.fetch(symbol, timeframe, days)
            if df.empty or len(df) < 100:
                continue
            
            strategy = MomentumBreakout(params)
            signals = strategy.generate_signals(df)
            result = self.backtester.run(df, signals)
            
            all_trades.extend(result.trades)
            symbol_results.append({
                'symbol': symbol,
                'trades': result.total_trades,
                'win_rate': result.win_rate,
                'pf': result.profit_factor,
                'pnl': result.total_pnl,
                'max_dd': result.max_drawdown,
            })
        
        # Aggregate results
        if not all_trades:
            return {'valid': False}
        
        combined = BacktestResult(trades=all_trades)
        
        return {
            'valid': True,
            'total_trades': combined.total_trades,
            'win_rate': combined.win_rate,
            'profit_factor': combined.profit_factor,
            'total_pnl': combined.total_pnl,
            'max_drawdown': combined.max_drawdown,
            'expectancy': combined.expectancy,
            'meets_targets': combined.meets_targets,
            'symbol_results': symbol_results,
            'params': params,
        }
    
    def grid_search(self, symbols: List[str], param_grid: Dict[str, List],
                    timeframe: str = '4h', days: int = 365) -> List[Dict]:
        """Grid search over parameter space"""
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        print(f"Testing {len(combinations)} parameter combinations...")
        
        results = []
        best_pf = 0
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            result = self.test_params(symbols, params, timeframe, days)
            
            if result['valid']:
                results.append(result)
                
                # Track best
                if result['profit_factor'] > best_pf and result['win_rate'] >= 50:
                    best_pf = result['profit_factor']
                    print(f"  [{i+1}/{len(combinations)}] New best: WR={result['win_rate']:.1f}%, "
                          f"PF={result['profit_factor']:.2f}, Trades={result['total_trades']}")
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(combinations)}")
        
        # Sort by profit factor (with minimum win rate)
        valid_results = [r for r in results if r['win_rate'] >= 50]
        valid_results.sort(key=lambda x: x['profit_factor'], reverse=True)
        
        return valid_results


def main():
    """Main optimization loop"""
    
    print("=" * 70)
    print("STRATEGY OPTIMIZATION")
    print("=" * 70)
    print("Targets: WinRate >= 50%, PF >= 1.5, MaxDD <= 20%")
    print("=" * 70)
    
    optimizer = Optimizer()
    
    # Test coins
    test_symbols = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT',
        'AVAX/USDT', 'LINK/USDT', 'DOT/USDT', 'ATOM/USDT',
    ]
    
    print(f"\nTesting on {len(test_symbols)} symbols...")
    
    # Phase 1: Test default parameters
    print("\n" + "-" * 70)
    print("PHASE 1: Default Parameters")
    print("-" * 70)
    
    default_result = optimizer.test_params(test_symbols, MomentumBreakout.DEFAULT_PARAMS)
    
    if default_result['valid']:
        print(f"\nDefault Results:")
        print(f"  Trades:        {default_result['total_trades']}")
        print(f"  Win Rate:      {default_result['win_rate']:.1f}%")
        print(f"  Profit Factor: {default_result['profit_factor']:.2f}")
        print(f"  Total P&L:     {default_result['total_pnl']:.1f}%")
        print(f"  Max Drawdown:  {default_result['max_drawdown']:.1f}%")
        print(f"  Meets Targets: {default_result['meets_targets']}")
        
        if default_result['meets_targets']:
            print("\n✓ DEFAULT PARAMETERS MEET TARGETS!")
            return default_result
    
    # Phase 2: Grid search for better parameters
    print("\n" + "-" * 70)
    print("PHASE 2: Grid Search Optimization")
    print("-" * 70)
    
    param_grid = {
        'breakout_mult': [1.5, 2.0, 2.5, 3.0],
        'rsi_buy_min': [30, 40, 45],
        'rsi_buy_max': [65, 70, 75],
        'sl_atr_mult': [1.0, 1.5, 2.0],
        'tp_atr_mult': [2.0, 2.5, 3.0, 4.0],
        'use_trend_filter': [True, False],
    }
    
    results = optimizer.grid_search(test_symbols, param_grid)
    
    if results:
        best = results[0]
        print(f"\n{'=' * 70}")
        print("BEST PARAMETERS FOUND:")
        print(f"{'=' * 70}")
        print(f"  Trades:        {best['total_trades']}")
        print(f"  Win Rate:      {best['win_rate']:.1f}%")
        print(f"  Profit Factor: {best['profit_factor']:.2f}")
        print(f"  Total P&L:     {best['total_pnl']:.1f}%")
        print(f"  Max Drawdown:  {best['max_drawdown']:.1f}%")
        print(f"  Expectancy:    {best['expectancy']:.2f}%")
        print(f"\n  Parameters:")
        for k, v in best['params'].items():
            print(f"    {k}: {v}")
        
        print(f"\n  Per Symbol:")
        for sr in best['symbol_results']:
            status = "✓" if sr['win_rate'] >= 50 and sr['pf'] >= 1.0 else "✗"
            print(f"    {sr['symbol']:<12} Trades:{sr['trades']:3} WR:{sr['win_rate']:5.1f}% "
                  f"PF:{sr['pf']:5.2f} P&L:{sr['pnl']:+6.1f}% {status}")
        
        print(f"\n  MEETS TARGETS: {best['meets_targets']}")
        
        return best
    else:
        print("\nNo valid results found. Need to expand search space.")
        return None


if __name__ == "__main__":
    result = main()
