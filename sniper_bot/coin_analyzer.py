"""
COIN ANALYZER
=============
Test alle coins met volume > 50k en vind:
1. Welke coins het beste presteren
2. Wat de kenmerken zijn van goede breakout coins
3. Optimale coin selectie criteria
"""
import sys
sys.path.insert(0, '/workspace/sniper_bot')

import pandas as pd
import numpy as np
from core.data import DataFetcher
from strategies.swing_trader import SwingTrader
from core.indicators import Indicators
import time


class CoinAnalyzer:
    """Analyze all coins to find best performers"""
    
    def __init__(self):
        self.fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
        self.trader = SwingTrader()
    
    def get_all_coins(self, min_volume: float = 50000) -> list:
        """Get all USDT pairs with minimum volume"""
        try:
            tickers = self.fetcher.exchange.fetch_tickers()
            
            coins = []
            for symbol, ticker in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if '/USDT:' in symbol:  # Skip futures
                    continue
                
                vol = ticker.get('quoteVolume', 0) or 0
                price = ticker.get('last', 0) or 0
                change = ticker.get('percentage', 0) or 0
                
                if vol >= min_volume and price > 0:
                    coins.append({
                        'symbol': symbol,
                        'price': price,
                        'volume_24h': vol,
                        'change_24h': change,
                    })
            
            coins.sort(key=lambda x: x['volume_24h'], reverse=True)
            return coins
            
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def analyze_coin(self, symbol: str, days: int = 180) -> dict:
        """Full analysis of a single coin"""
        
        df = self.fetcher.fetch(symbol, '4h', days=days)
        
        if df.empty or len(df) < 100:
            return {'symbol': symbol, 'valid': False, 'error': 'Insufficient data'}
        
        # Calculate coin characteristics
        df = self.trader.calculate_indicators(df)
        
        # Run backtest
        trades = self.trader.backtest(df, symbol)
        
        if not trades:
            return {
                'symbol': symbol, 
                'valid': True,
                'trades': 0,
                'characteristics': self._calc_characteristics(df),
            }
        
        # Calculate stats
        stats = self.trader.analyze_trades(trades)
        
        return {
            'symbol': symbol,
            'valid': True,
            'trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'profit_factor': stats['profit_factor'],
            'total_pnl': stats['total_pnl'],
            'avg_win': stats['avg_win'],
            'avg_loss': stats['avg_loss'],
            'best_trade': stats['best_trade'],
            'worst_trade': stats['worst_trade'],
            'characteristics': self._calc_characteristics(df),
        }
    
    def _calc_characteristics(self, df: pd.DataFrame) -> dict:
        """Calculate coin characteristics for analysis"""
        
        # Volatility
        avg_atr_pct = df['atr_pct'].mean()
        
        # Trend strength
        uptrend_pct = df['uptrend'].sum() / len(df) * 100
        downtrend_pct = df['downtrend'].sum() / len(df) * 100
        
        # Volume consistency
        vol_std = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 0
        
        # Price movement
        total_return = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
        
        # Breakout frequency (how often does it breakout)
        breakouts = (df['hist'].abs() > df['breakout_threshold']).sum()
        breakout_freq = breakouts / len(df) * 100
        
        # RSI characteristics
        avg_rsi = df['rsi'].mean()
        rsi_volatility = df['rsi'].std()
        
        return {
            'avg_atr_pct': avg_atr_pct,
            'uptrend_pct': uptrend_pct,
            'downtrend_pct': downtrend_pct,
            'vol_consistency': 1 / (1 + vol_std),  # Higher = more consistent
            'total_return': total_return,
            'breakout_freq': breakout_freq,
            'avg_rsi': avg_rsi,
            'rsi_volatility': rsi_volatility,
        }
    
    def run_full_analysis(self, min_volume: float = 50000, max_coins: int = 200) -> pd.DataFrame:
        """Run analysis on all coins"""
        
        print("=" * 80)
        print("COIN ANALYZER - Finding Best Breakout Coins")
        print("=" * 80)
        
        # Get all coins
        print(f"\nFetching coins with volume > ${min_volume:,.0f}...")
        coins = self.get_all_coins(min_volume)
        print(f"Found {len(coins)} coins")
        
        # Limit for speed
        coins = coins[:max_coins]
        print(f"Analyzing top {len(coins)} by volume...")
        
        results = []
        
        for i, coin in enumerate(coins):
            symbol = coin['symbol']
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(coins)}")
            
            try:
                result = self.analyze_coin(symbol, days=180)
                result['volume_24h'] = coin['volume_24h']
                result['change_24h'] = coin['change_24h']
                results.append(result)
                
                time.sleep(0.05)  # Rate limit
                
            except Exception as e:
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def find_best_coins(self, results_df: pd.DataFrame, 
                        min_trades: int = 5,
                        min_win_rate: float = 50,
                        min_pf: float = 1.0) -> pd.DataFrame:
        """Filter and rank best performing coins"""
        
        # Filter valid results with enough trades
        df = results_df[
            (results_df['valid'] == True) & 
            (results_df['trades'] >= min_trades)
        ].copy()
        
        if df.empty:
            return df
        
        # Filter by performance
        df = df[
            (df['win_rate'] >= min_win_rate) &
            (df['profit_factor'] >= min_pf)
        ]
        
        # Sort by profit factor
        df = df.sort_values('profit_factor', ascending=False)
        
        return df
    
    def analyze_patterns(self, results_df: pd.DataFrame) -> dict:
        """Analyze what makes a coin good for breakout trading"""
        
        # Split into good and bad performers
        valid = results_df[results_df['trades'] >= 5].copy()
        
        if valid.empty:
            return {}
        
        good = valid[valid['profit_factor'] >= 1.5]
        bad = valid[valid['profit_factor'] < 1.0]
        
        if good.empty or bad.empty:
            return {}
        
        # Extract characteristics
        def extract_chars(df):
            chars = pd.DataFrame([r['characteristics'] for r in df.to_dict('records') if 'characteristics' in r and r['characteristics']])
            return chars.mean() if not chars.empty else pd.Series()
        
        good_chars = extract_chars(good)
        bad_chars = extract_chars(bad)
        
        return {
            'good_performers': good_chars.to_dict() if not good_chars.empty else {},
            'bad_performers': bad_chars.to_dict() if not bad_chars.empty else {},
            'num_good': len(good),
            'num_bad': len(bad),
        }


def main():
    """Run full analysis"""
    
    analyzer = CoinAnalyzer()
    
    # Run analysis on all coins
    results = analyzer.run_full_analysis(min_volume=50000, max_coins=150)
    
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    
    # Get valid results with trades
    valid = results[(results['valid'] == True) & (results['trades'] > 0)]
    print(f"\nCoins with trades: {len(valid)}")
    
    # Find best performers
    print("\n" + "-" * 80)
    print("TOP PERFORMING COINS (WR >= 50%, PF >= 1.5)")
    print("-" * 80)
    
    best = analyzer.find_best_coins(results, min_trades=5, min_win_rate=50, min_pf=1.5)
    
    if not best.empty:
        print(f"\n{'SYMBOL':<14} {'TRADES':<8} {'WR%':<8} {'PF':<8} {'P&L%':<10} {'VOL_24H':<12}")
        print("-" * 80)
        
        for _, row in best.head(30).iterrows():
            print(f"{row['symbol']:<14} {row['trades']:<8} {row['win_rate']:<8.1f} "
                  f"{row['profit_factor']:<8.2f} {row['total_pnl']:<+10.1f} "
                  f"${row['volume_24h']/1e6:<.1f}M")
        
        print("-" * 80)
        print(f"Total: {len(best)} coins meeting criteria")
        
        # Combined stats for best coins
        if len(best) >= 3:
            print(f"\nCombined stats for top {min(10, len(best))} coins:")
            print(f"  Avg Win Rate: {best.head(10)['win_rate'].mean():.1f}%")
            print(f"  Avg PF: {best.head(10)['profit_factor'].mean():.2f}")
            print(f"  Avg P&L: {best.head(10)['total_pnl'].mean():.1f}%")
    else:
        print("No coins meet the criteria")
    
    # Show worst performers too
    print("\n" + "-" * 80)
    print("WORST PERFORMERS (to avoid)")
    print("-" * 80)
    
    worst = valid[valid['profit_factor'] < 0.5].sort_values('profit_factor')
    if not worst.empty:
        for _, row in worst.head(10).iterrows():
            print(f"{row['symbol']:<14} Trades:{row['trades']:<4} WR:{row['win_rate']:<5.1f}% "
                  f"PF:{row['profit_factor']:<5.2f} P&L:{row['total_pnl']:<+.1f}%")
    
    # Pattern analysis
    print("\n" + "-" * 80)
    print("PATTERN ANALYSIS - What makes a good breakout coin?")
    print("-" * 80)
    
    patterns = analyzer.analyze_patterns(results)
    
    if patterns and patterns.get('good_performers') and patterns.get('bad_performers'):
        good = patterns['good_performers']
        bad = patterns['bad_performers']
        
        print(f"\nComparing {patterns['num_good']} good vs {patterns['num_bad']} bad performers:")
        print(f"\n{'CHARACTERISTIC':<20} {'GOOD':<12} {'BAD':<12} {'INSIGHT'}")
        print("-" * 70)
        
        insights = [
            ('avg_atr_pct', 'ATR %', 'Higher volatility = better'),
            ('breakout_freq', 'Breakout Freq', 'More breakouts = better'),
            ('uptrend_pct', 'Uptrend %', 'More uptrend = better'),
            ('vol_consistency', 'Vol Consistency', 'More consistent = better'),
            ('total_return', 'Total Return %', 'Trending coins = better'),
        ]
        
        for key, name, insight in insights:
            g = good.get(key, 0)
            b = bad.get(key, 0)
            diff = "↑" if g > b else "↓"
            print(f"{name:<20} {g:<12.2f} {b:<12.2f} {diff} {insight}")
    
    # Save top coins list
    if not best.empty:
        top_symbols = best['symbol'].head(20).tolist()
        print(f"\n" + "=" * 80)
        print("RECOMMENDED COIN LIST (copy to swing_trader.py):")
        print("=" * 80)
        print(f"\nTOP_COINS = {top_symbols}")
    
    return results, best


if __name__ == "__main__":
    results, best = main()
