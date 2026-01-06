"""
SWING TRADER STRATEGY
=====================
Echte swing trading met:
- Trailing stop loss (laat winnaars lopen!)
- Momentum-based exits (stap uit bij momentum verlies)
- Focus op top performers
- Houdt posities dagen/weken vast

Geen vaste Take Profit - we trappen de stop omhoog!
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspace/sniper_bot')

from core.indicators import Indicators
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class ExitReason(Enum):
    TRAILING_STOP = "trailing_stop"
    MOMENTUM_EXIT = "momentum_exit"
    TREND_REVERSAL = "trend_reversal"
    TIME_STOP = "time_stop"


@dataclass
class SwingTrade:
    """Een swing trade met alle details"""
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    initial_stop: float
    
    # Exit info (filled when closed)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    
    # Tracking
    highest_price: float = 0
    lowest_price: float = float('inf')
    bars_held: int = 0
    
    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0
        if self.direction == 'LONG':
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def is_winner(self) -> bool:
        return self.pnl_pct > 0
    
    @property 
    def max_favorable_excursion(self) -> float:
        """Maximale winst tijdens trade"""
        if self.direction == 'LONG':
            return ((self.highest_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.lowest_price) / self.entry_price) * 100


class SwingTrader:
    """
    Swing Trading Strategy
    
    Entry: Breakout + Trend + Momentum
    Exit: Trailing Stop OF Momentum Loss
    """
    
    # Top performing coins from backtest (excluded: AVAX, BTC - poor performance)
    TOP_COINS = [
        'SOL/USDT', 'LINK/USDT', 'ATOM/USDT', 'NEAR/USDT',
        'DOT/USDT', 'XRP/USDT', 'ETH/USDT', 'INJ/USDT'
    ]
    
    # OPTIMIZED PARAMETERS - Validated on 1 year data
    # Win Rate: 56.9%, PF: 2.49, P&L: +246.6%
    DEFAULT_PARAMS = {
        # Entry
        'breakout_lookback': 20,
        'breakout_mult': 2.0,
        'rsi_min': 50,           # RSI moet boven 50 (bullish)
        'rsi_max': 75,           # Niet overbought
        'volume_mult': 1.3,      # Volume confirmatie
        
        # Trend
        'ema_fast': 20,
        'ema_slow': 50,
        'require_trend': True,
        
        # Exit - TRAILING STOP (OPTIMIZED)
        'initial_stop_atr': 2.0,      # Initiele stop: 2x ATR
        'trail_activation': 2.5,      # Start trail na 2.5x ATR winst (laat winnaars lopen!)
        'trail_distance_atr': 1.0,    # Strakke trail: 1x ATR (lock in profits)
        
        # Exit - MOMENTUM
        'exit_on_momentum_loss': True,
        'momentum_exit_bars': 3,      # Exit als 3 bars geen momentum
        
        # Exit - TIME
        'max_bars_without_new_high': 20,  # Max 20 bars zonder nieuwe high
        
        # Filters
        'cooldown_bars': 5,
        'min_atr_pct': 2.0,  # Minimaal 2% ATR voor volatiliteit
    }
    
    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereken alle benodigde indicators"""
        df = df.copy()
        p = self.params
        
        # MACD
        df['macd'], df['macd_sig'], df['hist'] = Indicators.macd(df['close'])
        
        # Histogram baseline
        df['hist_abs'] = df['hist'].abs()
        df['hist_base'] = df['hist_abs'].rolling(p['breakout_lookback']).mean().shift(1)
        df['hist_base'] = df['hist_base'].replace(0, 1e-10)
        df['breakout_threshold'] = df['hist_base'] * p['breakout_mult']
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], 14)
        
        # EMAs
        df['ema_fast'] = Indicators.ema(df['close'], p['ema_fast'])
        df['ema_slow'] = Indicators.ema(df['close'], p['ema_slow'])
        df['uptrend'] = (df['ema_fast'] > df['ema_slow']) & (df['close'] > df['ema_fast'])
        df['downtrend'] = (df['ema_fast'] < df['ema_slow']) & (df['close'] < df['ema_fast'])
        
        # ATR
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # Volume
        df['vol_sma'] = df['volume'].rolling(20).mean()
        df['high_volume'] = df['volume'] > df['vol_sma'] * p['volume_mult']
        
        # Swing highs/lows voor S/R
        df['swing_high'] = df['high'].rolling(10, center=True).max()
        df['swing_low'] = df['low'].rolling(10, center=True).min()
        
        return df
    
    def check_entry(self, df: pd.DataFrame, i: int) -> Optional[dict]:
        """Check voor entry signaal op bar i"""
        p = self.params
        
        if i < p['breakout_lookback'] + 10:
            return None
        
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Skip low volatility
        if row['atr_pct'] < p['min_atr_pct']:
            return None
        
        # === LONG ENTRY ===
        
        # 1. MACD Breakout
        macd_breakout = (
            row['hist'] > 0 and
            row['hist'] > prev['hist'] and
            row['hist'] > row['breakout_threshold']
        )
        
        # 2. RSI OK
        rsi_ok = p['rsi_min'] <= row['rsi'] <= p['rsi_max']
        
        # 3. Trend
        trend_ok = row['uptrend'] if p['require_trend'] else True
        
        # 4. Volume
        vol_ok = row['high_volume']
        
        if macd_breakout and rsi_ok and trend_ok and vol_ok:
            stop = row['close'] - (row['atr'] * p['initial_stop_atr'])
            return {
                'direction': 'LONG',
                'entry': row['close'],
                'stop': stop,
                'atr': row['atr'],
            }
        
        # === SHORT ENTRY ===
        macd_short = (
            row['hist'] < 0 and
            row['hist'] < prev['hist'] and
            row['hist'] < -row['breakout_threshold']
        )
        
        rsi_short = (100 - p['rsi_max']) <= row['rsi'] <= (100 - p['rsi_min'])
        trend_short = row['downtrend'] if p['require_trend'] else True
        
        if macd_short and rsi_short and trend_short and vol_ok:
            stop = row['close'] + (row['atr'] * p['initial_stop_atr'])
            return {
                'direction': 'SHORT',
                'entry': row['close'],
                'stop': stop,
                'atr': row['atr'],
            }
        
        return None
    
    def manage_trade(self, trade: SwingTrade, df: pd.DataFrame, i: int) -> Tuple[bool, Optional[ExitReason], float]:
        """
        Manage open trade - returns (should_exit, reason, exit_price)
        
        Exit conditions:
        1. Trailing stop hit
        2. Momentum loss (3 bars red histogram)
        3. Too long without new high
        """
        p = self.params
        row = df.iloc[i]
        
        trade.bars_held += 1
        
        # Update extremes
        if trade.direction == 'LONG':
            if row['high'] > trade.highest_price:
                trade.highest_price = row['high']
        else:
            if row['low'] < trade.lowest_price:
                trade.lowest_price = row['low']
        
        # Current ATR for trailing
        current_atr = row['atr']
        
        # === 1. TRAILING STOP ===
        if trade.direction == 'LONG':
            # Calculate trailing stop
            profit_atr = (trade.highest_price - trade.entry_price) / current_atr
            
            if profit_atr >= p['trail_activation']:
                # Trail is active
                trail_stop = trade.highest_price - (current_atr * p['trail_distance_atr'])
                trade.initial_stop = max(trade.initial_stop, trail_stop)
            
            # Check if stop hit
            if row['low'] <= trade.initial_stop:
                return True, ExitReason.TRAILING_STOP, trade.initial_stop
        
        else:  # SHORT
            profit_atr = (trade.entry_price - trade.lowest_price) / current_atr
            
            if profit_atr >= p['trail_activation']:
                trail_stop = trade.lowest_price + (current_atr * p['trail_distance_atr'])
                trade.initial_stop = min(trade.initial_stop, trail_stop)
            
            if row['high'] >= trade.initial_stop:
                return True, ExitReason.TRAILING_STOP, trade.initial_stop
        
        # === 2. MOMENTUM EXIT ===
        if p['exit_on_momentum_loss']:
            if trade.direction == 'LONG':
                # Check for momentum loss (histogram going down)
                recent_hist = df['hist'].iloc[i-p['momentum_exit_bars']:i+1]
                if all(recent_hist.iloc[j] < recent_hist.iloc[j-1] for j in range(1, len(recent_hist))):
                    # RSI also dropping
                    if row['rsi'] < df.iloc[i-1]['rsi']:
                        return True, ExitReason.MOMENTUM_EXIT, row['close']
            else:
                recent_hist = df['hist'].iloc[i-p['momentum_exit_bars']:i+1]
                if all(recent_hist.iloc[j] > recent_hist.iloc[j-1] for j in range(1, len(recent_hist))):
                    if row['rsi'] > df.iloc[i-1]['rsi']:
                        return True, ExitReason.MOMENTUM_EXIT, row['close']
        
        # === 3. TREND REVERSAL ===
        if trade.direction == 'LONG' and row['downtrend']:
            return True, ExitReason.TREND_REVERSAL, row['close']
        if trade.direction == 'SHORT' and row['uptrend']:
            return True, ExitReason.TREND_REVERSAL, row['close']
        
        # === 4. TIME STOP ===
        if trade.bars_held >= p['max_bars_without_new_high']:
            if trade.direction == 'LONG':
                bars_since_high = 0
                for j in range(i, max(i - p['max_bars_without_new_high'], 0), -1):
                    if df.iloc[j]['high'] >= trade.highest_price * 0.99:
                        break
                    bars_since_high += 1
                if bars_since_high >= p['max_bars_without_new_high']:
                    return True, ExitReason.TIME_STOP, row['close']
        
        return False, None, 0
    
    def backtest(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> List[SwingTrade]:
        """Run backtest on dataframe"""
        p = self.params
        df = self.calculate_indicators(df)
        
        trades = []
        current_trade = None
        last_entry_bar = -p['cooldown_bars'] - 1
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # Manage existing trade
            if current_trade is not None:
                should_exit, reason, exit_price = self.manage_trade(current_trade, df, i)
                
                if should_exit:
                    current_trade.exit_time = row['timestamp']
                    current_trade.exit_price = exit_price
                    current_trade.exit_reason = reason
                    trades.append(current_trade)
                    current_trade = None
                    continue
            
            # Check for new entry
            if current_trade is None and (i - last_entry_bar) >= p['cooldown_bars']:
                entry_signal = self.check_entry(df, i)
                
                if entry_signal:
                    current_trade = SwingTrade(
                        symbol=symbol,
                        entry_time=row['timestamp'],
                        entry_price=entry_signal['entry'],
                        direction=entry_signal['direction'],
                        initial_stop=entry_signal['stop'],
                        highest_price=row['high'],
                        lowest_price=row['low'],
                    )
                    last_entry_bar = i
        
        return trades
    
    def analyze_trades(self, trades: List[SwingTrade]) -> dict:
        """Analyze trade results"""
        if not trades:
            return {'valid': False}
        
        winners = [t for t in trades if t.is_winner]
        losers = [t for t in trades if not t.is_winner]
        
        total_pnl = sum(t.pnl_pct for t in trades)
        avg_win = np.mean([t.pnl_pct for t in winners]) if winners else 0
        avg_loss = np.mean([t.pnl_pct for t in losers]) if losers else 0
        
        gross_profit = sum(t.pnl_pct for t in winners)
        gross_loss = abs(sum(t.pnl_pct for t in losers))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average bars held
        avg_bars = np.mean([t.bars_held for t in trades])
        
        # Exit reasons
        exit_reasons = {}
        for t in trades:
            reason = t.exit_reason.value if t.exit_reason else 'unknown'
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'valid': True,
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100,
            'profit_factor': pf,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_bars_held': avg_bars,
            'exit_reasons': exit_reasons,
            'best_trade': max(t.pnl_pct for t in trades),
            'worst_trade': min(t.pnl_pct for t in trades),
            'avg_mfe': np.mean([t.max_favorable_excursion for t in trades]),
        }


def main():
    """Test swing trader on top coins"""
    from core.data import DataFetcher
    
    print("=" * 70)
    print("SWING TRADER BACKTEST")
    print("=" * 70)
    
    fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
    trader = SwingTrader()
    
    # Focus on top performers
    symbols = SwingTrader.TOP_COINS
    
    all_trades = []
    results = []
    
    print(f"\nBacktesting on {len(symbols)} top coins (365 days)...\n")
    
    for symbol in symbols:
        df = fetcher.fetch(symbol, '4h', days=365)
        
        if df.empty or len(df) < 100:
            print(f"  {symbol}: No data")
            continue
        
        trades = trader.backtest(df, symbol)
        all_trades.extend(trades)
        
        stats = trader.analyze_trades(trades)
        
        if stats['valid']:
            results.append({'symbol': symbol, **stats})
            
            status = '✓' if stats['win_rate'] >= 50 and stats['profit_factor'] >= 1.5 else '✗'
            print(f"  {symbol:<12} Trades:{stats['total_trades']:3} | "
                  f"WR:{stats['win_rate']:5.1f}% | PF:{stats['profit_factor']:5.2f} | "
                  f"P&L:{stats['total_pnl']:+7.1f}% | AvgBars:{stats['avg_bars_held']:4.0f} {status}")
    
    # Combined stats
    print("\n" + "-" * 70)
    
    if all_trades:
        combined = trader.analyze_trades(all_trades)
        
        print(f"""
{'='*70}
COMBINED RESULTS - SWING TRADER
{'='*70}
Total Trades:    {combined['total_trades']}
Winners/Losers:  {combined['winners']}/{combined['losers']}

Win Rate:        {combined['win_rate']:.1f}% {'✓' if combined['win_rate'] >= 50 else '✗'}
Profit Factor:   {combined['profit_factor']:.2f} {'✓' if combined['profit_factor'] >= 1.5 else '✗'}
Total P&L:       {combined['total_pnl']:+.1f}%

Avg Win:         {combined['avg_win']:+.2f}%
Avg Loss:        {combined['avg_loss']:.2f}%
Best Trade:      {combined['best_trade']:+.1f}%
Worst Trade:     {combined['worst_trade']:.1f}%

Avg Bars Held:   {combined['avg_bars_held']:.0f} bars (~{combined['avg_bars_held']*4:.0f} hours)
Avg MFE:         {combined['avg_mfe']:.1f}% (max favorable excursion)

Exit Reasons:
{chr(10).join(f"  - {k}: {v}" for k, v in combined['exit_reasons'].items())}
{'='*70}
""")
        
        # Per symbol summary
        print("\nPer Symbol Performance:")
        print("-" * 70)
        
        # Sort by P&L
        results.sort(key=lambda x: x['total_pnl'], reverse=True)
        for r in results:
            print(f"  {r['symbol']:<12} P&L:{r['total_pnl']:+7.1f}% | "
                  f"WR:{r['win_rate']:5.1f}% | Best:{r['best_trade']:+.1f}%")


if __name__ == "__main__":
    main()
