"""
Backtesting Engine - Core backtesting logic with proper metrics
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class Side(Enum):
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Single trade record"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Side
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    size: float = 1.0
    exit_reason: str = ""  # 'tp', 'sl', 'signal'
    
    @property
    def pnl_pct(self) -> float:
        if self.side == Side.LONG:
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100
    
    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0
    
    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0


@dataclass
class BacktestResult:
    """Backtest results with all metrics"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    # Targets
    TARGET_WIN_RATE = 50.0
    TARGET_PROFIT_FACTOR = 1.5
    TARGET_MAX_DD = 20.0
    
    @property
    def total_trades(self) -> int:
        return len(self.trades)
    
    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)
    
    @property
    def losses(self) -> int:
        return self.total_trades - self.wins
    
    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0
    
    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_pct for t in self.trades)
    
    @property
    def avg_win(self) -> float:
        wins = [t.pnl_pct for t in self.trades if t.is_win]
        return np.mean(wins) if wins else 0
    
    @property
    def avg_loss(self) -> float:
        losses = [t.pnl_pct for t in self.trades if not t.is_win]
        return np.mean(losses) if losses else 0
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_pct for t in self.trades if t.is_win)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if not t.is_win))
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0
        peak = self.equity_curve[0]
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            max_dd = max(max_dd, dd)
        return max_dd
    
    @property
    def sharpe_ratio(self) -> float:
        if not self.trades:
            return 0
        returns = [t.pnl_pct for t in self.trades]
        if np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    @property
    def expectancy(self) -> float:
        """Expected value per trade"""
        if self.total_trades == 0:
            return 0
        wr = self.win_rate / 100
        return (wr * self.avg_win) - ((1 - wr) * abs(self.avg_loss))
    
    @property
    def meets_targets(self) -> bool:
        return (
            self.win_rate >= self.TARGET_WIN_RATE and
            self.profit_factor >= self.TARGET_PROFIT_FACTOR and
            self.max_drawdown <= self.TARGET_MAX_DD
        )
    
    def summary(self) -> str:
        """Pretty print summary"""
        wr_status = "✓" if self.win_rate >= self.TARGET_WIN_RATE else "✗"
        pf_status = "✓" if self.profit_factor >= self.TARGET_PROFIT_FACTOR else "✗"
        dd_status = "✓" if self.max_drawdown <= self.TARGET_MAX_DD else "✗"
        
        return f"""
{'='*60}
BACKTEST RESULTS
{'='*60}
Trades:        {self.total_trades}
Wins/Losses:   {self.wins}/{self.losses}

Win Rate:      {self.win_rate:.1f}% {wr_status} (target: >={self.TARGET_WIN_RATE}%)
Profit Factor: {self.profit_factor:.2f} {pf_status} (target: >={self.TARGET_PROFIT_FACTOR})
Max Drawdown:  {self.max_drawdown:.1f}% {dd_status} (target: <={self.TARGET_MAX_DD}%)

Total P&L:     {self.total_pnl:+.1f}%
Avg Win:       {self.avg_win:+.2f}%
Avg Loss:      {self.avg_loss:.2f}%
Expectancy:    {self.expectancy:.2f}% per trade
Sharpe Ratio:  {self.sharpe_ratio:.2f}

TARGETS MET:   {'YES ✓' if self.meets_targets else 'NO ✗'}
{'='*60}
"""


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """
        Run backtest on signals.
        
        Args:
            df: OHLCV dataframe
            signals: DataFrame with columns: 
                     'signal' (1=buy, -1=sell, 0=none),
                     'stop_loss', 'take_profit'
        """
        result = BacktestResult()
        equity = self.initial_capital
        result.equity_curve = [equity]
        
        in_position = False
        position_side = None
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        
        for i in range(len(df)):
            row = df.iloc[i]
            sig_row = signals.iloc[i]
            
            current_price = row['close']
            high = row['high']
            low = row['low']
            
            if in_position:
                # Check exit conditions
                exit_reason = None
                exit_price = None
                
                if position_side == Side.LONG:
                    # Check stop loss
                    if low <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'sl'
                    # Check take profit
                    elif high >= take_profit:
                        exit_price = take_profit
                        exit_reason = 'tp'
                else:  # SHORT
                    if high >= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'sl'
                    elif low <= take_profit:
                        exit_price = take_profit
                        exit_reason = 'tp'
                
                if exit_reason:
                    trade = Trade(
                        entry_time=entry_time,
                        exit_time=row['timestamp'],
                        side=position_side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        exit_reason=exit_reason,
                    )
                    result.trades.append(trade)
                    
                    # Update equity
                    equity *= (1 + trade.pnl_pct / 100)
                    in_position = False
            
            # Check for new entry
            if not in_position:
                signal = sig_row.get('signal', 0)
                
                if signal == 1:  # Long
                    in_position = True
                    position_side = Side.LONG
                    entry_price = current_price
                    entry_time = row['timestamp']
                    stop_loss = sig_row['stop_loss']
                    take_profit = sig_row['take_profit']
                    
                elif signal == -1:  # Short
                    in_position = True
                    position_side = Side.SHORT
                    entry_price = current_price
                    entry_time = row['timestamp']
                    stop_loss = sig_row['stop_loss']
                    take_profit = sig_row['take_profit']
            
            result.equity_curve.append(equity)
        
        return result
