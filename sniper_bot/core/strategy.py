"""
Strategy Base Class - Base class for all strategies
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class StrategyParams:
    """Strategy parameters - override in subclass"""
    pass


class Strategy(ABC):
    """Base strategy class"""
    
    name: str = "BaseStrategy"
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Must return DataFrame with columns:
        - signal: 1 (buy), -1 (sell), 0 (none)
        - stop_loss: stop loss price
        - take_profit: take profit price
        """
        pass
    
    def calculate_position_size(self, capital: float, risk_pct: float,
                               entry: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = capital * (risk_pct / 100)
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit == 0:
            return 0
        return risk_amount / risk_per_unit


class SignalBuilder:
    """Helper to build signals dataframe"""
    
    def __init__(self, length: int):
        self.signals = pd.DataFrame({
            'signal': [0] * length,
            'stop_loss': [np.nan] * length,
            'take_profit': [np.nan] * length,
        })
    
    def set_buy(self, idx: int, stop_loss: float, take_profit: float):
        self.signals.loc[idx, 'signal'] = 1
        self.signals.loc[idx, 'stop_loss'] = stop_loss
        self.signals.loc[idx, 'take_profit'] = take_profit
    
    def set_sell(self, idx: int, stop_loss: float, take_profit: float):
        self.signals.loc[idx, 'signal'] = -1
        self.signals.loc[idx, 'stop_loss'] = stop_loss
        self.signals.loc[idx, 'take_profit'] = take_profit
    
    def get(self) -> pd.DataFrame:
        return self.signals
