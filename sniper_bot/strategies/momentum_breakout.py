"""
Momentum Breakout Strategy V1
=============================
Entry: MACD histogram breakout + RSI confirmation + Trend alignment
Exit: ATR-based SL/TP
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspace/sniper_bot')

from core.strategy import Strategy, SignalBuilder
from core.indicators import Indicators


class MomentumBreakout(Strategy):
    """
    Momentum breakout strategy using MACD + RSI + Trend
    """
    
    name = "MomentumBreakout"
    
    # Default parameters
    DEFAULT_PARAMS = {
        # MACD
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        
        # Breakout
        'breakout_lookback': 20,
        'breakout_mult': 2.5,
        
        # RSI
        'rsi_period': 14,
        'rsi_buy_min': 40,
        'rsi_buy_max': 70,
        'rsi_sell_min': 30,
        'rsi_sell_max': 60,
        
        # Trend
        'trend_ema_fast': 50,
        'trend_ema_slow': 200,
        'use_trend_filter': True,
        
        # Risk Management
        'atr_period': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,
        
        # Signal filters
        'cooldown_bars': 3,
    }
    
    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals"""
        p = self.params
        
        # Calculate indicators
        df = df.copy()
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = Indicators.macd(
            df['close'], p['macd_fast'], p['macd_slow'], p['macd_signal']
        )
        
        # MACD histogram baseline (for breakout detection)
        df['hist_abs'] = df['macd_hist'].abs()
        df['hist_baseline'] = df['hist_abs'].rolling(p['breakout_lookback']).mean().shift(1)
        df['hist_baseline'] = df['hist_baseline'].replace(0, 1e-10)
        df['breakout_threshold'] = df['hist_baseline'] * p['breakout_mult']
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], p['rsi_period'])
        
        # Trend EMAs
        df['ema_fast'] = Indicators.ema(df['close'], p['trend_ema_fast'])
        df['ema_slow'] = Indicators.ema(df['close'], p['trend_ema_slow'])
        df['uptrend'] = df['ema_fast'] > df['ema_slow']
        df['downtrend'] = df['ema_fast'] < df['ema_slow']
        
        # ATR for SL/TP
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], p['atr_period'])
        
        # Build signals
        signals = SignalBuilder(len(df))
        last_signal_bar = -p['cooldown_bars'] - 1
        
        for i in range(p['breakout_lookback'] + 2, len(df)):
            # Cooldown check
            if i - last_signal_bar < p['cooldown_bars']:
                continue
            
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            
            # BUY CONDITIONS
            buy_macd = (
                row['macd_hist'] > 0 and
                row['macd_hist'] > prev['macd_hist'] and
                row['macd_hist'] > row['breakout_threshold']
            )
            buy_rsi = p['rsi_buy_min'] <= row['rsi'] <= p['rsi_buy_max']
            buy_trend = row['uptrend'] if p['use_trend_filter'] else True
            
            if buy_macd and buy_rsi and buy_trend:
                stop_loss = row['close'] - (row['atr'] * p['sl_atr_mult'])
                take_profit = row['close'] + (row['atr'] * p['tp_atr_mult'])
                signals.set_buy(i, stop_loss, take_profit)
                last_signal_bar = i
                continue
            
            # SELL CONDITIONS
            sell_macd = (
                row['macd_hist'] < 0 and
                row['macd_hist'] < prev['macd_hist'] and
                row['macd_hist'] < -row['breakout_threshold']
            )
            sell_rsi = p['rsi_sell_min'] <= row['rsi'] <= p['rsi_sell_max']
            sell_trend = row['downtrend'] if p['use_trend_filter'] else True
            
            if sell_macd and sell_rsi and sell_trend:
                stop_loss = row['close'] + (row['atr'] * p['sl_atr_mult'])
                take_profit = row['close'] - (row['atr'] * p['tp_atr_mult'])
                signals.set_sell(i, stop_loss, take_profit)
                last_signal_bar = i
        
        return signals.get()


# Quick test
if __name__ == "__main__":
    from core.data import DataFetcher
    from core.backtest import Backtester
    
    print("Testing MomentumBreakout Strategy...")
    
    fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
    df = fetcher.fetch('BTC/USDT', '4h', days=365)
    
    if not df.empty:
        strategy = MomentumBreakout()
        signals = strategy.generate_signals(df)
        
        backtester = Backtester()
        result = backtester.run(df, signals)
        
        print(result.summary())
