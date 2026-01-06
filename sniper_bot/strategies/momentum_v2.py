"""
Momentum Strategy V2 - Verbeterde versie
========================================
Verbeteringen:
- RSI momentum bevestiging (niet alleen range)
- Betere trend detectie
- Volume confirmatie
- Strengere entry criteria
"""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/workspace/sniper_bot')

from core.strategy import Strategy, SignalBuilder
from core.indicators import Indicators


class MomentumV2(Strategy):
    """Improved momentum strategy"""
    
    name = "MomentumV2"
    
    DEFAULT_PARAMS = {
        # MACD
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        
        # Breakout - strenger
        'breakout_lookback': 20,
        'breakout_mult': 2.0,  # Iets lager voor meer signalen
        
        # RSI - met momentum check
        'rsi_period': 14,
        'rsi_min': 45,        # RSI moet boven 45 zijn
        'rsi_max': 70,        # Maar niet overbought
        'rsi_rising': True,   # RSI moet stijgen
        
        # Trend - dubbele bevestiging
        'ema_fast': 20,
        'ema_slow': 50,
        'price_above_ema': True,  # Prijs moet boven EMA zijn
        
        # Volume
        'use_volume': True,
        'volume_mult': 1.2,   # Volume moet 1.2x gemiddeld zijn
        
        # Risk
        'atr_period': 14,
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 3.0,   # 2:1 R:R
        
        # Filters
        'cooldown': 5,
        'min_atr_pct': 1.0,   # Minimale volatiliteit
    }
    
    def __init__(self, params: dict = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        df = df.copy()
        
        # === INDICATORS ===
        
        # MACD
        df['macd'], df['macd_sig'], df['hist'] = Indicators.macd(
            df['close'], p['macd_fast'], p['macd_slow'], p['macd_signal']
        )
        
        # MACD histogram baseline
        df['hist_abs'] = df['hist'].abs()
        df['hist_base'] = df['hist_abs'].rolling(p['breakout_lookback']).mean().shift(1)
        df['hist_base'] = df['hist_base'].replace(0, 1e-10)
        df['threshold'] = df['hist_base'] * p['breakout_mult']
        
        # RSI
        df['rsi'] = Indicators.rsi(df['close'], p['rsi_period'])
        df['rsi_prev'] = df['rsi'].shift(1)
        
        # Trend EMAs
        df['ema_fast'] = Indicators.ema(df['close'], p['ema_fast'])
        df['ema_slow'] = Indicators.ema(df['close'], p['ema_slow'])
        
        # Volume
        df['vol_avg'] = df['volume'].rolling(20).mean()
        
        # ATR
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], p['atr_period'])
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # === SIGNAL GENERATION ===
        signals = SignalBuilder(len(df))
        last_signal = -p['cooldown'] - 1
        
        for i in range(max(p['breakout_lookback'], p['ema_slow']) + 2, len(df)):
            if i - last_signal < p['cooldown']:
                continue
            
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Skip low volatility
            if row['atr_pct'] < p['min_atr_pct']:
                continue
            
            # === BUY CONDITIONS ===
            
            # 1. MACD breakout
            macd_buy = (
                row['hist'] > 0 and                    # Histogram groen
                row['hist'] > prev['hist'] and         # Histogram groeit
                row['hist'] > row['threshold']         # Boven threshold
            )
            
            # 2. RSI confirmation
            rsi_buy = (
                p['rsi_min'] <= row['rsi'] <= p['rsi_max'] and  # In range
                (row['rsi'] > row['rsi_prev'] if p['rsi_rising'] else True)  # Stijgend
            )
            
            # 3. Trend alignment
            trend_buy = (
                row['ema_fast'] > row['ema_slow'] and          # Uptrend
                (row['close'] > row['ema_fast'] if p['price_above_ema'] else True)
            )
            
            # 4. Volume confirmation
            vol_buy = (
                row['volume'] > row['vol_avg'] * p['volume_mult']
                if p['use_volume'] else True
            )
            
            # ALL conditions must be true
            if macd_buy and rsi_buy and trend_buy and vol_buy:
                sl = row['close'] - (row['atr'] * p['sl_atr_mult'])
                tp = row['close'] + (row['atr'] * p['tp_atr_mult'])
                signals.set_buy(i, sl, tp)
                last_signal = i
                continue
            
            # === SELL CONDITIONS ===
            macd_sell = (
                row['hist'] < 0 and
                row['hist'] < prev['hist'] and
                row['hist'] < -row['threshold']
            )
            
            rsi_sell = (
                (100 - p['rsi_max']) <= row['rsi'] <= (100 - p['rsi_min']) and
                (row['rsi'] < row['rsi_prev'] if p['rsi_rising'] else True)
            )
            
            trend_sell = (
                row['ema_fast'] < row['ema_slow'] and
                (row['close'] < row['ema_fast'] if p['price_above_ema'] else True)
            )
            
            vol_sell = row['volume'] > row['vol_avg'] * p['volume_mult'] if p['use_volume'] else True
            
            if macd_sell and rsi_sell and trend_sell and vol_sell:
                sl = row['close'] + (row['atr'] * p['sl_atr_mult'])
                tp = row['close'] - (row['atr'] * p['tp_atr_mult'])
                signals.set_sell(i, sl, tp)
                last_signal = i
        
        return signals.get()


# Test
if __name__ == "__main__":
    from core.data import DataFetcher
    from core.backtest import Backtester, BacktestResult
    
    print("=" * 70)
    print("MOMENTUM V2 TEST")
    print("=" * 70)
    
    fetcher = DataFetcher(cache_dir='/workspace/sniper_bot/data')
    backtester = Backtester()
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'AVAX/USDT']
    all_trades = []
    
    for symbol in symbols:
        print(f"\n  {symbol}:")
        df = fetcher.fetch(symbol, '4h', days=180)
        
        if df.empty:
            print("    No data")
            continue
        
        strategy = MomentumV2()
        signals = strategy.generate_signals(df)
        result = backtester.run(df, signals)
        all_trades.extend(result.trades)
        
        status = '✓' if result.win_rate >= 50 and result.profit_factor >= 1.0 else '✗'
        print(f"    Trades: {result.total_trades:2} | WR: {result.win_rate:5.1f}% | "
              f"PF: {result.profit_factor:5.2f} | P&L: {result.total_pnl:+6.1f}% {status}")
    
    if all_trades:
        combined = BacktestResult(trades=all_trades)
        print(combined.summary())
