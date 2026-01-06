"""
Technical Indicators - All indicators needed for strategies
"""
import pandas as pd
import numpy as np
from typing import Tuple


class Indicators:
    """Technical indicators library"""
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD - Returns (macd_line, signal_line, histogram)"""
        ema_fast = Indicators.ema(close, fast)
        ema_slow = Indicators.ema(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, 
                        std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands - Returns (upper, middle, lower)"""
        middle = Indicators.sma(close, period)
        std_dev = close.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator - Returns (%K, %D)"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """Average Directional Index"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = Indicators.atr(high, low, close, period)
        
        plus_di = 100 * Indicators.ema(plus_dm, period) / atr
        minus_di = 100 * Indicators.ema(minus_dm, period) / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = Indicators.ema(dx, period)
        
        return adx
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series,
                     lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Find pivot highs and lows for S/R detection.
        Returns (pivot_highs, pivot_lows) as boolean series.
        """
        pivot_high = pd.Series([False] * len(high), index=high.index)
        pivot_low = pd.Series([False] * len(low), index=low.index)
        
        for i in range(lookback, len(high) - lookback):
            # Pivot high
            is_high = True
            for j in range(1, lookback + 1):
                if high.iloc[i] <= high.iloc[i - j] or high.iloc[i] <= high.iloc[i + j]:
                    is_high = False
                    break
            pivot_high.iloc[i] = is_high
            
            # Pivot low
            is_low = True
            for j in range(1, lookback + 1):
                if low.iloc[i] >= low.iloc[i - j] or low.iloc[i] >= low.iloc[i + j]:
                    is_low = False
                    break
            pivot_low.iloc[i] = is_low
        
        return pivot_high, pivot_low
    
    @staticmethod
    def support_resistance(df: pd.DataFrame, lookback: int = 5, 
                          cluster_pct: float = 0.02) -> Tuple[list, list]:
        """
        Find key support and resistance levels.
        Returns (resistance_levels, support_levels).
        """
        pivot_high, pivot_low = Indicators.pivot_points(
            df['high'], df['low'], df['close'], lookback
        )
        
        resistance = df.loc[pivot_high, 'high'].tolist()
        support = df.loc[pivot_low, 'low'].tolist()
        
        # Cluster nearby levels
        def cluster(levels, threshold):
            if not levels:
                return []
            levels = sorted(levels)
            clusters = []
            current = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - np.mean(current)) / np.mean(current) <= threshold:
                    current.append(level)
                else:
                    clusters.append(np.mean(current))
                    current = [level]
            clusters.append(np.mean(current))
            return clusters
        
        return cluster(resistance, cluster_pct), cluster(support, cluster_pct)
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.Series:
        """Simple volume profile - volume at price levels"""
        price_range = df['close'].max() - df['close'].min()
        bin_size = price_range / bins
        
        levels = {}
        for _, row in df.iterrows():
            bin_idx = int((row['close'] - df['close'].min()) / bin_size)
            bin_idx = min(bin_idx, bins - 1)
            price_level = df['close'].min() + (bin_idx + 0.5) * bin_size
            levels[price_level] = levels.get(price_level, 0) + row['volume']
        
        return pd.Series(levels).sort_index()
