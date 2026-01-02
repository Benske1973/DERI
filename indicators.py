# indicators.py - Technical Analysis Indicators Module
"""
Comprehensive technical analysis module with:
- Trend indicators (EMA, SMA, ADX)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Volume indicators (OBV, Volume Profile)
- SMC/ICT concepts (FVG, Order Blocks, Liquidity)
- Market structure analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from config import config, TrendDirection

@dataclass
class FairValueGap:
    """Fair Value Gap (FVG) structure."""
    type: str  # BULLISH or BEARISH
    top: float
    bottom: float
    midpoint: float
    size: float
    bar_index: int
    filled: bool = False
    fill_percent: float = 0.0

@dataclass
class OrderBlock:
    """Order Block structure."""
    type: str  # BULLISH or BEARISH
    high: float
    low: float
    open_price: float
    close_price: float
    bar_index: int
    validated: bool = False
    mitigated: bool = False

@dataclass
class SwingPoint:
    """Swing High/Low structure."""
    type: str  # HIGH or LOW
    price: float
    bar_index: int
    broken: bool = False

@dataclass
class MarketStructure:
    """Market structure analysis result."""
    trend: TrendDirection
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    bos_detected: bool  # Break of Structure
    choch_detected: bool  # Change of Character
    structure_breaks: List[Dict]

@dataclass
class SignalScore:
    """Signal scoring result."""
    total_score: float
    trend_score: float
    momentum_score: float
    volume_score: float
    structure_score: float
    confluence_count: int
    details: Dict


class Indicators:
    """Technical Analysis Indicators Calculator."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()
        self.cfg = config.indicators

    # ==================== TREND INDICATORS ====================

    def ema(self, period: int, column: str = "close") -> pd.Series:
        """Exponential Moving Average."""
        return self.df[column].ewm(span=period, adjust=False).mean()

    def sma(self, period: int, column: str = "close") -> pd.Series:
        """Simple Moving Average."""
        return self.df[column].rolling(window=period).mean()

    def add_emas(self) -> pd.DataFrame:
        """Add all configured EMAs to DataFrame."""
        self.df["ema_fast"] = self.ema(self.cfg.ema_fast)
        self.df["ema_medium"] = self.ema(self.cfg.ema_medium)
        self.df["ema_slow"] = self.ema(self.cfg.ema_slow)
        self.df["ema_trend"] = self.ema(self.cfg.ema_trend)
        return self.df

    def adx(self, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed averages
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def get_trend(self) -> TrendDirection:
        """Determine current market trend."""
        if len(self.df) < self.cfg.ema_trend:
            return TrendDirection.NEUTRAL

        close = self.df["close"].iloc[-1]
        ema_fast = self.ema(self.cfg.ema_fast).iloc[-1]
        ema_slow = self.ema(self.cfg.ema_slow).iloc[-1]
        ema_trend = self.ema(self.cfg.ema_trend).iloc[-1]

        # Strong uptrend
        if close > ema_fast > ema_slow > ema_trend:
            return TrendDirection.BULLISH

        # Strong downtrend
        if close < ema_fast < ema_slow < ema_trend:
            return TrendDirection.BEARISH

        return TrendDirection.NEUTRAL

    # ==================== MOMENTUM INDICATORS ====================

    def rsi(self, period: int = None) -> pd.Series:
        """Relative Strength Index."""
        period = period or self.cfg.rsi_period
        delta = self.df["close"].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def macd(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD indicator.

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast = self.df["close"].ewm(span=self.cfg.macd_fast, adjust=False).mean()
        slow = self.df["close"].ewm(span=self.cfg.macd_slow, adjust=False).mean()

        macd_line = fast - slow
        signal_line = macd_line.ewm(span=self.cfg.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Returns:
            Tuple of (%K, %D)
        """
        low_min = self.df["low"].rolling(window=k_period).min()
        high_max = self.df["high"].rolling(window=k_period).max()

        stoch_k = 100 * (self.df["close"] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def momentum(self, period: int = 10) -> pd.Series:
        """Price momentum."""
        return self.df["close"] - self.df["close"].shift(period)

    def roc(self, period: int = 10) -> pd.Series:
        """Rate of Change."""
        return (self.df["close"] / self.df["close"].shift(period) - 1) * 100

    # ==================== VOLATILITY INDICATORS ====================

    def atr(self, period: int = None) -> pd.Series:
        """Average True Range."""
        period = period or self.cfg.atr_period

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.ewm(span=period, adjust=False).mean()

    def bollinger_bands(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle = self.df["close"].rolling(window=self.cfg.bb_period).mean()
        std = self.df["close"].rolling(window=self.cfg.bb_period).std()

        upper = middle + (std * self.cfg.bb_std)
        lower = middle - (std * self.cfg.bb_std)

        return upper, middle, lower

    def keltner_channels(self, period: int = 20, atr_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.

        Returns:
            Tuple of (Upper, Middle, Lower)
        """
        middle = self.ema(period)
        atr_val = self.atr(period)

        upper = middle + (atr_mult * atr_val)
        lower = middle - (atr_mult * atr_val)

        return upper, middle, lower

    # ==================== VOLUME INDICATORS ====================

    def obv(self) -> pd.Series:
        """On Balance Volume."""
        obv = pd.Series(index=self.df.index, dtype=float)
        obv.iloc[0] = self.df["volume"].iloc[0]

        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > self.df["close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.df["volume"].iloc[i]
            elif self.df["close"].iloc[i] < self.df["close"].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def volume_sma(self, period: int = None) -> pd.Series:
        """Volume Simple Moving Average."""
        period = period or self.cfg.volume_ma_period
        return self.df["volume"].rolling(window=period).mean()

    def is_volume_spike(self) -> bool:
        """Check if current volume is a spike."""
        if len(self.df) < self.cfg.volume_ma_period:
            return False

        current_vol = self.df["volume"].iloc[-1]
        avg_vol = self.volume_sma().iloc[-1]

        return current_vol > avg_vol * self.cfg.volume_spike_multiplier

    def vwap(self) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return (typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    # ==================== SMC/ICT CONCEPTS ====================

    def find_swing_highs(self, lookback: int = None) -> List[SwingPoint]:
        """Find swing high points."""
        lookback = lookback or self.cfg.swing_lookback
        swings = []

        for i in range(lookback, len(self.df) - lookback):
            high = self.df["high"].iloc[i]
            is_swing = True

            for j in range(1, lookback + 1):
                if self.df["high"].iloc[i - j] >= high or self.df["high"].iloc[i + j] >= high:
                    is_swing = False
                    break

            if is_swing:
                swings.append(SwingPoint(
                    type="HIGH",
                    price=high,
                    bar_index=i,
                    broken=False
                ))

        return swings

    def find_swing_lows(self, lookback: int = None) -> List[SwingPoint]:
        """Find swing low points."""
        lookback = lookback or self.cfg.swing_lookback
        swings = []

        for i in range(lookback, len(self.df) - lookback):
            low = self.df["low"].iloc[i]
            is_swing = True

            for j in range(1, lookback + 1):
                if self.df["low"].iloc[i - j] <= low or self.df["low"].iloc[i + j] <= low:
                    is_swing = False
                    break

            if is_swing:
                swings.append(SwingPoint(
                    type="LOW",
                    price=low,
                    bar_index=i,
                    broken=False
                ))

        return swings

    def find_fvg(self, min_size_atr: float = None) -> List[FairValueGap]:
        """
        Find Fair Value Gaps (imbalances).

        A bullish FVG: Low[i] > High[i-2]
        A bearish FVG: High[i] < Low[i-2]
        """
        min_size_atr = min_size_atr or self.cfg.fvg_min_size_atr
        atr_values = self.atr()
        fvgs = []

        for i in range(2, len(self.df)):
            atr_val = atr_values.iloc[i]

            # Bullish FVG
            if self.df["low"].iloc[i] > self.df["high"].iloc[i-2]:
                gap_size = self.df["low"].iloc[i] - self.df["high"].iloc[i-2]
                if gap_size >= atr_val * min_size_atr:
                    fvgs.append(FairValueGap(
                        type="BULLISH",
                        top=self.df["low"].iloc[i],
                        bottom=self.df["high"].iloc[i-2],
                        midpoint=(self.df["low"].iloc[i] + self.df["high"].iloc[i-2]) / 2,
                        size=gap_size,
                        bar_index=i
                    ))

            # Bearish FVG
            if self.df["high"].iloc[i] < self.df["low"].iloc[i-2]:
                gap_size = self.df["low"].iloc[i-2] - self.df["high"].iloc[i]
                if gap_size >= atr_val * min_size_atr:
                    fvgs.append(FairValueGap(
                        type="BEARISH",
                        top=self.df["low"].iloc[i-2],
                        bottom=self.df["high"].iloc[i],
                        midpoint=(self.df["low"].iloc[i-2] + self.df["high"].iloc[i]) / 2,
                        size=gap_size,
                        bar_index=i
                    ))

        return fvgs

    def find_order_blocks(self, validity_bars: int = None) -> List[OrderBlock]:
        """
        Find Order Blocks.

        Bullish OB: Last bearish candle before a bullish impulse
        Bearish OB: Last bullish candle before a bearish impulse
        """
        validity_bars = validity_bars or self.cfg.ob_validity_bars
        obs = []

        for i in range(2, len(self.df) - 1):
            # Check for impulse move (significant price movement)
            atr_val = self.atr().iloc[i]

            # Bullish impulse (price moving up significantly)
            if self.df["close"].iloc[i] - self.df["low"].iloc[i-1] > atr_val * 1.5:
                # Find last bearish candle before impulse
                for j in range(i-1, max(0, i-5), -1):
                    if self.df["close"].iloc[j] < self.df["open"].iloc[j]:  # Bearish candle
                        obs.append(OrderBlock(
                            type="BULLISH",
                            high=self.df["high"].iloc[j],
                            low=self.df["low"].iloc[j],
                            open_price=self.df["open"].iloc[j],
                            close_price=self.df["close"].iloc[j],
                            bar_index=j
                        ))
                        break

            # Bearish impulse (price moving down significantly)
            if self.df["high"].iloc[i-1] - self.df["close"].iloc[i] > atr_val * 1.5:
                # Find last bullish candle before impulse
                for j in range(i-1, max(0, i-5), -1):
                    if self.df["close"].iloc[j] > self.df["open"].iloc[j]:  # Bullish candle
                        obs.append(OrderBlock(
                            type="BEARISH",
                            high=self.df["high"].iloc[j],
                            low=self.df["low"].iloc[j],
                            open_price=self.df["open"].iloc[j],
                            close_price=self.df["close"].iloc[j],
                            bar_index=j
                        ))
                        break

        return obs

    def analyze_market_structure(self) -> MarketStructure:
        """
        Complete market structure analysis.

        Returns:
            MarketStructure with trend, swing points, BOS, and CHoCH
        """
        swing_highs = self.find_swing_highs()
        swing_lows = self.find_swing_lows()

        # Get last swing points
        last_high = swing_highs[-1] if swing_highs else None
        last_low = swing_lows[-1] if swing_lows else None

        # Determine trend based on swing structure
        trend = self.get_trend()

        # Check for BOS (Break of Structure)
        bos_detected = False
        choch_detected = False
        structure_breaks = []

        current_price = self.df["close"].iloc[-1]

        if last_high and last_low:
            # In uptrend, BOS = new higher high
            if trend == TrendDirection.BULLISH:
                if current_price > last_high.price:
                    bos_detected = True
                    structure_breaks.append({
                        "type": "BOS",
                        "direction": "BULLISH",
                        "level": last_high.price
                    })
                # CHoCH = break below last low in uptrend
                if current_price < last_low.price:
                    choch_detected = True
                    structure_breaks.append({
                        "type": "CHOCH",
                        "direction": "BEARISH",
                        "level": last_low.price
                    })

            # In downtrend, BOS = new lower low
            elif trend == TrendDirection.BEARISH:
                if current_price < last_low.price:
                    bos_detected = True
                    structure_breaks.append({
                        "type": "BOS",
                        "direction": "BEARISH",
                        "level": last_low.price
                    })
                # CHoCH = break above last high in downtrend
                if current_price > last_high.price:
                    choch_detected = True
                    structure_breaks.append({
                        "type": "CHOCH",
                        "direction": "BULLISH",
                        "level": last_high.price
                    })

        return MarketStructure(
            trend=trend,
            last_swing_high=last_high,
            last_swing_low=last_low,
            bos_detected=bos_detected,
            choch_detected=choch_detected,
            structure_breaks=structure_breaks
        )

    # ==================== SIGNAL SCORING ====================

    def calculate_signal_score(self, direction: str = "LONG") -> SignalScore:
        """
        Calculate comprehensive signal score.

        Args:
            direction: "LONG" or "SHORT"

        Returns:
            SignalScore with detailed scoring breakdown
        """
        scores = {
            "trend": 0,
            "momentum": 0,
            "volume": 0,
            "structure": 0
        }
        details = {}
        confluence_count = 0

        # === Trend Score (25 points max) ===
        trend = self.get_trend()
        if direction == "LONG" and trend == TrendDirection.BULLISH:
            scores["trend"] = 25
            confluence_count += 1
            details["trend"] = "Aligned with bullish trend"
        elif direction == "SHORT" and trend == TrendDirection.BEARISH:
            scores["trend"] = 25
            confluence_count += 1
            details["trend"] = "Aligned with bearish trend"
        elif trend == TrendDirection.NEUTRAL:
            scores["trend"] = 10
            details["trend"] = "Neutral trend"
        else:
            scores["trend"] = 0
            details["trend"] = "Against trend"

        # === Momentum Score (25 points max) ===
        rsi_value = self.rsi().iloc[-1]
        macd_line, signal_line, histogram = self.macd()
        macd_val = histogram.iloc[-1]

        if direction == "LONG":
            if 30 <= rsi_value <= 50:  # RSI coming out of oversold
                scores["momentum"] += 10
                confluence_count += 1
            if macd_val > 0 and macd_val > histogram.iloc[-2]:  # MACD bullish
                scores["momentum"] += 15
                confluence_count += 1
        else:  # SHORT
            if 50 <= rsi_value <= 70:  # RSI coming out of overbought
                scores["momentum"] += 10
                confluence_count += 1
            if macd_val < 0 and macd_val < histogram.iloc[-2]:  # MACD bearish
                scores["momentum"] += 15
                confluence_count += 1

        details["momentum"] = f"RSI: {rsi_value:.1f}, MACD Hist: {macd_val:.4f}"

        # === Volume Score (25 points max) ===
        if self.is_volume_spike():
            scores["volume"] = 25
            confluence_count += 1
            details["volume"] = "Volume spike detected"
        else:
            vol_ratio = self.df["volume"].iloc[-1] / self.volume_sma().iloc[-1]
            scores["volume"] = min(25, int(vol_ratio * 12.5))
            details["volume"] = f"Volume ratio: {vol_ratio:.2f}x"

        # === Structure Score (25 points max) ===
        structure = self.analyze_market_structure()

        if direction == "LONG":
            if structure.bos_detected:
                scores["structure"] += 15
                confluence_count += 1
            if structure.choch_detected:
                scores["structure"] += 10
        else:  # SHORT
            if structure.bos_detected:
                scores["structure"] += 15
                confluence_count += 1
            if structure.choch_detected:
                scores["structure"] += 10

        details["structure"] = f"BOS: {structure.bos_detected}, CHoCH: {structure.choch_detected}"

        total_score = sum(scores.values())

        return SignalScore(
            total_score=total_score,
            trend_score=scores["trend"],
            momentum_score=scores["momentum"],
            volume_score=scores["volume"],
            structure_score=scores["structure"],
            confluence_count=confluence_count,
            details=details
        )

    def add_all_indicators(self) -> pd.DataFrame:
        """Add all indicators to the DataFrame."""
        # EMAs
        self.add_emas()

        # Momentum
        self.df["rsi"] = self.rsi()
        macd_line, signal_line, histogram = self.macd()
        self.df["macd"] = macd_line
        self.df["macd_signal"] = signal_line
        self.df["macd_hist"] = histogram

        # Volatility
        self.df["atr"] = self.atr()
        upper, middle, lower = self.bollinger_bands()
        self.df["bb_upper"] = upper
        self.df["bb_middle"] = middle
        self.df["bb_lower"] = lower

        # Volume
        self.df["volume_sma"] = self.volume_sma()
        self.df["obv"] = self.obv()

        return self.df
