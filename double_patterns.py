# double_patterns.py - Double Top and Double Bottom Pattern Detection
"""
Detects double top and double bottom chart patterns.
Based on peak detection algorithm from Marcos Duarte (github.com/demotu/BMC)

Double Top: Bearish reversal pattern - two peaks at similar price levels
Double Bottom: Bullish reversal pattern - two troughs at similar price levels
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Result of pattern detection."""
    pattern_type: str  # "DOUBLE_TOP", "DOUBLE_BOTTOM", or "NONE"
    confidence: float  # 0-100
    price1: float  # First peak/trough price
    price2: float  # Second peak/trough price
    date1: datetime  # First peak/trough date
    date2: datetime  # Second peak/trough date
    neckline: float  # Support/resistance level
    target_price: float  # Projected target after breakout
    volume_confirmed: bool  # Volume pattern confirms
    signal: str  # "LONG", "SHORT", or "WAIT"
    reasons: List[str]


def detect_peaks(x: np.ndarray, mph: float = None, mpd: int = 1,
                 threshold: float = 0, edge: str = 'rising',
                 kpsh: bool = False, valley: bool = False) -> np.ndarray:
    """
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array
        Input data
    mph : float, optional
        Minimum peak height
    mpd : int, optional (default = 1)
        Minimum peak distance
    threshold : float, optional (default = 0)
        Minimum difference with neighbors
    edge : str, optional (default = 'rising')
        For flat peaks: 'rising', 'falling', 'both', or None
    kpsh : bool, optional (default = False)
        Keep peaks with same height even if closer than mpd
    valley : bool, optional (default = False)
        If True, detect valleys (local minima) instead of peaks

    Returns
    -------
    ind : 1D array
        Indices of the peaks in x
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)

    if valley:
        x = -x
        if mph is not None:
            mph = -mph

    # First derivative
    dx = x[1:] - x[:-1]

    # Handle NaN values
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    # Find indices of peaks
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Remove NaN-adjacent peaks
    if ind.size and indnan.size:
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]

    # Remove first/last if at boundary
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    # Filter by minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    # Filter by threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind-1], x[ind] - x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])

    # Filter by minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # Sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # Mark peaks too close for deletion
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        ind = np.sort(ind[~idel])

    return ind


class DoublePatternDetector:
    """
    Detects Double Top and Double Bottom patterns.

    Double Top:
    - Two peaks at approximately the same price level
    - Bearish signal when price breaks below neckline
    - First peak should have higher volume than second

    Double Bottom:
    - Two troughs at approximately the same price level
    - Bullish signal when price breaks above neckline
    - First trough should have higher volume than second
    """

    def __init__(self, tolerance_percent: float = 3.0, min_distance: int = 5,
                 lookback: int = 50):
        """
        Initialize detector.

        Args:
            tolerance_percent: Max % difference between peaks/troughs (default 3%)
            min_distance: Minimum candles between peaks/troughs (default 5)
            lookback: How many candles to analyze (default 50)
        """
        self.tolerance_percent = tolerance_percent
        self.min_distance = min_distance
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect double top or double bottom pattern.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
                Must have DatetimeIndex or 'date' column

        Returns:
            PatternResult with pattern details
        """
        if len(df) < self.lookback:
            return self._no_pattern("Insufficient data")

        # Use recent data
        df = df.tail(self.lookback).copy()

        # Get prices and volume
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values

        # Get dates
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.to_pydatetime()
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date']).values
        else:
            dates = [datetime.now()] * len(df)

        current_price = closes[-1]

        # Calculate threshold for peak detection
        threshold = 0.02 / max(highs) if max(highs) > 0 else 0

        # Detect peaks (for double top)
        peak_indices = detect_peaks(highs, threshold=threshold, mpd=self.min_distance)

        # Detect valleys (for double bottom)
        valley_indices = detect_peaks(lows, threshold=threshold, mpd=self.min_distance, valley=True)

        # Check for double top
        double_top = self._find_double_top(highs, volumes, dates, peak_indices, current_price)

        # Check for double bottom
        double_bottom = self._find_double_bottom(lows, volumes, dates, valley_indices, current_price)

        # Return the most recent/relevant pattern
        if double_top and double_bottom:
            # Both patterns found - return the more recent one
            if double_top.date2 > double_bottom.date2:
                return double_top
            else:
                return double_bottom
        elif double_top:
            return double_top
        elif double_bottom:
            return double_bottom
        else:
            return self._no_pattern("No double pattern detected")

    def _find_double_top(self, highs: np.ndarray, volumes: np.ndarray,
                         dates: np.ndarray, peak_indices: np.ndarray,
                         current_price: float) -> Optional[PatternResult]:
        """Find double top pattern in peaks."""
        if len(peak_indices) < 2:
            return None

        # Check pairs of peaks (most recent first)
        for i in range(len(peak_indices) - 1, 0, -1):
            idx2 = peak_indices[i]  # Second peak (more recent)

            for j in range(i - 1, -1, -1):
                idx1 = peak_indices[j]  # First peak (older)

                price1 = highs[idx1]
                price2 = highs[idx2]
                vol1 = volumes[idx1]
                vol2 = volumes[idx2]

                # Check if peaks are at similar levels
                diff_percent = abs(price1 - price2) / price1 * 100

                if diff_percent <= self.tolerance_percent:
                    # Found potential double top
                    # Neckline is the lowest point between the two peaks
                    neckline = min(highs[idx1:idx2+1].min(),
                                   min(highs[idx1:idx2+1]))

                    # Calculate target (pattern height projected down)
                    pattern_height = max(price1, price2) - neckline
                    target = neckline - pattern_height

                    # Volume confirmation: first peak should have higher volume
                    volume_confirmed = vol1 > vol2

                    # Determine signal
                    if current_price < neckline:
                        signal = "SHORT"  # Breakdown confirmed
                        confidence = 80 if volume_confirmed else 60
                    elif current_price >= min(price1, price2) * 0.98:
                        signal = "WAIT"  # Near resistance, wait for breakdown
                        confidence = 50 if volume_confirmed else 35
                    else:
                        signal = "WAIT"
                        confidence = 40

                    reasons = [
                        f"Double Top: peaks at {price1:.4f} and {price2:.4f}",
                        f"Difference: {diff_percent:.2f}%",
                        f"Neckline: {neckline:.4f}",
                        f"Volume confirmed: {volume_confirmed}"
                    ]

                    return PatternResult(
                        pattern_type="DOUBLE_TOP",
                        confidence=confidence,
                        price1=price1,
                        price2=price2,
                        date1=dates[idx1] if hasattr(dates[idx1], 'timestamp') else datetime.now(),
                        date2=dates[idx2] if hasattr(dates[idx2], 'timestamp') else datetime.now(),
                        neckline=neckline,
                        target_price=target,
                        volume_confirmed=volume_confirmed,
                        signal=signal,
                        reasons=reasons
                    )

        return None

    def _find_double_bottom(self, lows: np.ndarray, volumes: np.ndarray,
                            dates: np.ndarray, valley_indices: np.ndarray,
                            current_price: float) -> Optional[PatternResult]:
        """Find double bottom pattern in valleys."""
        if len(valley_indices) < 2:
            return None

        # Check pairs of valleys (most recent first)
        for i in range(len(valley_indices) - 1, 0, -1):
            idx2 = valley_indices[i]  # Second valley (more recent)

            for j in range(i - 1, -1, -1):
                idx1 = valley_indices[j]  # First valley (older)

                price1 = lows[idx1]
                price2 = lows[idx2]
                vol1 = volumes[idx1]
                vol2 = volumes[idx2]

                # Check if valleys are at similar levels
                diff_percent = abs(price1 - price2) / price1 * 100

                if diff_percent <= self.tolerance_percent:
                    # Found potential double bottom
                    # Neckline is the highest point between the two valleys
                    neckline = max(lows[idx1:idx2+1].max(),
                                   max(lows[idx1:idx2+1]))

                    # Calculate target (pattern height projected up)
                    pattern_height = neckline - min(price1, price2)
                    target = neckline + pattern_height

                    # Volume confirmation: first valley should have higher volume
                    volume_confirmed = vol1 > vol2

                    # Determine signal
                    if current_price > neckline:
                        signal = "LONG"  # Breakout confirmed
                        confidence = 80 if volume_confirmed else 60
                    elif current_price <= max(price1, price2) * 1.02:
                        signal = "WAIT"  # Near support, wait for breakout
                        confidence = 50 if volume_confirmed else 35
                    else:
                        signal = "WAIT"
                        confidence = 40

                    reasons = [
                        f"Double Bottom: troughs at {price1:.4f} and {price2:.4f}",
                        f"Difference: {diff_percent:.2f}%",
                        f"Neckline: {neckline:.4f}",
                        f"Volume confirmed: {volume_confirmed}"
                    ]

                    return PatternResult(
                        pattern_type="DOUBLE_BOTTOM",
                        confidence=confidence,
                        price1=price1,
                        price2=price2,
                        date1=dates[idx1] if hasattr(dates[idx1], 'timestamp') else datetime.now(),
                        date2=dates[idx2] if hasattr(dates[idx2], 'timestamp') else datetime.now(),
                        neckline=neckline,
                        target_price=target,
                        volume_confirmed=volume_confirmed,
                        signal=signal,
                        reasons=reasons
                    )

        return None

    def _no_pattern(self, reason: str) -> PatternResult:
        """Return a no-pattern result."""
        return PatternResult(
            pattern_type="NONE",
            confidence=0,
            price1=0,
            price2=0,
            date1=datetime.now(),
            date2=datetime.now(),
            neckline=0,
            target_price=0,
            volume_confirmed=False,
            signal="WAIT",
            reasons=[reason]
        )


# Singleton instance
double_pattern_detector = DoublePatternDetector()


def detect_double_pattern(df: pd.DataFrame,
                          tolerance_percent: float = 3.0,
                          min_distance: int = 5,
                          lookback: int = 50) -> PatternResult:
    """
    Convenience function to detect double top/bottom patterns.

    Args:
        df: OHLCV DataFrame
        tolerance_percent: Max % difference between peaks/troughs
        min_distance: Minimum candles between peaks/troughs
        lookback: How many candles to analyze

    Returns:
        PatternResult with pattern details
    """
    detector = DoublePatternDetector(
        tolerance_percent=tolerance_percent,
        min_distance=min_distance,
        lookback=lookback
    )
    return detector.detect(df)


# Test function
if __name__ == "__main__":
    # Create test data with a double bottom pattern
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='4h')

    # Simulate double bottom
    prices = []
    base = 100
    for i in range(60):
        if i < 15:
            base -= 0.5  # Downtrend
        elif i < 20:
            base += 0.3  # First recovery
        elif i < 30:
            base -= 0.3  # Second dip
        elif i < 35:
            base += 0.2  # Second recovery
        else:
            base += 0.4  # Breakout
        prices.append(base + np.random.uniform(-0.5, 0.5))

    df = pd.DataFrame({
        'open': prices,
        'high': [p + np.random.uniform(0, 1) for p in prices],
        'low': [p - np.random.uniform(0, 1) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000, 5000) for _ in prices]
    }, index=dates)

    result = detect_double_pattern(df)
    print(f"Pattern: {result.pattern_type}")
    print(f"Confidence: {result.confidence}%")
    print(f"Signal: {result.signal}")
    print(f"Reasons: {result.reasons}")
