# volume_momentum_strategy.py - Volume-Momentum Breakout Strategy
"""
A statistically-backed trading strategy based on:
1. Volume analysis - Only trade when volume confirms
2. Volatility squeeze - Low volatility precedes big moves
3. Momentum confirmation - Trade in direction of momentum
4. Liquidity sweeps - Enter after stop hunts for better entries

This strategy aims for 50%+ win rate with 2:1+ R:R ratio.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    SQUEEZE = "SQUEEZE"  # Low volatility, breakout imminent


class SignalStrength(Enum):
    """Signal strength classification."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class VolumeProfile:
    """Volume analysis results."""
    current_volume: float
    avg_volume_20: float
    avg_volume_50: float
    volume_ratio: float  # current / avg
    volume_trend: str  # INCREASING, DECREASING, STABLE
    is_climax: bool  # Extreme volume spike
    is_dry_up: bool  # Very low volume
    obv_trend: str  # On-balance volume trend
    vwap: float
    price_vs_vwap: str  # ABOVE, BELOW, AT


@dataclass
class VolatilityState:
    """Volatility analysis results."""
    atr: float
    atr_percent: float  # ATR as % of price
    bb_width: float  # Bollinger Band width
    bb_squeeze: bool  # Bands are squeezed
    squeeze_duration: int  # How many candles in squeeze
    historical_volatility: float
    regime: MarketRegime


@dataclass
class MomentumState:
    """Momentum analysis results."""
    rsi: float
    rsi_divergence: Optional[str]  # BULLISH, BEARISH, None
    macd_histogram: float
    macd_crossover: Optional[str]  # BULLISH, BEARISH, None
    price_momentum: float  # Rate of change
    ema_alignment: str  # BULLISH, BEARISH, MIXED


@dataclass
class LiquiditySweep:
    """Liquidity sweep detection."""
    detected: bool
    direction: str  # BULLISH (swept lows), BEARISH (swept highs)
    sweep_price: float
    recovery_price: float
    strength: SignalStrength


@dataclass
class TradeSetup:
    """Complete trade setup."""
    symbol: str
    direction: str  # LONG, SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float  # First target (1:1)
    take_profit_2: float  # Second target (2:1)
    take_profit_3: float  # Third target (3:1)
    position_size_percent: float
    confidence: float  # 0-100
    reasons: List[str]
    regime: MarketRegime
    volume_confirmed: bool
    timestamp: datetime = field(default_factory=datetime.now)


class VolumeAnalyzer:
    """Advanced volume analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_volume_indicators()

    def _calculate_volume_indicators(self):
        """Calculate all volume-based indicators."""
        df = self.df

        # Basic volume metrics
        df['vol_sma_20'] = df['volume'].rolling(20).mean()
        df['vol_sma_50'] = df['volume'].rolling(50).mean()
        df['vol_ratio'] = df['volume'] / df['vol_sma_20']

        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()

        # Volume-Weighted Average Price
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Volume momentum
        df['vol_momentum'] = df['volume'].pct_change(5)

        # Accumulation/Distribution
        df['ad'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
        df['ad'] = df['ad'].fillna(0).cumsum()

        self.df = df

    def analyze(self) -> VolumeProfile:
        """Get current volume profile."""
        df = self.df
        current = df.iloc[-1]
        prev_5 = df.iloc[-6:-1]

        # Volume trend
        vol_change = df['volume'].iloc[-5:].mean() / df['volume'].iloc[-10:-5].mean()
        if vol_change > 1.2:
            volume_trend = "INCREASING"
        elif vol_change < 0.8:
            volume_trend = "DECREASING"
        else:
            volume_trend = "STABLE"

        # Climax detection (extreme volume)
        vol_std = df['volume'].rolling(50).std().iloc[-1]
        vol_mean = df['vol_sma_50'].iloc[-1]
        is_climax = current['volume'] > vol_mean + (2.5 * vol_std)

        # Dry up detection (very low volume)
        is_dry_up = current['volume'] < vol_mean * 0.5

        # OBV trend
        obv_trend = "BULLISH" if current['obv'] > current['obv_sma'] else "BEARISH"

        # Price vs VWAP
        if current['close'] > current['vwap'] * 1.005:
            price_vs_vwap = "ABOVE"
        elif current['close'] < current['vwap'] * 0.995:
            price_vs_vwap = "BELOW"
        else:
            price_vs_vwap = "AT"

        return VolumeProfile(
            current_volume=current['volume'],
            avg_volume_20=current['vol_sma_20'],
            avg_volume_50=current['vol_sma_50'],
            volume_ratio=current['vol_ratio'],
            volume_trend=volume_trend,
            is_climax=is_climax,
            is_dry_up=is_dry_up,
            obv_trend=obv_trend,
            vwap=current['vwap'],
            price_vs_vwap=price_vs_vwap
        )

    def is_volume_confirmed(self, direction: str) -> Tuple[bool, str]:
        """
        Check if volume confirms the direction.

        For LONG: Want increasing volume, OBV bullish, price above VWAP
        For SHORT: Want increasing volume, OBV bearish, price below VWAP
        """
        profile = self.analyze()
        score = 0
        reasons = []

        # Volume ratio check
        if profile.volume_ratio > 1.2:
            score += 2
            reasons.append(f"High volume ({profile.volume_ratio:.1f}x avg)")
        elif profile.volume_ratio > 0.8:
            score += 1
            reasons.append("Normal volume")
        else:
            reasons.append("Low volume warning")

        # Volume trend
        if profile.volume_trend == "INCREASING":
            score += 1
            reasons.append("Volume increasing")

        # OBV confirmation
        if direction == "LONG" and profile.obv_trend == "BULLISH":
            score += 2
            reasons.append("OBV bullish")
        elif direction == "SHORT" and profile.obv_trend == "BEARISH":
            score += 2
            reasons.append("OBV bearish")

        # VWAP confirmation
        if direction == "LONG" and profile.price_vs_vwap == "ABOVE":
            score += 1
            reasons.append("Price above VWAP")
        elif direction == "SHORT" and profile.price_vs_vwap == "BELOW":
            score += 1
            reasons.append("Price below VWAP")

        # Climax warning (potential reversal)
        if profile.is_climax:
            score -= 2
            reasons.append("⚠️ Volume climax - potential reversal")

        confirmed = score >= 3
        return confirmed, ", ".join(reasons)


class VolatilityAnalyzer:
    """Volatility and squeeze detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_volatility_indicators()

    def _calculate_volatility_indicators(self):
        """Calculate volatility indicators."""
        df = self.df

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_percent'] = df['atr'] / df['close'] * 100

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

        # Keltner Channels (for squeeze detection)
        df['kc_mid'] = df['close'].ewm(span=20).mean()
        df['kc_upper'] = df['kc_mid'] + (1.5 * df['atr'])
        df['kc_lower'] = df['kc_mid'] - (1.5 * df['atr'])

        # Squeeze detection: BB inside KC
        df['squeeze'] = (df['bb_lower'] > df['kc_lower']) & (df['bb_upper'] < df['kc_upper'])

        # Historical volatility
        df['returns'] = df['close'].pct_change()
        df['hist_vol'] = df['returns'].rolling(20).std() * np.sqrt(252) * 100

        self.df = df

    def analyze(self) -> VolatilityState:
        """Get current volatility state."""
        df = self.df
        current = df.iloc[-1]

        # Count squeeze duration
        squeeze_count = 0
        for i in range(len(df) - 1, -1, -1):
            if df.iloc[i]['squeeze']:
                squeeze_count += 1
            else:
                break

        # Determine regime
        atr_avg = df['atr_percent'].rolling(50).mean().iloc[-1]
        if current['squeeze']:
            regime = MarketRegime.SQUEEZE
        elif current['atr_percent'] > atr_avg * 1.5:
            regime = MarketRegime.VOLATILE
        else:
            # Check trend
            ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
            ema_50 = df['close'].ewm(span=50).mean().iloc[-1]
            if ema_20 > ema_50 * 1.02:
                regime = MarketRegime.TRENDING_UP
            elif ema_20 < ema_50 * 0.98:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING

        return VolatilityState(
            atr=current['atr'],
            atr_percent=current['atr_percent'],
            bb_width=current['bb_width'],
            bb_squeeze=current['squeeze'],
            squeeze_duration=squeeze_count,
            historical_volatility=current['hist_vol'],
            regime=regime
        )

    def is_squeeze_breakout(self) -> Tuple[bool, str, str]:
        """
        Detect squeeze breakout.

        Returns:
            (is_breakout, direction, reason)
        """
        df = self.df
        vol_state = self.analyze()

        # Need recent squeeze
        if vol_state.squeeze_duration > 0:
            return False, "", "Still in squeeze"

        # Check if just exited squeeze (within last 3 candles)
        recent_squeeze = df['squeeze'].iloc[-4:-1].any()
        if not recent_squeeze:
            return False, "", "No recent squeeze"

        # Determine breakout direction
        current_close = df['close'].iloc[-1]
        bb_mid = df['bb_mid'].iloc[-1]
        bb_upper = df['bb_upper'].iloc[-1]
        bb_lower = df['bb_lower'].iloc[-1]

        if current_close > bb_upper:
            return True, "LONG", f"Squeeze breakout UP (price {current_close:.4f} > BB upper {bb_upper:.4f})"
        elif current_close < bb_lower:
            return True, "SHORT", f"Squeeze breakout DOWN (price {current_close:.4f} < BB lower {bb_lower:.4f})"

        return False, "", "No clear breakout direction"


class MomentumAnalyzer:
    """Momentum analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_momentum_indicators()

    def _calculate_momentum_indicators(self):
        """Calculate momentum indicators."""
        df = self.df

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # EMAs for trend
        df['ema_9'] = df['close'].ewm(span=9).mean()
        df['ema_21'] = df['close'].ewm(span=21).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()

        # Rate of Change
        df['roc'] = df['close'].pct_change(10) * 100

        self.df = df

    def analyze(self) -> MomentumState:
        """Get current momentum state."""
        df = self.df
        current = df.iloc[-1]

        # RSI divergence detection
        rsi_divergence = self._detect_rsi_divergence()

        # MACD crossover
        macd_crossover = None
        if df['macd_hist'].iloc[-1] > 0 and df['macd_hist'].iloc[-2] <= 0:
            macd_crossover = "BULLISH"
        elif df['macd_hist'].iloc[-1] < 0 and df['macd_hist'].iloc[-2] >= 0:
            macd_crossover = "BEARISH"

        # EMA alignment
        if current['ema_9'] > current['ema_21'] > current['ema_50']:
            ema_alignment = "BULLISH"
        elif current['ema_9'] < current['ema_21'] < current['ema_50']:
            ema_alignment = "BEARISH"
        else:
            ema_alignment = "MIXED"

        return MomentumState(
            rsi=current['rsi'],
            rsi_divergence=rsi_divergence,
            macd_histogram=current['macd_hist'],
            macd_crossover=macd_crossover,
            price_momentum=current['roc'],
            ema_alignment=ema_alignment
        )

    def _detect_rsi_divergence(self) -> Optional[str]:
        """Detect RSI divergence."""
        df = self.df
        lookback = 20

        # Find recent price highs/lows
        recent = df.iloc[-lookback:]

        # Bullish divergence: price makes lower low, RSI makes higher low
        price_lows = recent['low'].rolling(5).min()
        rsi_at_lows = recent['rsi'].rolling(5).min()

        if len(price_lows) >= 10:
            if price_lows.iloc[-1] < price_lows.iloc[-10] and rsi_at_lows.iloc[-1] > rsi_at_lows.iloc[-10]:
                return "BULLISH"

        # Bearish divergence: price makes higher high, RSI makes lower high
        price_highs = recent['high'].rolling(5).max()
        rsi_at_highs = recent['rsi'].rolling(5).max()

        if len(price_highs) >= 10:
            if price_highs.iloc[-1] > price_highs.iloc[-10] and rsi_at_highs.iloc[-1] < rsi_at_highs.iloc[-10]:
                return "BEARISH"

        return None

    def is_momentum_confirmed(self, direction: str) -> Tuple[bool, str]:
        """Check if momentum confirms direction."""
        state = self.analyze()
        score = 0
        reasons = []

        if direction == "LONG":
            # RSI not overbought
            if state.rsi < 70:
                score += 1
                if state.rsi > 50:
                    score += 1
                    reasons.append(f"RSI bullish ({state.rsi:.0f})")
            else:
                reasons.append(f"⚠️ RSI overbought ({state.rsi:.0f})")

            # MACD
            if state.macd_histogram > 0:
                score += 1
                reasons.append("MACD positive")
            if state.macd_crossover == "BULLISH":
                score += 2
                reasons.append("MACD bullish crossover")

            # EMA alignment
            if state.ema_alignment == "BULLISH":
                score += 2
                reasons.append("EMAs aligned bullish")

            # RSI divergence
            if state.rsi_divergence == "BULLISH":
                score += 2
                reasons.append("Bullish RSI divergence")

        else:  # SHORT
            if state.rsi > 30:
                score += 1
                if state.rsi < 50:
                    score += 1
                    reasons.append(f"RSI bearish ({state.rsi:.0f})")
            else:
                reasons.append(f"⚠️ RSI oversold ({state.rsi:.0f})")

            if state.macd_histogram < 0:
                score += 1
                reasons.append("MACD negative")
            if state.macd_crossover == "BEARISH":
                score += 2
                reasons.append("MACD bearish crossover")

            if state.ema_alignment == "BEARISH":
                score += 2
                reasons.append("EMAs aligned bearish")

            if state.rsi_divergence == "BEARISH":
                score += 2
                reasons.append("Bearish RSI divergence")

        confirmed = score >= 3
        return confirmed, ", ".join(reasons) if reasons else "No momentum signals"


class LiquiditySweepDetector:
    """Detect liquidity sweeps (stop hunts)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._find_swing_points()

    def _find_swing_points(self):
        """Identify swing highs and lows."""
        df = self.df
        lookback = 5

        df['swing_high'] = df['high'][(df['high'] == df['high'].rolling(lookback * 2 + 1, center=True).max())]
        df['swing_low'] = df['low'][(df['low'] == df['low'].rolling(lookback * 2 + 1, center=True).min())]

        self.df = df

    def detect(self) -> LiquiditySweep:
        """
        Detect if a liquidity sweep just occurred.

        Liquidity sweep = price briefly breaks a swing point then reverses.
        This is a high-probability entry signal.
        """
        df = self.df
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev_2 = df.iloc[-3]

        # Find recent swing lows (for bullish sweep)
        recent_swing_lows = df['swing_low'].dropna().tail(3)
        recent_swing_highs = df['swing_high'].dropna().tail(3)

        # Bullish sweep: wick below swing low, close above
        for swing_low in recent_swing_lows:
            # Check if we swept the low
            if current['low'] < swing_low and current['close'] > swing_low:
                # Strong reversal candle
                if current['close'] > current['open']:  # Bullish candle
                    body = abs(current['close'] - current['open'])
                    lower_wick = min(current['open'], current['close']) - current['low']

                    if lower_wick > body * 0.5:  # Significant rejection
                        strength = SignalStrength.STRONG if lower_wick > body else SignalStrength.MODERATE
                        return LiquiditySweep(
                            detected=True,
                            direction="BULLISH",
                            sweep_price=current['low'],
                            recovery_price=current['close'],
                            strength=strength
                        )

        # Bearish sweep: wick above swing high, close below
        for swing_high in recent_swing_highs:
            if current['high'] > swing_high and current['close'] < swing_high:
                if current['close'] < current['open']:  # Bearish candle
                    body = abs(current['close'] - current['open'])
                    upper_wick = current['high'] - max(current['open'], current['close'])

                    if upper_wick > body * 0.5:
                        strength = SignalStrength.STRONG if upper_wick > body else SignalStrength.MODERATE
                        return LiquiditySweep(
                            detected=True,
                            direction="BEARISH",
                            sweep_price=current['high'],
                            recovery_price=current['close'],
                            strength=strength
                        )

        return LiquiditySweep(
            detected=False,
            direction="",
            sweep_price=0,
            recovery_price=0,
            strength=SignalStrength.WEAK
        )


class VolumeMomentumStrategy:
    """
    Main strategy class combining all analysis.

    Entry Criteria (need 3+ confirmations):
    1. Volume confirmation (required)
    2. Momentum aligned
    3. Volatility regime favorable
    4. Liquidity sweep (bonus)
    5. Higher timeframe trend alignment

    Exit Strategy:
    - TP1 at 1:1 R:R (close 50%)
    - TP2 at 2:1 R:R (close 30%)
    - TP3 at 3:1 R:R (close remaining 20%)
    - Trail stop after TP1
    """

    def __init__(self):
        self.min_volume_ratio = 0.8  # Minimum volume vs average
        self.min_confidence = 60  # Minimum confidence score
        self.max_atr_percent = 5.0  # Max volatility for entry

    def analyze(self, df: pd.DataFrame, symbol: str) -> Optional[TradeSetup]:
        """
        Analyze a symbol and generate trade setup if valid.

        Args:
            df: OHLCV DataFrame
            symbol: Trading pair

        Returns:
            TradeSetup if valid signal, None otherwise
        """
        if len(df) < 200:
            logger.debug(f"{symbol}: Insufficient data ({len(df)} candles)")
            return None

        # Initialize analyzers
        vol_analyzer = VolumeAnalyzer(df)
        volatility_analyzer = VolatilityAnalyzer(df)
        momentum_analyzer = MomentumAnalyzer(df)
        sweep_detector = LiquiditySweepDetector(df)

        # Get current state
        vol_profile = vol_analyzer.analyze()
        vol_state = volatility_analyzer.analyze()
        mom_state = momentum_analyzer.analyze()
        sweep = sweep_detector.detect()

        current_price = df['close'].iloc[-1]
        atr = vol_state.atr

        # Determine direction candidates
        directions_to_check = []

        # 1. Check for squeeze breakout
        is_breakout, breakout_dir, breakout_reason = volatility_analyzer.is_squeeze_breakout()
        if is_breakout:
            directions_to_check.append((breakout_dir, "SQUEEZE_BREAKOUT", breakout_reason))

        # 2. Check for liquidity sweep
        if sweep.detected:
            directions_to_check.append((
                "LONG" if sweep.direction == "BULLISH" else "SHORT",
                "LIQUIDITY_SWEEP",
                f"Swept {sweep.sweep_price:.6f}, recovered to {sweep.recovery_price:.6f}"
            ))

        # 3. Check momentum-based direction
        if mom_state.ema_alignment == "BULLISH" and mom_state.rsi > 40 and mom_state.rsi < 70:
            directions_to_check.append(("LONG", "MOMENTUM", "EMA alignment bullish"))
        elif mom_state.ema_alignment == "BEARISH" and mom_state.rsi < 60 and mom_state.rsi > 30:
            directions_to_check.append(("SHORT", "MOMENTUM", "EMA alignment bearish"))

        # Evaluate each direction
        best_setup = None
        best_confidence = 0

        for direction, signal_type, initial_reason in directions_to_check:
            confidence = 0
            reasons = [f"{signal_type}: {initial_reason}"]

            # REQUIRED: Volume confirmation
            vol_confirmed, vol_reason = vol_analyzer.is_volume_confirmed(direction)
            if not vol_confirmed:
                logger.debug(f"{symbol} {direction}: Volume not confirmed - {vol_reason}")
                continue
            confidence += 25
            reasons.append(f"Volume: {vol_reason}")

            # Momentum confirmation
            mom_confirmed, mom_reason = momentum_analyzer.is_momentum_confirmed(direction)
            if mom_confirmed:
                confidence += 25
                reasons.append(f"Momentum: {mom_reason}")
            else:
                confidence += 5  # Small penalty but not disqualifying

            # Volatility check
            if vol_state.regime == MarketRegime.SQUEEZE:
                confidence += 15
                reasons.append("Volatility squeeze (high probability)")
            elif vol_state.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                if (direction == "LONG" and vol_state.regime == MarketRegime.TRENDING_UP) or \
                   (direction == "SHORT" and vol_state.regime == MarketRegime.TRENDING_DOWN):
                    confidence += 20
                    reasons.append(f"Aligned with {vol_state.regime.value}")
            elif vol_state.atr_percent > self.max_atr_percent:
                confidence -= 10
                reasons.append(f"⚠️ High volatility ({vol_state.atr_percent:.1f}%)")

            # Liquidity sweep bonus
            if sweep.detected and sweep.direction == ("BULLISH" if direction == "LONG" else "BEARISH"):
                confidence += 20
                reasons.append(f"Liquidity sweep ({sweep.strength.name})")

            # RSI extreme check
            if direction == "LONG" and mom_state.rsi < 35:
                confidence += 10
                reasons.append("RSI oversold bounce")
            elif direction == "SHORT" and mom_state.rsi > 65:
                confidence += 10
                reasons.append("RSI overbought rejection")

            # Divergence bonus
            if mom_state.rsi_divergence:
                if (direction == "LONG" and mom_state.rsi_divergence == "BULLISH") or \
                   (direction == "SHORT" and mom_state.rsi_divergence == "BEARISH"):
                    confidence += 15
                    reasons.append(f"{mom_state.rsi_divergence} divergence")

            if confidence > best_confidence and confidence >= self.min_confidence:
                best_confidence = confidence

                # Calculate entry, SL, TP
                entry = current_price

                # Dynamic SL based on ATR (1.5x ATR)
                sl_distance = atr * 1.5

                # Cap SL at 2% of entry
                max_sl_distance = entry * 0.015  # 1.5% max SL (leaving room for costs)
                sl_distance = min(sl_distance, max_sl_distance)

                if direction == "LONG":
                    sl = entry - sl_distance
                    tp1 = entry + sl_distance  # 1:1
                    tp2 = entry + (sl_distance * 2)  # 2:1
                    tp3 = entry + (sl_distance * 3)  # 3:1
                else:
                    sl = entry + sl_distance
                    tp1 = entry - sl_distance
                    tp2 = entry - (sl_distance * 2)
                    tp3 = entry - (sl_distance * 3)

                best_setup = TradeSetup(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    take_profit_3=tp3,
                    position_size_percent=self._calculate_position_size(confidence),
                    confidence=confidence,
                    reasons=reasons,
                    regime=vol_state.regime,
                    volume_confirmed=vol_confirmed
                )

        if best_setup:
            logger.info(
                f"SETUP FOUND: {best_setup.symbol} {best_setup.direction} | "
                f"Confidence: {best_setup.confidence}% | "
                f"Entry: {best_setup.entry_price:.6f} | "
                f"SL: {best_setup.stop_loss:.6f} | "
                f"Reasons: {', '.join(best_setup.reasons[:3])}"
            )

        return best_setup

    def _calculate_position_size(self, confidence: float) -> float:
        """
        Calculate position size based on confidence.

        Higher confidence = larger position (up to limits).
        """
        # Base: 1% of portfolio per trade
        base_size = 0.01

        if confidence >= 80:
            return base_size * 1.5  # 1.5% for high confidence
        elif confidence >= 70:
            return base_size * 1.25  # 1.25%
        else:
            return base_size  # 1%


# Singleton instance
volume_momentum_strategy = VolumeMomentumStrategy()
