# strategies.py - Trading Strategy Engine
"""
Trading strategy implementations:
- SMC/ICT based strategy
- Signal generation and validation
- Entry/Exit logic
"""

import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from enum import Enum

from config import config, TrendDirection, SignalType
from indicators import Indicators, SignalScore
from scanner import POI, ScanResult

logger = logging.getLogger(__name__)

class TradingSession(Enum):
    """Trading sessions."""
    ASIAN = "ASIAN"       # 00:00-08:00 UTC
    LONDON = "LONDON"     # 08:00-17:00 UTC
    NEW_YORK = "NEW_YORK" # 13:00-22:00 UTC
    OVERLAP = "OVERLAP"   # 13:00-17:00 UTC (London/NY)
    OFF_SESSION = "OFF"

@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    signal_type: SignalType
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    score: float
    confidence: float
    timeframe: str
    poi: Optional[POI] = None
    analysis: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None
    executed: bool = False

    @property
    def risk_reward(self) -> float:
        """Calculate risk/reward ratio."""
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        return reward / risk if risk > 0 else 0

@dataclass
class StrategyResult:
    """Result of strategy evaluation."""
    has_signal: bool
    signal: Optional[Signal] = None
    reason: str = ""
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)

class TradingStrategy:
    """
    Base trading strategy class.

    Implements SMC/ICT concepts for entry signals.
    """

    def __init__(self):
        self.cfg = config.strategy
        self.risk_cfg = config.risk

    def get_current_session(self) -> TradingSession:
        """Determine current trading session."""
        now = datetime.utcnow().time()

        # Define session times (UTC)
        asian_start = dt_time(0, 0)
        asian_end = dt_time(8, 0)
        london_start = dt_time(8, 0)
        london_end = dt_time(17, 0)
        ny_start = dt_time(13, 0)
        ny_end = dt_time(22, 0)

        # Check overlap first (most important)
        if dt_time(13, 0) <= now <= dt_time(17, 0):
            return TradingSession.OVERLAP

        if london_start <= now <= london_end:
            return TradingSession.LONDON

        if ny_start <= now <= ny_end:
            return TradingSession.NEW_YORK

        if asian_start <= now <= asian_end:
            return TradingSession.ASIAN

        return TradingSession.OFF_SESSION

    def is_session_allowed(self) -> Tuple[bool, str]:
        """Check if current session is allowed for trading."""
        if not self.cfg.use_session_filter:
            return True, "Session filter disabled"

        session = self.get_current_session()

        if session.value in self.cfg.allowed_sessions:
            return True, f"In {session.value} session"

        return False, f"Outside allowed sessions ({session.value})"

    def is_trading_day(self) -> Tuple[bool, str]:
        """Check if current day is a trading day."""
        if not self.cfg.trade_weekends:
            day = datetime.utcnow().weekday()
            if day >= 5:  # Saturday = 5, Sunday = 6
                return False, "Weekend trading disabled"

        return True, "Trading day"

    def evaluate_htf_bias(self, scan_result: ScanResult) -> Tuple[bool, str]:
        """Evaluate higher timeframe bias."""
        if not self.cfg.require_trend_alignment:
            return True, "Trend alignment not required"

        # Allow NEUTRAL trends for paper trading - just note it
        if scan_result.trend == TrendDirection.NEUTRAL:
            return True, "Neutral HTF trend (allowed for paper trading)"

        return True, f"HTF bias: {scan_result.htf_bias}"

    def evaluate_poi(self, poi: POI, current_price: float) -> Tuple[bool, str]:
        """Evaluate Point of Interest validity."""
        checks = []

        # Check if price is near POI
        zone_size = poi.top - poi.bottom
        distance_to_zone = min(
            abs(current_price - poi.top),
            abs(current_price - poi.bottom)
        )

        if poi.bottom <= current_price <= poi.top:
            checks.append("Price in zone")
        elif distance_to_zone <= zone_size * 0.5:
            checks.append("Price near zone")
        else:
            return False, "Price too far from POI"

        # Check POI type requirements
        if self.cfg.require_fvg and poi.poi_type == "FVG":
            checks.append("FVG present")
        elif self.cfg.require_ob and poi.poi_type == "OB":
            checks.append("Order Block present")

        return True, ", ".join(checks)

    def evaluate_confirmation(self, indicators: Indicators,
                               direction: str) -> Tuple[bool, str, SignalScore]:
        """Evaluate LTF confirmation signals."""
        score = indicators.calculate_signal_score(direction)

        if score.total_score < self.cfg.min_score:
            return False, f"Score too low: {score.total_score:.1f}", score

        checks = []

        # Check confluences
        if score.trend_score > 15:
            checks.append("Trend aligned")

        if score.momentum_score > 15:
            checks.append("Momentum confirmed")

        if self.cfg.require_volume_confirmation and score.volume_score > 15:
            checks.append("Volume confirmed")

        if self.cfg.require_momentum_confirmation:
            if score.momentum_score < 10:
                return False, "Momentum not confirmed", score

        # Need at least 1 confluence for paper trading (was 2)
        if score.confluence_count < 1:
            return False, f"Insufficient confluences: {score.confluence_count}", score

        return True, f"Score: {score.total_score:.1f} ({', '.join(checks)})", score

    def calculate_entry_exit(self, poi: POI, current_price: float,
                             atr: float) -> Tuple[float, float, float]:
        """
        Calculate entry, stop loss, and take profit levels.

        Returns:
            Tuple of (entry_price, stop_loss, take_profit)
        """
        if poi.direction == "BULLISH":
            # Entry at POI top (when tapped and confirmed)
            entry = poi.top

            # Stop loss below POI with ATR buffer
            if self.risk_cfg.use_atr_stops:
                sl = poi.bottom - (atr * self.risk_cfg.atr_sl_multiplier)
            else:
                sl = poi.bottom

            # Ensure SL is not too far
            max_sl_distance = entry * self.risk_cfg.max_sl_percent
            if entry - sl > max_sl_distance:
                sl = entry - max_sl_distance

            # Take profit based on R:R
            risk = entry - sl
            tp = entry + (risk * self.risk_cfg.default_risk_reward)

        else:  # BEARISH
            entry = poi.bottom
            if self.risk_cfg.use_atr_stops:
                sl = poi.top + (atr * self.risk_cfg.atr_sl_multiplier)
            else:
                sl = poi.top

            max_sl_distance = entry * self.risk_cfg.max_sl_percent
            if sl - entry > max_sl_distance:
                sl = entry + max_sl_distance

            risk = sl - entry
            tp = entry - (risk * self.risk_cfg.default_risk_reward)

        return entry, sl, tp

    def generate_signal(self, scan_result: ScanResult, poi: POI,
                         indicators: Indicators,
                         current_price: float) -> StrategyResult:
        """
        Generate trading signal based on all criteria.

        Args:
            scan_result: HTF scan result
            poi: Point of Interest
            indicators: LTF indicators
            current_price: Current price

        Returns:
            StrategyResult with signal if valid
        """
        checks_passed = []
        checks_failed = []

        # 1. Check trading day
        day_ok, day_msg = self.is_trading_day()
        if not day_ok:
            checks_failed.append(day_msg)
            return StrategyResult(False, reason=day_msg, checks_failed=checks_failed)
        checks_passed.append(day_msg)

        # 2. Check session
        session_ok, session_msg = self.is_session_allowed()
        if not session_ok:
            checks_failed.append(session_msg)
            return StrategyResult(False, reason=session_msg,
                                  checks_passed=checks_passed, checks_failed=checks_failed)
        checks_passed.append(session_msg)

        # 3. Check HTF bias
        htf_ok, htf_msg = self.evaluate_htf_bias(scan_result)
        if not htf_ok:
            checks_failed.append(htf_msg)
            return StrategyResult(False, reason=htf_msg,
                                  checks_passed=checks_passed, checks_failed=checks_failed)
        checks_passed.append(htf_msg)

        # 4. Check POI
        poi_ok, poi_msg = self.evaluate_poi(poi, current_price)
        if not poi_ok:
            checks_failed.append(poi_msg)
            return StrategyResult(False, reason=poi_msg,
                                  checks_passed=checks_passed, checks_failed=checks_failed)
        checks_passed.append(poi_msg)

        # 5. Check LTF confirmation
        direction = "LONG" if poi.direction == "BULLISH" else "SHORT"
        conf_ok, conf_msg, score = self.evaluate_confirmation(indicators, direction)
        if not conf_ok:
            checks_failed.append(conf_msg)
            return StrategyResult(False, reason=conf_msg,
                                  checks_passed=checks_passed, checks_failed=checks_failed)
        checks_passed.append(conf_msg)

        # All checks passed - generate signal
        atr = indicators.atr().iloc[-1]
        entry, sl, tp = self.calculate_entry_exit(poi, current_price, atr)

        # Calculate confidence
        confidence = min(100, score.total_score + (score.confluence_count * 5))

        signal = Signal(
            symbol=scan_result.symbol,
            signal_type=SignalType.LONG if direction == "LONG" else SignalType.SHORT,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            score=score.total_score,
            confidence=confidence,
            timeframe=config.scanner.ltf_timeframe.value,
            poi=poi,
            analysis={
                "trend_score": score.trend_score,
                "momentum_score": score.momentum_score,
                "volume_score": score.volume_score,
                "structure_score": score.structure_score,
                "confluences": score.confluence_count,
                "atr": atr,
                "htf_bias": scan_result.htf_bias,
                "session": self.get_current_session().value
            }
        )

        logger.info(
            f"SIGNAL GENERATED: {direction} {scan_result.symbol} | "
            f"Entry: {entry:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | "
            f"Score: {score.total_score:.1f} | Confidence: {confidence:.0f}%"
        )

        return StrategyResult(
            has_signal=True,
            signal=signal,
            reason="All criteria met",
            checks_passed=checks_passed,
            checks_failed=checks_failed
        )


class SMCStrategy(TradingStrategy):
    """
    Smart Money Concepts strategy.

    Focuses on:
    - Fair Value Gaps (FVG)
    - Order Blocks (OB)
    - Break of Structure (BOS)
    - Change of Character (CHoCH)
    - Liquidity sweeps
    """

    def __init__(self):
        super().__init__()
        self.name = "SMC_Strategy"

    def evaluate_structure_shift(self, indicators: Indicators,
                                 direction: str) -> Tuple[bool, str]:
        """Check for market structure shift."""
        structure = indicators.analyze_market_structure()

        if direction == "LONG":
            if structure.choch_detected:
                for sb in structure.structure_breaks:
                    if sb.get("direction") == "BULLISH":
                        return True, "Bullish CHoCH confirmed"

            if structure.bos_detected:
                return True, "Bullish BOS detected"

        else:  # SHORT
            if structure.choch_detected:
                for sb in structure.structure_breaks:
                    if sb.get("direction") == "BEARISH":
                        return True, "Bearish CHoCH confirmed"

            if structure.bos_detected:
                return True, "Bearish BOS detected"

        return False, "No structure shift"

    def generate_signal(self, scan_result: ScanResult, poi: POI,
                         indicators: Indicators,
                         current_price: float) -> StrategyResult:
        """Generate SMC-based signal."""
        # Run base checks first
        base_result = super().generate_signal(
            scan_result, poi, indicators, current_price
        )

        if not base_result.has_signal:
            return base_result

        # Additional SMC checks
        direction = "LONG" if poi.direction == "BULLISH" else "SHORT"

        # Check for structure shift if required
        if self.cfg.require_choch or self.cfg.require_bos:
            shift_ok, shift_msg = self.evaluate_structure_shift(indicators, direction)

            if self.cfg.require_choch and not shift_ok:
                return StrategyResult(
                    False,
                    reason="CHoCH required but not detected",
                    checks_passed=base_result.checks_passed,
                    checks_failed=base_result.checks_failed + [shift_msg]
                )

            base_result.checks_passed.append(shift_msg)

        return base_result


# Strategy factory
def get_strategy(name: str = "smc") -> TradingStrategy:
    """Get strategy by name."""
    strategies = {
        "smc": SMCStrategy,
        "base": TradingStrategy
    }

    strategy_class = strategies.get(name.lower(), SMCStrategy)
    return strategy_class()


# Default strategy instance
strategy = SMCStrategy()
