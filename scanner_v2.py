# scanner_v2.py - Volume-Momentum Scanner
"""
New scanner using Volume-Momentum strategy.
Scans multiple symbols for high-probability setups.
"""

import time
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config, TimeFrame
from kucoin_client import kucoin_client
from volume_momentum_strategy import (
    VolumeMomentumStrategy,
    TradeSetup,
    VolumeAnalyzer,
    VolatilityAnalyzer,
    MomentumAnalyzer,
    MarketRegime
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolAnalysis:
    """Analysis result for a symbol."""
    symbol: str
    timestamp: datetime
    current_price: float
    volume_ratio: float
    regime: MarketRegime
    has_setup: bool
    setup: Optional[TradeSetup] = None
    reasons: List[str] = field(default_factory=list)


class VolumeMomentumScanner:
    """
    Scanner for Volume-Momentum strategy.

    Features:
    - Parallel symbol scanning
    - Prioritizes high-volume symbols
    - Filters by volatility regime
    - Ranks setups by confidence
    """

    def __init__(self):
        self.client = kucoin_client
        self.strategy = VolumeMomentumStrategy()
        self.cfg = config.scanner
        self.symbols: List[str] = []
        self.analyses: Dict[str, SymbolAnalysis] = {}
        self.active_setups: Dict[str, TradeSetup] = {}
        self._last_scan: float = 0

    def initialize(self):
        """Initialize scanner with symbols."""
        if self.cfg.symbols:
            self.symbols = self.cfg.symbols
        elif self.cfg.auto_discover:
            self.symbols = self.client.get_top_volume_pairs(
                count=self.cfg.top_pairs_count
            )
            logger.info(f"Auto-discovered {len(self.symbols)} high-volume pairs")
        else:
            self.symbols = self.client.get_usdt_pairs()

        logger.info(f"Scanner initialized with {len(self.symbols)} symbols")
        return self.symbols

    def scan_symbol(self, symbol: str) -> Optional[SymbolAnalysis]:
        """
        Scan a single symbol for setups.

        Args:
            symbol: Trading pair

        Returns:
            SymbolAnalysis or None
        """
        try:
            # Get candles (4H for main analysis)
            df = self.client.get_candles(
                symbol=symbol,
                timeframe=self.cfg.htf_timeframe,
                limit=250  # Need more data for proper analysis
            )

            if df.empty or len(df) < 100:
                return None

            current_price = df['close'].iloc[-1]

            # Quick volume check first (filter out low volume)
            vol_analyzer = VolumeAnalyzer(df)
            vol_profile = vol_analyzer.analyze()

            if vol_profile.volume_ratio < 0.5:
                # Skip very low volume symbols
                return SymbolAnalysis(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    current_price=current_price,
                    volume_ratio=vol_profile.volume_ratio,
                    regime=MarketRegime.RANGING,
                    has_setup=False,
                    reasons=["Low volume - skipped"]
                )

            # Get volatility regime
            volatility_analyzer = VolatilityAnalyzer(df)
            vol_state = volatility_analyzer.analyze()

            # Run full strategy analysis
            setup = self.strategy.analyze(df, symbol)

            analysis = SymbolAnalysis(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                volume_ratio=vol_profile.volume_ratio,
                regime=vol_state.regime,
                has_setup=setup is not None,
                setup=setup,
                reasons=setup.reasons if setup else []
            )

            return analysis

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None

    def run_scan(self) -> Dict[str, TradeSetup]:
        """
        Run scan on all symbols.

        Returns:
            Dictionary of symbol to TradeSetup for valid setups
        """
        setups = {}
        start_time = time.time()

        logger.info(f"Starting scan on {len(self.symbols)} symbols...")

        # Parallel scanning (increased workers for 300 coins)
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self.scan_symbol, symbol): symbol
                for symbol in self.symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    analysis = future.result()
                    if analysis:
                        self.analyses[symbol] = analysis

                        if analysis.has_setup and analysis.setup:
                            # Only LONG in long-only mode
                            if config.strategy.long_only and analysis.setup.direction == "SHORT":
                                continue

                            setups[symbol] = analysis.setup
                            self.active_setups[symbol] = analysis.setup

                            logger.info(
                                f"SETUP: {symbol} {analysis.setup.direction} | "
                                f"Confidence: {analysis.setup.confidence}% | "
                                f"Regime: {analysis.regime.value}"
                            )

                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        self._last_scan = time.time()
        elapsed = time.time() - start_time

        # Sort setups by confidence
        sorted_setups = dict(
            sorted(setups.items(), key=lambda x: x[1].confidence, reverse=True)
        )

        logger.info(
            f"Scan complete: {len(sorted_setups)} setups found "
            f"({elapsed:.1f}s)"
        )

        # Log top setups
        for i, (symbol, setup) in enumerate(list(sorted_setups.items())[:5]):
            logger.info(
                f"  #{i+1}: {symbol} {setup.direction} - "
                f"Confidence: {setup.confidence}% - "
                f"{setup.reasons[0] if setup.reasons else 'N/A'}"
            )

        return sorted_setups

    def get_best_setup(self) -> Optional[TradeSetup]:
        """Get the highest confidence setup."""
        if not self.active_setups:
            return None

        return max(self.active_setups.values(), key=lambda x: x.confidence)

    def get_setups_by_confidence(self, min_confidence: float = 60) -> List[TradeSetup]:
        """Get all setups above minimum confidence."""
        return [
            setup for setup in self.active_setups.values()
            if setup.confidence >= min_confidence
        ]

    def get_statistics(self) -> Dict:
        """Get scanner statistics."""
        regimes = {}
        for analysis in self.analyses.values():
            regime = analysis.regime.value
            regimes[regime] = regimes.get(regime, 0) + 1

        setups_by_direction = {"LONG": 0, "SHORT": 0}
        for setup in self.active_setups.values():
            setups_by_direction[setup.direction] += 1

        avg_confidence = 0
        if self.active_setups:
            avg_confidence = sum(s.confidence for s in self.active_setups.values()) / len(self.active_setups)

        return {
            "total_symbols": len(self.symbols),
            "symbols_analyzed": len(self.analyses),
            "active_setups": len(self.active_setups),
            "setups_long": setups_by_direction["LONG"],
            "setups_short": setups_by_direction["SHORT"],
            "avg_confidence": avg_confidence,
            "regimes": regimes,
            "last_scan": datetime.fromtimestamp(self._last_scan).isoformat() if self._last_scan else None
        }

    def remove_setup(self, symbol: str):
        """Remove a setup (after trade opened)."""
        if symbol in self.active_setups:
            del self.active_setups[symbol]


# Singleton
scanner_v2 = VolumeMomentumScanner()
