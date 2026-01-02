# scanner.py - Multi-Symbol Scanner Engine
"""
Advanced multi-symbol scanner for KuCoin.
Scans multiple timeframes and generates trading signals.
"""

import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config, TimeFrame, TrendDirection, PositionStatus
from kucoin_client import kucoin_client
from indicators import Indicators, FairValueGap, OrderBlock, MarketStructure

logger = logging.getLogger(__name__)

@dataclass
class POI:
    """Point of Interest (Zone to watch)."""
    symbol: str
    poi_type: str  # FVG, OB, LIQUIDITY
    direction: str  # BULLISH, BEARISH
    top: float
    bottom: float
    midpoint: float
    timeframe: str
    created_at: datetime = field(default_factory=datetime.now)
    status: PositionStatus = PositionStatus.SCANNING
    tap_count: int = 0
    last_tap: Optional[datetime] = None

@dataclass
class ScanResult:
    """Result of scanning a symbol."""
    symbol: str
    timestamp: datetime
    current_price: float
    trend: TrendDirection
    htf_bias: str
    ltf_confirmation: bool
    pois: List[POI]
    fvgs: List[FairValueGap]
    order_blocks: List[OrderBlock]
    structure: MarketStructure
    score: float
    analysis: Dict

class MultiScanner:
    """
    Multi-symbol scanner engine.

    Features:
    - Parallel scanning of multiple symbols
    - Multi-timeframe analysis
    - POI detection (FVG, OB, Liquidity)
    - Signal scoring and filtering
    """

    def __init__(self):
        self.client = kucoin_client
        self.cfg = config.scanner
        self.symbols: List[str] = []
        self.scan_results: Dict[str, ScanResult] = {}
        self.active_pois: Dict[str, List[POI]] = {}
        self._last_htf_scan: float = 0
        self._last_ltf_scan: float = 0

    def initialize(self):
        """Initialize scanner with symbols to scan."""
        if self.cfg.symbols:
            self.symbols = self.cfg.symbols
        elif self.cfg.auto_discover:
            self.symbols = self.client.get_top_volume_pairs(
                count=self.cfg.top_pairs_count
            )
            logger.info(f"Auto-discovered {len(self.symbols)} trading pairs")
        else:
            self.symbols = self.client.get_usdt_pairs()

        logger.info(f"Scanner initialized with {len(self.symbols)} symbols")
        return self.symbols

    def scan_symbol_htf(self, symbol: str) -> Optional[ScanResult]:
        """
        Scan a single symbol on higher timeframe.

        Args:
            symbol: Trading pair symbol

        Returns:
            ScanResult or None if no setup found
        """
        try:
            # Get HTF candles
            df = self.client.get_candles(
                symbol=symbol,
                timeframe=self.cfg.htf_timeframe,
                limit=200
            )

            if df.empty or len(df) < 50:
                return None

            # Initialize indicators
            ind = Indicators(df)

            # Get trend
            trend = ind.get_trend()

            # Find POIs
            fvgs = ind.find_fvg()
            order_blocks = ind.find_order_blocks()

            # Analyze market structure
            structure = ind.analyze_market_structure()

            # Get current price
            current_price = df["close"].iloc[-1]

            # Build POIs list
            pois = []

            # Check if we're in LONG-only mode
            long_only = config.strategy.long_only

            # Add valid FVGs as POIs
            for fvg in fvgs[-5:]:  # Last 5 FVGs
                # Check if FVG is still valid (not filled)
                if fvg.type == "BULLISH" and current_price > fvg.bottom:
                    if current_price < fvg.top * 1.1:  # Within 10% above
                        pois.append(POI(
                            symbol=symbol,
                            poi_type="FVG",
                            direction=fvg.type,
                            top=fvg.top,
                            bottom=fvg.bottom,
                            midpoint=fvg.midpoint,
                            timeframe=self.cfg.htf_timeframe.value
                        ))
                elif fvg.type == "BEARISH" and current_price < fvg.top:
                    # Skip BEARISH in LONG-only mode
                    if long_only:
                        continue
                    if current_price > fvg.bottom * 0.9:  # Within 10% below
                        pois.append(POI(
                            symbol=symbol,
                            poi_type="FVG",
                            direction=fvg.type,
                            top=fvg.top,
                            bottom=fvg.bottom,
                            midpoint=fvg.midpoint,
                            timeframe=self.cfg.htf_timeframe.value
                        ))

            # Add valid Order Blocks as POIs
            for ob in order_blocks[-5:]:  # Last 5 OBs
                if ob.type == "BULLISH" and current_price > ob.low:
                    pois.append(POI(
                        symbol=symbol,
                        poi_type="OB",
                        direction=ob.type,
                        top=ob.high,
                        bottom=ob.low,
                        midpoint=(ob.high + ob.low) / 2,
                        timeframe=self.cfg.htf_timeframe.value
                    ))
                elif ob.type == "BEARISH" and current_price < ob.high:
                    # Skip BEARISH in LONG-only mode
                    if long_only:
                        continue
                    pois.append(POI(
                        symbol=symbol,
                        poi_type="OB",
                        direction=ob.type,
                        top=ob.high,
                        bottom=ob.low,
                        midpoint=(ob.high + ob.low) / 2,
                        timeframe=self.cfg.htf_timeframe.value
                    ))

            # Calculate signal score
            htf_bias = "BULLISH" if trend == TrendDirection.BULLISH else \
                       "BEARISH" if trend == TrendDirection.BEARISH else "NEUTRAL"

            score = 0
            if pois:
                score += 30  # Has POIs
            if trend != TrendDirection.NEUTRAL:
                score += 20  # Has clear trend
            if structure.bos_detected:
                score += 25
            if structure.choch_detected:
                score += 25

            # Get additional analysis
            ind_df = ind.add_all_indicators()
            analysis = {
                "rsi": ind_df["rsi"].iloc[-1] if "rsi" in ind_df else 50,
                "ema_fast": ind_df["ema_fast"].iloc[-1] if "ema_fast" in ind_df else current_price,
                "ema_slow": ind_df["ema_slow"].iloc[-1] if "ema_slow" in ind_df else current_price,
                "atr": ind_df["atr"].iloc[-1] if "atr" in ind_df else 0,
                "volume_ratio": df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
            }

            return ScanResult(
                symbol=symbol,
                timestamp=datetime.now(),
                current_price=current_price,
                trend=trend,
                htf_bias=htf_bias,
                ltf_confirmation=False,
                pois=pois,
                fvgs=fvgs,
                order_blocks=order_blocks,
                structure=structure,
                score=score,
                analysis=analysis
            )

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            return None

    def scan_symbol_ltf(self, symbol: str, poi: POI) -> Dict:
        """
        Scan LTF for entry confirmation.

        Args:
            symbol: Trading pair symbol
            poi: Point of Interest to confirm

        Returns:
            Confirmation result dictionary
        """
        try:
            # Get LTF candles
            df = self.client.get_candles(
                symbol=symbol,
                timeframe=self.cfg.ltf_timeframe,
                limit=100
            )

            if df.empty or len(df) < 20:
                return {"confirmed": False, "reason": "Insufficient data"}

            ind = Indicators(df)
            current_price = df["close"].iloc[-1]

            # Check if price has tapped the zone
            zone_tapped = poi.bottom <= current_price <= poi.top

            if not zone_tapped:
                return {"confirmed": False, "reason": "Zone not tapped"}

            # Look for LTF confirmation
            structure = ind.analyze_market_structure()
            score = ind.calculate_signal_score(
                "LONG" if poi.direction == "BULLISH" else "SHORT"
            )

            # Confirmation criteria
            confirmed = False
            reasons = []

            if poi.direction == "BULLISH":
                # Need CHoCH or BOS to upside
                if structure.choch_detected and structure.structure_breaks:
                    for sb in structure.structure_breaks:
                        if sb["direction"] == "BULLISH":
                            confirmed = True
                            reasons.append("Bullish CHoCH detected")
                            break

                # Check for bullish engulfing or momentum
                if df["close"].iloc[-1] > df["open"].iloc[-1]:  # Bullish candle
                    if df["close"].iloc[-1] > df["high"].iloc[-2]:  # Engulfing
                        confirmed = True
                        reasons.append("Bullish engulfing")

            else:  # BEARISH
                if structure.choch_detected and structure.structure_breaks:
                    for sb in structure.structure_breaks:
                        if sb["direction"] == "BEARISH":
                            confirmed = True
                            reasons.append("Bearish CHoCH detected")
                            break

                if df["close"].iloc[-1] < df["open"].iloc[-1]:  # Bearish candle
                    if df["close"].iloc[-1] < df["low"].iloc[-2]:  # Engulfing
                        confirmed = True
                        reasons.append("Bearish engulfing")

            # Additional filters
            if score.total_score < config.strategy.min_score:
                confirmed = False
                reasons.append(f"Score too low: {score.total_score}")

            return {
                "confirmed": confirmed,
                "reasons": reasons,
                "score": score.total_score,
                "current_price": current_price,
                "entry_price": poi.top if poi.direction == "BULLISH" else poi.bottom,
                "atr": ind.atr().iloc[-1]
            }

        except Exception as e:
            logger.error(f"Error in LTF scan for {symbol}: {e}")
            return {"confirmed": False, "reason": str(e)}

    def run_htf_scan(self) -> Dict[str, ScanResult]:
        """
        Run HTF scan on all symbols.

        Returns:
            Dictionary of symbol to ScanResult
        """
        results = {}
        start_time = time.time()

        logger.info(f"Starting HTF scan on {len(self.symbols)} symbols...")

        # Use thread pool for parallel scanning
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.scan_symbol_htf, symbol): symbol
                for symbol in self.symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result and result.pois:
                        results[symbol] = result
                        self.active_pois[symbol] = result.pois

                        logger.info(
                            f"Found {len(result.pois)} POIs for {symbol} "
                            f"(Score: {result.score}, Trend: {result.htf_bias})"
                        )
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        self.scan_results = results
        self._last_htf_scan = time.time()

        elapsed = time.time() - start_time
        logger.info(
            f"HTF scan complete: {len(results)} symbols with setups "
            f"({elapsed:.1f}s)"
        )

        return results

    def check_poi_taps(self, prices: Dict[str, float]) -> List[Tuple]:
        """
        Check if any POIs have been tapped.

        Args:
            prices: Dictionary of symbol to current price

        Returns:
            List of (symbol, poi, confirmation) tuples
        """
        tapped = []

        for symbol, pois in self.active_pois.items():
            if symbol not in prices:
                continue

            price = prices[symbol]

            for poi in pois:
                if poi.status != PositionStatus.SCANNING:
                    continue

                # Check if price is in the zone
                if poi.bottom <= price <= poi.top:
                    poi.tap_count += 1
                    poi.last_tap = datetime.now()
                    poi.status = PositionStatus.TAPPED

                    logger.info(
                        f"POI TAPPED: {symbol} @ {price:.4f} "
                        f"(Zone: {poi.bottom:.4f}-{poi.top:.4f})"
                    )

                    # Check LTF confirmation
                    confirmation = self.scan_symbol_ltf(symbol, poi)

                    if confirmation.get("confirmed"):
                        poi.status = PositionStatus.CONFIRMED
                        tapped.append((symbol, poi, confirmation))

                        logger.info(
                            f"CONFIRMED: {symbol} - {confirmation.get('reasons', [])}"
                        )

        return tapped

    def get_active_setups(self) -> List[Dict]:
        """Get all active setups waiting for entry."""
        setups = []

        for symbol, result in self.scan_results.items():
            for poi in result.pois:
                if poi.status in [PositionStatus.SCANNING, PositionStatus.TAPPED]:
                    setups.append({
                        "symbol": symbol,
                        "direction": poi.direction,
                        "poi_type": poi.poi_type,
                        "zone_top": poi.top,
                        "zone_bottom": poi.bottom,
                        "status": poi.status.value,
                        "trend": result.htf_bias,
                        "score": result.score
                    })

        return setups

    def get_statistics(self) -> Dict:
        """Get scanner statistics."""
        total_pois = sum(len(pois) for pois in self.active_pois.values())
        bullish_pois = sum(
            1 for pois in self.active_pois.values()
            for poi in pois if poi.direction == "BULLISH"
        )
        bearish_pois = total_pois - bullish_pois

        return {
            "total_symbols": len(self.symbols),
            "symbols_with_setups": len(self.scan_results),
            "total_pois": total_pois,
            "bullish_pois": bullish_pois,
            "bearish_pois": bearish_pois,
            "last_htf_scan": datetime.fromtimestamp(self._last_htf_scan).isoformat()
            if self._last_htf_scan else None,
            "next_htf_scan": datetime.fromtimestamp(
                self._last_htf_scan + self.cfg.htf_scan_interval
            ).isoformat() if self._last_htf_scan else None
        }


# Singleton instance
scanner = MultiScanner()


# Compatibility functions for existing code
from typing import Tuple as T

def detect_smc_setup(df) -> T[Optional[str], float, float, float, float]:
    """Legacy compatibility function."""
    ind = Indicators(df)
    fvgs = ind.find_fvg()
    obs = ind.find_order_blocks()

    if fvgs:
        fvg = fvgs[-1]
        if obs:
            ob = obs[-1]
            return fvg.type, fvg.top, fvg.bottom, ob.high, ob.low

    return None, 0, 0, 0, 0


def run_scanner():
    """Legacy compatibility function."""
    from database import db

    scanner.initialize()
    results = scanner.run_htf_scan()

    for symbol, result in results.items():
        for poi in result.pois:
            db.save_signal(
                symbol=symbol,
                trend=poi.direction,
                fvg_top=poi.top,
                fvg_bottom=poi.bottom,
                ob_top=poi.top,  # Using POI boundaries
                ob_bottom=poi.bottom,
                status=poi.status.value
            )
            print(f"POI saved for {symbol} (Zone: {poi.bottom:.4f}-{poi.top:.4f})")
