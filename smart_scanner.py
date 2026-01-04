# smart_scanner.py - Smart KuCoin Scanner for Liquid Pairs
"""
Scans KuCoin for the best tradable pairs based on:
- Liquidity (can handle 200 USDT position)
- Spread (low spread = less trading cost)
- Volatility (coins that move, not dead)
- Double Top/Bottom patterns
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config, TimeFrame
from kucoin_client import kucoin_client
from double_patterns import detect_double_pattern, PatternResult
from volume_momentum_strategy import (
    VolumeAnalyzer,
    VolatilityAnalyzer,
    MomentumAnalyzer,
    MarketRegime
)

logger = logging.getLogger(__name__)


@dataclass
class CoinMetrics:
    """Metrics for evaluating coin tradability."""
    symbol: str
    price: float
    volume_24h: float  # USDT volume
    spread_percent: float  # Bid-ask spread
    volatility_percent: float  # ATR as % of price
    avg_candle_range: float  # Average high-low range %
    liquidity_score: float  # 0-100
    movement_score: float  # 0-100 (how much it moves)
    overall_score: float  # Combined score
    can_trade_200usdt: bool
    regime: MarketRegime
    pattern: Optional[PatternResult] = None


@dataclass
class TradeSignal:
    """Trade signal from scanner."""
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    pattern_type: str
    reasons: List[str]
    metrics: CoinMetrics
    timestamp: datetime = field(default_factory=datetime.now)


class SmartScanner:
    """
    Smart scanner that finds the best coins to trade.

    Criteria:
    1. Minimum 24h volume for liquidity
    2. Low spread (< 0.3%)
    3. Good volatility (1-10% daily range)
    4. Not stablecoins or dead coins
    """

    def __init__(self, position_size: float = 200.0):
        """
        Initialize scanner.

        Args:
            position_size: Position size in USDT (default 200)
        """
        self.client = kucoin_client
        self.position_size = position_size
        self.cfg = config.scanner

        # Minimum requirements
        self.min_volume_24h = position_size * 1000  # Need 1000x position size in volume
        self.max_spread_percent = 0.3  # Max 0.3% spread
        self.min_volatility = 1.0  # Min 1% daily range
        self.max_volatility = 15.0  # Max 15% (too volatile = risky)

        # Cache
        self.coin_metrics: Dict[str, CoinMetrics] = {}
        self.trade_signals: Dict[str, TradeSignal] = {}
        self.qualified_coins: List[str] = []

    def get_all_usdt_pairs(self) -> List[Dict]:
        """Get all USDT trading pairs with their stats."""
        try:
            tickers = self.client.get_all_tickers()
            usdt_pairs = []
            for t in tickers:
                if t.symbol.endswith('-USDT'):
                    # Skip stablecoins
                    if any(stable in t.symbol for stable in ['USDC', 'DAI', 'BUSD', 'TUSD', 'UST']):
                        continue
                    usdt_pairs.append({
                        'symbol': t.symbol,
                        'last': t.price,
                        'volValue': t.volume_24h
                    })
            return usdt_pairs
        except Exception as e:
            logger.error(f"Error getting tickers: {e}")
            return []

    def calculate_spread(self, symbol: str) -> float:
        """
        Calculate bid-ask spread for a symbol.

        Returns:
            Spread as percentage
        """
        try:
            orderbook = self.client.get_orderbook(symbol, depth=20)  # KuCoin only supports 20 or 100
            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                if best_bid > 0:
                    spread = (best_ask - best_bid) / best_bid * 100
                    return spread
        except Exception as e:
            logger.debug(f"Error getting orderbook for {symbol}: {e}")
        return 999  # High spread if can't calculate

    def analyze_coin(self, symbol: str, ticker_data: Dict = None) -> Optional[CoinMetrics]:
        """
        Analyze a single coin for tradability.

        Args:
            symbol: Trading pair symbol
            ticker_data: Optional pre-fetched ticker data

        Returns:
            CoinMetrics or None if not tradable
        """
        try:
            # Get price and volume from ticker
            if ticker_data:
                price = float(ticker_data.get('last', 0))
                volume_24h = float(ticker_data.get('volValue', 0))  # USDT volume
            else:
                ticker = self.client.get_ticker(symbol)
                price = ticker.price
                volume_24h = ticker.volume_24h

            # Quick filter: minimum volume
            if volume_24h < self.min_volume_24h:
                return None

            # Get candles for volatility analysis
            df = self.client.get_candles(symbol, TimeFrame.H4, limit=100)
            if df.empty or len(df) < 20:
                return None

            # Calculate volatility
            vol_analyzer = VolatilityAnalyzer(df)
            vol_state = vol_analyzer.analyze()
            volatility = vol_state.atr_percent

            # Check volatility range
            if volatility < self.min_volatility or volatility > self.max_volatility:
                return None

            # Calculate average candle range (movement)
            df['range'] = (df['high'] - df['low']) / df['low'] * 100
            avg_range = df['range'].mean()

            # Estimate spread from recent candles (bid-ask is typically a fraction of range)
            # This avoids slow orderbook calls
            estimated_spread = avg_range * 0.05  # Assume spread is ~5% of avg range

            # Calculate scores
            liquidity_score = min(100, (volume_24h / self.min_volume_24h) * 10)
            movement_score = min(100, (avg_range / 2) * 50)  # 2% range = 50 score

            # Overall score (no spread penalty for now)
            overall_score = (liquidity_score * 0.3 + movement_score * 0.5 + 20)  # Base 20 points
            overall_score = max(0, min(100, overall_score))

            # Can we trade 200 USDT without issues?
            can_trade = volume_24h > self.position_size * 100

            return CoinMetrics(
                symbol=symbol,
                price=price,
                volume_24h=volume_24h,
                spread_percent=estimated_spread,
                volatility_percent=volatility,
                avg_candle_range=avg_range,
                liquidity_score=liquidity_score,
                movement_score=movement_score,
                overall_score=overall_score,
                can_trade_200usdt=can_trade,
                regime=vol_state.regime
            )

        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
            return None

    def scan_for_patterns(self, symbol: str, metrics: CoinMetrics) -> Optional[TradeSignal]:
        """
        Scan a coin for trading patterns.

        Args:
            symbol: Trading pair
            metrics: Pre-calculated metrics

        Returns:
            TradeSignal if pattern found
        """
        try:
            # Get candles
            df = self.client.get_candles(symbol, TimeFrame.H4, limit=100)
            if df.empty or len(df) < 50:
                return None

            # Detect double patterns
            pattern = detect_double_pattern(df, tolerance_percent=3.0, lookback=50)
            metrics.pattern = pattern

            # Check if we have a valid signal
            if pattern.pattern_type == "NONE" or pattern.signal == "WAIT":
                return None

            # Only take LONG signals (as configured)
            if config.strategy.long_only and pattern.signal != "LONG":
                return None

            # Calculate entry, SL, TP
            current_price = df['close'].iloc[-1]
            atr = df['high'].sub(df['low']).rolling(14).mean().iloc[-1]

            if pattern.signal == "LONG":
                entry = current_price
                sl = max(pattern.neckline * 0.98, entry - atr * 2)
                tp = pattern.target_price
            else:  # SHORT
                entry = current_price
                sl = min(pattern.neckline * 1.02, entry + atr * 2)
                tp = pattern.target_price

            # Build reasons
            reasons = pattern.reasons.copy()
            reasons.append(f"Liquidity: ${metrics.volume_24h/1000:.0f}K")
            reasons.append(f"Spread: {metrics.spread_percent:.3f}%")
            reasons.append(f"Volatility: {metrics.volatility_percent:.1f}%")

            return TradeSignal(
                symbol=symbol,
                direction=pattern.signal,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=pattern.confidence,
                pattern_type=pattern.pattern_type,
                reasons=reasons,
                metrics=metrics
            )

        except Exception as e:
            logger.debug(f"Error scanning patterns for {symbol}: {e}")
            return None

    def run_full_scan(self) -> Tuple[List[CoinMetrics], List[TradeSignal]]:
        """
        Run a full scan of all USDT pairs.

        Returns:
            Tuple of (qualified coins, trade signals)
        """
        start_time = time.time()
        logger.info("Starting smart scan...")

        # Get all USDT pairs
        all_tickers = self.get_all_usdt_pairs()
        logger.info(f"Found {len(all_tickers)} USDT pairs")

        # Analyze coins in parallel
        qualified = []
        signals = []

        with ThreadPoolExecutor(max_workers=15) as executor:
            # First pass: analyze all coins
            futures = {
                executor.submit(self.analyze_coin, t['symbol'], t): t['symbol']
                for t in all_tickers
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    metrics = future.result()
                    if metrics and metrics.overall_score >= 30:
                        qualified.append(metrics)
                        self.coin_metrics[symbol] = metrics
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")

        # Sort by score
        qualified.sort(key=lambda x: x.overall_score, reverse=True)

        # Take top 50 for pattern scanning
        top_coins = qualified[:50]
        logger.info(f"Qualified {len(qualified)} coins, scanning top {len(top_coins)} for patterns...")

        # Second pass: scan for patterns
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.scan_for_patterns, m.symbol, m): m.symbol
                for m in top_coins
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    signal = future.result()
                    if signal:
                        signals.append(signal)
                        self.trade_signals[symbol] = signal
                        logger.info(
                            f"SIGNAL: {signal.direction} {symbol} | "
                            f"Pattern: {signal.pattern_type} | "
                            f"Confidence: {signal.confidence}%"
                        )
                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")

        # Sort signals by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)

        self.qualified_coins = [m.symbol for m in qualified]

        elapsed = time.time() - start_time
        logger.info(f"Scan complete: {len(qualified)} qualified coins, {len(signals)} signals ({elapsed:.1f}s)")

        return qualified, signals

    def get_top_coins(self, limit: int = 20) -> List[CoinMetrics]:
        """Get top coins by overall score."""
        sorted_coins = sorted(
            self.coin_metrics.values(),
            key=lambda x: x.overall_score,
            reverse=True
        )
        return sorted_coins[:limit]

    def get_active_signals(self) -> List[TradeSignal]:
        """Get all active trade signals."""
        return sorted(
            self.trade_signals.values(),
            key=lambda x: x.confidence,
            reverse=True
        )

    def print_summary(self):
        """Print scan summary."""
        print("\n" + "=" * 80)
        print("SMART SCANNER RESULTS")
        print("=" * 80)

        top_coins = self.get_top_coins(10)
        if top_coins:
            print(f"\nTOP 10 TRADABLE COINS (Position: ${self.position_size})")
            print("-" * 80)
            print(f"{'Symbol':<12} {'Price':>10} {'Vol 24h':>12} {'Spread':>8} {'ATR%':>6} {'Score':>6}")
            print("-" * 80)
            for coin in top_coins:
                print(
                    f"{coin.symbol:<12} "
                    f"${coin.price:>9.4f} "
                    f"${coin.volume_24h/1000:>10.0f}K "
                    f"{coin.spread_percent:>7.3f}% "
                    f"{coin.volatility_percent:>5.1f}% "
                    f"{coin.overall_score:>5.0f}"
                )

        signals = self.get_active_signals()
        if signals:
            print(f"\nACTIVE SIGNALS ({len(signals)})")
            print("-" * 80)
            for signal in signals[:5]:
                print(
                    f"{signal.direction:<5} {signal.symbol:<12} | "
                    f"{signal.pattern_type:<15} | "
                    f"Conf: {signal.confidence}% | "
                    f"Entry: {signal.entry_price:.6f}"
                )
                print(f"       Reasons: {', '.join(signal.reasons[:2])}")
        else:
            print("\nNo active signals found")

        print("=" * 80 + "\n")


# Singleton
smart_scanner = SmartScanner(position_size=200.0)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s'
    )

    scanner = SmartScanner(position_size=200.0)
    qualified, signals = scanner.run_full_scan()
    scanner.print_summary()
