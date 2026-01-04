#!/usr/bin/env python3
# main_v3.py - Smart KuCoin Scanner with Double Pattern Detection
"""
Paper trader using smart coin selection:
- Filters for liquid pairs (can handle 200 USDT)
- Low spread and slippage
- Good volatility (coins that move)
- Double Top/Bottom pattern detection
"""

import asyncio
import argparse
import logging
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

from config import config
from database import db
from kucoin_client import kucoin_client
from paper_trader import paper_trader
from smart_scanner import SmartScanner, TradeSignal, CoinMetrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce noise
logging.getLogger('urllib3').setLevel(logging.WARNING)


class SmartTrader:
    """
    Paper trader with smart coin selection.

    Features:
    - Scans for liquid, low-spread pairs
    - Detects Double Top/Bottom patterns
    - Filters dead coins
    - Manages risk with proper position sizing
    """

    def __init__(self, position_size: float = 200.0):
        self.scanner = SmartScanner(position_size=position_size)
        self.trader = paper_trader
        self.position_size = position_size
        self.running = False
        self.scan_interval = 300  # 5 minutes
        self.last_scan = 0
        self.pending_signals: Dict[str, TradeSignal] = {}

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("SMART KUCOIN SCANNER v3.0")
        logger.info("=" * 60)
        logger.info(f"Position Size: ${self.position_size}")
        logger.info(f"Mode: {'LONG only' if config.strategy.long_only else 'LONG & SHORT'}")
        logger.info(f"Max Positions: {config.risk.max_open_positions}")
        logger.info("=" * 60)

        # Initialize database
        db.init_database()

        # Run initial scan
        await self._run_scan()

    async def _run_scan(self):
        """Run smart scan."""
        logger.info("-" * 40)
        logger.info("Running smart scan...")

        qualified, signals = self.scanner.run_full_scan()
        self.pending_signals = {s.symbol: s for s in signals}
        self.last_scan = time.time()

        logger.info(f"Found {len(qualified)} liquid coins, {len(signals)} trade signals")

        # Print summary
        self.scanner.print_summary()

    async def _process_signals(self):
        """Process pending signals and open trades."""
        if not self.pending_signals:
            return

        # Sort by confidence
        sorted_signals = sorted(
            self.pending_signals.values(),
            key=lambda x: x.confidence,
            reverse=True
        )

        for signal in sorted_signals:
            # Check if we can open more positions
            if not self.trader.can_open_position(signal.symbol):
                continue

            # Check direction filter
            if config.strategy.long_only and signal.direction == "SHORT":
                continue

            # Check minimum confidence
            if signal.confidence < 50:
                continue

            try:
                # Get current price
                ticker = kucoin_client.get_ticker(signal.symbol)
                if not ticker:
                    continue

                current_price = ticker.price

                # Verify price hasn't moved too much
                price_change = abs(current_price - signal.entry_price) / signal.entry_price
                if price_change > 0.02:  # 2% move
                    logger.warning(f"{signal.symbol}: Price moved {price_change*100:.1f}%, skipping")
                    del self.pending_signals[signal.symbol]
                    continue

                # Open position
                position = self.trader.open_position(
                    symbol=signal.symbol,
                    side=signal.direction,
                    entry_price=current_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit
                )

                if position:
                    logger.info(
                        f"TRADE OPENED: {signal.direction} {signal.symbol} | "
                        f"Pattern: {signal.pattern_type} | "
                        f"Entry: {position.entry_price:.6f} | "
                        f"SL: {signal.stop_loss:.6f} | "
                        f"TP: {signal.take_profit:.6f}"
                    )

                    # Remove from pending
                    del self.pending_signals[signal.symbol]

            except Exception as e:
                logger.error(f"Error opening {signal.symbol}: {e}")

    async def _update_positions(self):
        """Update all position prices."""
        for symbol in list(self.trader.portfolio.positions.keys()):
            try:
                ticker = kucoin_client.get_ticker(symbol)
                if ticker:
                    self.trader.update_price(symbol, ticker.price)
            except Exception as e:
                logger.debug(f"Error updating {symbol}: {e}")

    def _print_dashboard(self):
        """Print status dashboard."""
        print("\n" + "=" * 80)
        print(f"SMART KUCOIN SCANNER | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Portfolio
        summary = self.trader.get_portfolio_summary()
        print(f"\nPORTFOLIO (Position Size: ${self.position_size})")
        print("-" * 40)
        print(f"Balance:       ${summary['balance']:,.2f}")
        print(f"Equity:        ${summary['equity']:,.2f}")
        print(f"Total P&L:     ${summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:+.2f}%)")
        print(f"Open Positions: {summary['open_positions']}")
        print(f"Win Rate:      {summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})")

        # Open positions
        positions = self.trader.get_open_positions()
        if positions:
            print(f"\nOPEN POSITIONS")
            print("-" * 80)
            for pos in positions[:5]:
                pnl_str = f"+${pos['unrealized_pnl']:.2f}" if pos['unrealized_pnl'] >= 0 else f"-${abs(pos['unrealized_pnl']):.2f}"
                print(f"  {pos['symbol']:<12} {pos['side']:<5} Entry: {pos['entry_price']:.6f}  P&L: {pnl_str}")

        # Pending signals
        if self.pending_signals:
            print(f"\nPENDING SIGNALS ({len(self.pending_signals)})")
            print("-" * 80)
            for signal in list(self.pending_signals.values())[:5]:
                print(
                    f"  {signal.symbol:<12} {signal.direction:<5} "
                    f"{signal.pattern_type:<15} Conf: {signal.confidence}%"
                )

        # Top coins
        top_coins = self.scanner.get_top_coins(5)
        if top_coins:
            print(f"\nTOP LIQUID COINS")
            print("-" * 80)
            for coin in top_coins:
                print(
                    f"  {coin.symbol:<12} Vol: ${coin.volume_24h/1000:.0f}K  "
                    f"Spread: {coin.spread_percent:.3f}%  ATR: {coin.volatility_percent:.1f}%"
                )

        print("=" * 80 + "\n")

    async def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting trading loop...")

        def signal_handler(sig, frame):
            logger.info("Shutdown signal received...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        iteration = 0
        while self.running:
            try:
                iteration += 1

                # Periodic scan
                if time.time() - self.last_scan > self.scan_interval:
                    await self._run_scan()

                # Update positions
                await self._update_positions()

                # Process signals
                await self._process_signals()

                # Print dashboard every 30 seconds
                if iteration % 6 == 0:
                    self._print_dashboard()

                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

        logger.info("Trading loop stopped")
        self._print_dashboard()


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Smart KuCoin Scanner")

    parser.add_argument(
        "--position-size",
        type=float,
        default=200.0,
        help="Position size in USDT (default: 200)"
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run single scan and exit"
    )
    parser.add_argument(
        "--top-coins",
        type=int,
        default=0,
        help="Show top N liquid coins and exit"
    )

    args = parser.parse_args()

    if args.top_coins > 0:
        scanner = SmartScanner(position_size=args.position_size)
        qualified, _ = scanner.run_full_scan()
        print(f"\nTOP {args.top_coins} LIQUID COINS FOR ${args.position_size} TRADING:")
        print("-" * 80)
        print(f"{'Symbol':<12} {'Price':>10} {'Volume 24h':>12} {'Spread':>8} {'ATR%':>6} {'Score':>6}")
        print("-" * 80)
        for coin in qualified[:args.top_coins]:
            print(
                f"{coin.symbol:<12} "
                f"${coin.price:>9.4f} "
                f"${coin.volume_24h/1000:>10.0f}K "
                f"{coin.spread_percent:>7.3f}% "
                f"{coin.volatility_percent:>5.1f}% "
                f"{coin.overall_score:>5.0f}"
            )
        return

    if args.scan_only:
        scanner = SmartScanner(position_size=args.position_size)
        qualified, signals = scanner.run_full_scan()
        scanner.print_summary()
        return

    # Normal run
    trader = SmartTrader(position_size=args.position_size)
    await trader.initialize()
    await trader.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
