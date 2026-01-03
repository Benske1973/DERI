#!/usr/bin/env python3
# main_v2.py - Volume-Momentum Paper Trader
"""
Main orchestrator for Volume-Momentum strategy.
Uses volume, momentum, and volatility analysis for high-probability trades.
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
from scanner_v2 import scanner_v2, VolumeMomentumScanner
from volume_momentum_strategy import TradeSetup, MarketRegime
from websocket_feed import KuCoinWebSocket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger('websockets').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


class VolumeMomentumTrader:
    """
    Main trading orchestrator for Volume-Momentum strategy.

    Features:
    - Scans for high-probability setups
    - Executes paper trades with proper risk management
    - Manages positions with trailing stops
    - Real-time price updates via WebSocket
    """

    def __init__(self):
        self.scanner = scanner_v2
        self.trader = paper_trader
        self.ws_feed: Optional[KuCoinWebSocket] = None
        self.running = False
        self.scan_interval = 300  # 5 minutes
        self.last_scan = 0
        self.pending_setups: Dict[str, TradeSetup] = {}

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("VOLUME-MOMENTUM PAPER TRADER v2.0")
        logger.info("=" * 60)

        # Initialize database
        db.init_database()

        # Initialize scanner
        symbols = self.scanner.initialize()
        logger.info(f"Monitoring {len(symbols)} symbols")

        # Log configuration
        logger.info(f"Strategy: Volume-Momentum Breakout")
        logger.info(f"Mode: {'LONG only' if config.strategy.long_only else 'LONG & SHORT'}")
        logger.info(f"Risk per trade: {config.risk.risk_per_trade * 100:.1f}%")
        logger.info(f"Max positions: {config.risk.max_open_positions}")
        logger.info(f"Initial capital: ${config.risk.initial_capital:,.2f}")

        # Run initial scan
        await self._run_scan()

    async def _run_scan(self):
        """Run market scan."""
        logger.info("-" * 40)
        logger.info("Running market scan...")

        setups = self.scanner.run_scan()
        self.pending_setups = setups
        self.last_scan = time.time()

        stats = self.scanner.get_statistics()
        logger.info(f"Scan results:")
        logger.info(f"  Symbols analyzed: {stats['symbols_analyzed']}")
        logger.info(f"  Active setups: {stats['active_setups']}")
        logger.info(f"  LONG setups: {stats['setups_long']}")
        logger.info(f"  Avg confidence: {stats['avg_confidence']:.1f}%")

        if stats['regimes']:
            logger.info(f"  Market regimes: {stats['regimes']}")

    async def _process_setups(self):
        """Process pending setups and open trades."""
        if not self.pending_setups:
            return

        # Sort by confidence
        sorted_setups = sorted(
            self.pending_setups.items(),
            key=lambda x: x[1].confidence,
            reverse=True
        )

        for symbol, setup in sorted_setups:
            # Check if we can open more positions
            if not self.trader.can_open_position(symbol):
                continue

            # Check direction filter
            if config.strategy.long_only and setup.direction == "SHORT":
                continue

            # Check minimum confidence
            if setup.confidence < 60:
                continue

            # Get current price for verification
            try:
                ticker = kucoin_client.get_ticker(symbol)
                current_price = ticker.price

                # Verify setup is still valid (price hasn't moved too much)
                price_change = abs(current_price - setup.entry_price) / setup.entry_price
                if price_change > 0.01:  # More than 1% move
                    logger.warning(f"{symbol}: Price moved {price_change*100:.1f}%, skipping setup")
                    del self.pending_setups[symbol]
                    continue

                # Open position
                position = self.trader.open_position(
                    symbol=symbol,
                    side=setup.direction,
                    entry_price=current_price,  # Use current price
                    stop_loss=setup.stop_loss,
                    take_profit=setup.take_profit_2  # Use TP2 (2:1 R:R)
                )

                if position:
                    logger.info(
                        f"TRADE OPENED: {setup.direction} {symbol} | "
                        f"Entry: {position.entry_price:.6f} | "
                        f"SL: {setup.stop_loss:.6f} | "
                        f"TP: {setup.take_profit_2:.6f} | "
                        f"Confidence: {setup.confidence}%"
                    )
                    logger.info(f"  Reasons: {', '.join(setup.reasons[:3])}")

                    # Remove from pending
                    del self.pending_setups[symbol]
                    self.scanner.remove_setup(symbol)

                    # Save to database
                    db.save_signal(
                        symbol=symbol,
                        trend=setup.direction,
                        fvg_top=setup.take_profit_2,
                        fvg_bottom=setup.stop_loss,
                        ob_top=0,
                        ob_bottom=0,
                        status="EXECUTED"
                    )

            except Exception as e:
                logger.error(f"Error opening position for {symbol}: {e}")

    def _on_price_update(self, symbol: str, price: float):
        """Handle price update from WebSocket."""
        self.trader.update_price(symbol, price)

    async def _start_websocket(self):
        """Start WebSocket price feed."""
        # Get symbols with positions or setups
        symbols = set(self.trader.portfolio.positions.keys())
        symbols.update(self.pending_setups.keys())

        if not symbols:
            return

        try:
            self.ws_feed = KuCoinWebSocket(list(symbols))
            self.ws_feed.on_price_update = self._on_price_update

            # Run WebSocket in background
            asyncio.create_task(self.ws_feed.connect())
            logger.info(f"WebSocket started for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _update_positions(self):
        """Update all position prices via REST API."""
        for symbol in list(self.trader.portfolio.positions.keys()):
            try:
                ticker = kucoin_client.get_ticker(symbol)
                self.trader.update_price(symbol, ticker.price)
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")

    def _print_dashboard(self):
        """Print current status dashboard."""
        print("\n" + "=" * 80)
        print("VOLUME-MOMENTUM PAPER TRADER")
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Portfolio summary
        summary = self.trader.get_portfolio_summary()
        print(f"\nPORTFOLIO")
        print("-" * 40)
        print(f"Balance:       ${summary['balance']:,.2f}")
        print(f"Equity:        ${summary['equity']:,.2f}")
        print(f"Total P&L:     ${summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:+.2f}%)")
        print(f"Unrealized:    ${summary['unrealized_pnl']:,.2f}")
        print(f"Fees Paid:     ${summary['total_fees_paid']:,.2f}")
        print(f"Win Rate:      {summary['win_rate']:.1f}% ({summary['winning_trades']}/{summary['total_trades']})")
        print(f"Profit Factor: {summary['profit_factor']:.2f}")

        # Open positions
        positions = self.trader.get_open_positions()
        if positions:
            print(f"\nOPEN POSITIONS ({len(positions)})")
            print("-" * 80)
            print(f"{'Symbol':<12} {'Side':<6} {'Entry':<12} {'Current':<12} {'P&L':>10} {'P&L %':>8}")
            print("-" * 80)

            for pos in positions:
                pnl_color = "+" if pos['unrealized_pnl'] >= 0 else ""
                print(
                    f"{pos['symbol']:<12} {pos['side']:<6} "
                    f"{pos['entry_price']:<12.6f} {pos['current_price']:<12.6f} "
                    f"${pnl_color}{pos['unrealized_pnl']:>8.2f} "
                    f"{pnl_color}{pos['unrealized_pnl_percent']:>7.2f}%"
                )

        # Pending setups
        if self.pending_setups:
            print(f"\nPENDING SETUPS ({len(self.pending_setups)})")
            print("-" * 80)
            sorted_setups = sorted(
                self.pending_setups.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )[:5]

            for symbol, setup in sorted_setups:
                print(
                    f"  {symbol:<12} {setup.direction:<6} "
                    f"Confidence: {setup.confidence}% | "
                    f"{setup.reasons[0] if setup.reasons else 'N/A'}"
                )

        # Scanner stats
        stats = self.scanner.get_statistics()
        print(f"\nSCANNER")
        print("-" * 40)
        print(f"Active Setups: {stats['active_setups']} (LONG: {stats['setups_long']})")
        print(f"Avg Confidence: {stats['avg_confidence']:.1f}%")
        if stats['regimes']:
            regime_str = ", ".join(f"{k}: {v}" for k, v in stats['regimes'].items())
            print(f"Regimes: {regime_str}")

        # Recent trades
        history = self.trader.get_trade_history(5)
        if history:
            print(f"\nRECENT TRADES")
            print("-" * 80)
            for trade in history:
                pnl_color = "+" if trade['pnl'] >= 0 else ""
                print(
                    f"  {trade['symbol']:<12} {trade['side']:<6} "
                    f"${pnl_color}{trade['pnl']:>8.2f} ({pnl_color}{trade['pnl_percent']:.2f}%) "
                    f"- {trade['close_reason']}"
                )

        print("=" * 80 + "\n")

    async def run(self):
        """Main run loop."""
        self.running = True
        logger.info("Starting main trading loop...")

        # Setup signal handlers
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

                # Process setups
                await self._process_setups()

                # Print dashboard every 30 seconds
                if iteration % 6 == 0:
                    self._print_dashboard()

                # Sleep
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

        logger.info("Trading loop stopped")
        self._print_dashboard()


async def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="Volume-Momentum Paper Trader")

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run single scan and exit"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit"
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="Close all positions and exit"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset portfolio to initial capital"
    )

    args = parser.parse_args()

    if args.stats:
        db.init_database()
        summary = paper_trader.get_portfolio_summary()
        print("\nPortfolio Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return

    if args.close_all:
        db.init_database()
        scanner_v2.initialize()

        for symbol in list(paper_trader.portfolio.positions.keys()):
            try:
                ticker = kucoin_client.get_ticker(symbol)
                paper_trader.update_price(symbol, ticker.price)
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")

        closed = paper_trader.close_all_positions("MANUAL")
        print(f"Closed {len(closed)} positions")
        for pos in closed:
            print(f"  {pos.symbol}: P&L ${pos.pnl:.2f} ({pos.pnl_percent:.2f}%)")
        return

    if args.reset:
        paper_trader.portfolio.balance = config.risk.initial_capital
        paper_trader.portfolio.equity = config.risk.initial_capital
        paper_trader.portfolio.margin_used = 0
        paper_trader.portfolio.positions.clear()
        paper_trader.portfolio.trade_history.clear()
        paper_trader._total_fees_paid = 0
        print(f"Portfolio reset to ${config.risk.initial_capital:,.2f}")
        return

    if args.scan_only:
        db.init_database()
        scanner_v2.initialize()
        setups = scanner_v2.run_scan()
        print(f"\nFound {len(setups)} setups:")
        for symbol, setup in setups.items():
            print(
                f"  {symbol} {setup.direction} | "
                f"Confidence: {setup.confidence}% | "
                f"Entry: {setup.entry_price:.6f} | "
                f"SL: {setup.stop_loss:.6f} | "
                f"TP: {setup.take_profit_2:.6f}"
            )
            print(f"    Reasons: {', '.join(setup.reasons)}")
        return

    # Normal run
    trader = VolumeMomentumTrader()
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
