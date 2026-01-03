#!/usr/bin/env python3
# main.py - KuCoin Multiscanner Papertrader Main Orchestrator
"""
Main entry point for the KuCoin Multiscanner Papertrader.
Orchestrates all components and runs the trading loop.
"""

import asyncio
import signal
import sys
import time
import logging
import threading
from datetime import datetime
from typing import Optional

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('papertrader.log')
    ]
)

logger = logging.getLogger("main")

# Import components
from config import config, PositionStatus
from database import db
from kucoin_client import kucoin_client
from scanner import scanner, MultiScanner
from paper_trader import paper_trader
from websocket_feed import ws_feed
from strategies import strategy, Signal
from indicators import Indicators

class PaperTraderOrchestrator:
    """
    Main orchestrator for the paper trading system.

    Coordinates:
    - HTF scanning for setups
    - WebSocket price monitoring
    - Signal generation and execution
    - Position management
    - Statistics tracking
    """

    def __init__(self):
        self.is_running = False
        self.scanner = scanner
        self.trader = paper_trader
        self.strategy = strategy

        # Threads
        self._ws_thread: Optional[threading.Thread] = None
        self._scanner_thread: Optional[threading.Thread] = None
        self._executor_thread: Optional[threading.Thread] = None

        # State
        self._last_htf_scan = 0
        self._last_snapshot = 0
        self._pending_signals: dict = {}

    def setup(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("KUCOIN MULTISCANNER PAPERTRADER")
        logger.info("=" * 60)

        # Initialize database
        logger.info("Initializing database...")
        db.init_database()

        # Initialize scanner
        logger.info("Initializing scanner...")
        symbols = self.scanner.initialize()
        logger.info(f"Scanning {len(symbols)} symbols")

        # Log configuration
        logger.info(f"Initial Capital: ${config.risk.initial_capital:,.2f}")
        logger.info(f"Risk per Trade: {config.risk.risk_per_trade * 100:.1f}%")
        logger.info(f"Max Positions: {config.risk.max_open_positions}")
        logger.info(f"HTF Timeframe: {config.scanner.htf_timeframe.value}")
        logger.info(f"LTF Timeframe: {config.scanner.ltf_timeframe.value}")

        logger.info("Setup complete!")
        logger.info("=" * 60)

    def price_callback(self, symbol: str, price: float):
        """Handle price updates from WebSocket."""
        # Update paper trader
        self.trader.update_price(symbol, price)

        # Check partial TP levels
        self.trader.check_partial_tp_levels(symbol, price)

        # Check for POI taps
        if symbol in scanner.active_pois:
            for poi in scanner.active_pois[symbol]:
                if poi.status.value == "SCANNING":
                    if poi.bottom <= price <= poi.top:
                        poi.status = PositionStatus.TAPPED
                        db.update_signal_status(symbol, "TAPPED")
                        logger.info(f"POI TAPPED: {symbol} @ {price:.4f}")

                        # Check for immediate confirmation
                        self._check_confirmation(symbol, poi, price)

    def _check_confirmation(self, symbol: str, poi, price: float):
        """Check for entry confirmation on LTF."""
        try:
            logger.info(f"Checking confirmation for {symbol} @ {price:.4f}")

            # Get LTF data
            df = kucoin_client.get_candles(
                symbol=symbol,
                timeframe=config.scanner.ltf_timeframe,
                limit=100
            )

            if df.empty:
                logger.warning(f"No LTF data for {symbol}")
                return

            ind = Indicators(df)

            # Get scan result
            scan_result = scanner.scan_results.get(symbol)
            if not scan_result:
                logger.warning(f"No scan result for {symbol} - creating minimal result")
                # Create a minimal scan result for the trade
                from scanner import ScanResult
                scan_result = ScanResult(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    current_price=price,
                    trend=ind.get_trend(),
                    htf_bias=poi.direction,
                    ltf_confirmation=True,
                    pois=[poi],
                    fvgs=[],
                    order_blocks=[],
                    structure=ind.analyze_market_structure(),
                    score=50,
                    analysis={}
                )

            # Generate signal
            result = self.strategy.generate_signal(
                scan_result=scan_result,
                poi=poi,
                indicators=ind,
                current_price=price
            )

            logger.info(f"Signal result for {symbol}: has_signal={result.has_signal}, reason={result.reason}")
            logger.info(f"  Checks passed: {result.checks_passed}")
            logger.info(f"  Checks failed: {result.checks_failed}")

            if result.has_signal and result.signal:
                logger.info(f"EXECUTING SIGNAL for {symbol}")
                self._execute_signal(result.signal)
            else:
                logger.info(f"No signal for {symbol}: {result.reason}")

        except Exception as e:
            logger.error(f"Error checking confirmation for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        if signal.executed:
            return

        # Check if we can open position
        if not self.trader.can_open_position(signal.symbol):
            logger.warning(f"Cannot open position for {signal.symbol}")
            return

        # Open position
        position = self.trader.open_position(
            symbol=signal.symbol,
            side=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )

        if position:
            signal.executed = True

            # Save to database
            db.save_trade(
                trade_id=position.id,
                symbol=signal.symbol,
                side=signal.direction,
                entry_price=signal.entry_price,
                quantity=position.quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                metadata={
                    "score": signal.score,
                    "confidence": signal.confidence,
                    "analysis": signal.analysis
                }
            )

            # Update signal status
            db.update_signal_status(signal.symbol, "IN_TRADE")

            logger.info(
                f"TRADE EXECUTED: {signal.direction} {signal.symbol} | "
                f"Size: {position.quantity:.4f} | "
                f"Entry: {signal.entry_price:.4f}"
            )

    def run_htf_scanner(self):
        """Run HTF scanner loop."""
        logger.info("HTF Scanner started")

        while self.is_running:
            try:
                # Check if it's time to scan
                elapsed = time.time() - self._last_htf_scan

                if elapsed >= config.scanner.htf_scan_interval:
                    logger.info("Running HTF scan...")

                    results = self.scanner.run_htf_scan()

                    # Save signals to database
                    for symbol, result in results.items():
                        for poi in result.pois:
                            db.save_signal(
                                symbol=symbol,
                                trend=poi.direction,
                                fvg_top=poi.top,
                                fvg_bottom=poi.bottom,
                                ob_top=poi.top,
                                ob_bottom=poi.bottom,
                                status=poi.status.value,
                                poi_type=poi.poi_type,
                                score=result.score,
                                timeframe=poi.timeframe
                            )

                    self._last_htf_scan = time.time()

                    # Clean old signals
                    db.clear_old_signals(hours=24)

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"HTF Scanner error: {e}")
                time.sleep(30)

    def run_executor(self):
        """Run trade executor loop."""
        logger.info("Executor started")

        while self.is_running:
            try:
                # Get tapped signals waiting for confirmation
                tapped = db.get_signals_by_status("TAPPED")

                for signal_data in tapped:
                    symbol = signal_data['symbol']

                    # Get current price
                    price = ws_feed.get_price(symbol)
                    if not price:
                        continue

                    # Find POI
                    pois = scanner.active_pois.get(symbol, [])
                    for poi in pois:
                        if poi.status.value == "TAPPED":
                            self._check_confirmation(symbol, poi, price)

                # Save portfolio snapshot periodically
                elapsed = time.time() - self._last_snapshot
                if elapsed >= 300:  # Every 5 minutes
                    summary = self.trader.get_portfolio_summary()
                    db.save_portfolio_snapshot(
                        balance=summary['balance'],
                        equity=summary['equity'],
                        unrealized_pnl=summary['unrealized_pnl'],
                        open_positions=summary['open_positions'],
                        total_trades=summary['total_trades'],
                        win_rate=summary['win_rate'],
                        profit_factor=summary['profit_factor']
                    )
                    self._last_snapshot = time.time()

                time.sleep(5)

            except Exception as e:
                logger.error(f"Executor error: {e}")
                time.sleep(10)

    async def run_websocket(self):
        """Run WebSocket connection."""
        ws_feed.add_callback(self.price_callback)
        await ws_feed.run()

    def _run_ws_thread(self):
        """Run WebSocket in thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_websocket())

    def start(self):
        """Start all components."""
        self.is_running = True

        # Start WebSocket thread
        self._ws_thread = threading.Thread(target=self._run_ws_thread, daemon=True)
        self._ws_thread.start()
        logger.info("WebSocket thread started")

        # Start scanner thread
        self._scanner_thread = threading.Thread(target=self.run_htf_scanner, daemon=True)
        self._scanner_thread.start()
        logger.info("Scanner thread started")

        # Start executor thread
        self._executor_thread = threading.Thread(target=self.run_executor, daemon=True)
        self._executor_thread.start()
        logger.info("Executor thread started")

        # Run initial scan
        time.sleep(2)  # Wait for WS connection
        logger.info("Running initial scan...")
        self.scanner.run_htf_scan()
        self._last_htf_scan = time.time()

    def stop(self):
        """Stop all components."""
        logger.info("Stopping paper trader...")
        self.is_running = False

        # Save final snapshot
        summary = self.trader.get_portfolio_summary()
        db.save_portfolio_snapshot(
            balance=summary['balance'],
            equity=summary['equity'],
            unrealized_pnl=summary['unrealized_pnl'],
            open_positions=summary['open_positions'],
            total_trades=summary['total_trades'],
            win_rate=summary['win_rate'],
            profit_factor=summary['profit_factor']
        )

        logger.info("Paper trader stopped")

    def run_forever(self):
        """Run until interrupted."""
        self.setup()
        self.start()

        def signal_handler(sig, frame):
            logger.info("\nShutdown signal received")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Paper trader running. Press Ctrl+C to stop.")

        # Keep main thread alive
        while self.is_running:
            try:
                # Print status every 5 minutes
                time.sleep(300)

                summary = self.trader.get_portfolio_summary()
                stats = self.scanner.get_statistics()

                logger.info("-" * 40)
                logger.info("STATUS UPDATE")
                logger.info(f"Equity: ${summary['equity']:,.2f}")
                logger.info(f"P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_percent']:.2f}%)")
                logger.info(f"Open Positions: {summary['open_positions']}")
                logger.info(f"Active POIs: {stats['total_pois']}")
                logger.info("-" * 40)

            except Exception as e:
                logger.error(f"Status update error: {e}")


# Create orchestrator instance
orchestrator = PaperTraderOrchestrator()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="KuCoin Multiscanner Papertrader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Run the paper trader
  python main.py --dashboard  # Run with dashboard
  python main.py --scan-only  # Run single scan and exit
        """
    )

    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Run with terminal dashboard"
    )
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
        "--close-shorts",
        action="store_true",
        help="Close all SHORT positions and exit"
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="Close all positions and exit"
    )

    args = parser.parse_args()

    if args.stats:
        from dashboard import stats_viewer
        stats_viewer.show_portfolio_details()
        stats_viewer.show_database_stats()
        return

    if args.close_shorts:
        logger.info("Closing all SHORT positions...")
        # Need to initialize and load current prices
        db.init_database()
        scanner.initialize()

        # Get current prices for positions
        for symbol in list(paper_trader.portfolio.positions.keys()):
            try:
                ticker = kucoin_client.get_ticker(symbol)
                paper_trader.update_price(symbol, ticker.price)
            except Exception as e:
                logger.error(f"Could not get price for {symbol}: {e}")

        closed = paper_trader.close_all_short_positions()
        if closed:
            logger.info(f"Closed {len(closed)} SHORT positions")
            for pos in closed:
                print(f"  Closed {pos.symbol}: P&L ${pos.pnl:.2f} ({pos.pnl_percent:.2f}%)")
        else:
            logger.info("No SHORT positions to close")
        return

    if args.close_all:
        logger.info("Closing all positions...")
        db.init_database()
        scanner.initialize()

        for symbol in list(paper_trader.portfolio.positions.keys()):
            try:
                ticker = kucoin_client.get_ticker(symbol)
                paper_trader.update_price(symbol, ticker.price)
            except Exception as e:
                logger.error(f"Could not get price for {symbol}: {e}")

        closed = paper_trader.close_all_positions("MANUAL_CLOSE_ALL")
        if closed:
            logger.info(f"Closed {len(closed)} positions")
            for pos in closed:
                print(f"  Closed {pos.symbol}: P&L ${pos.pnl:.2f} ({pos.pnl_percent:.2f}%)")
        else:
            logger.info("No positions to close")
        return

    if args.scan_only:
        logger.info("Running single scan...")
        db.init_database()
        scanner.initialize()
        results = scanner.run_htf_scan()

        print(f"\nFound {len(results)} symbols with setups:")
        for symbol, result in results.items():
            print(f"  {symbol}: {len(result.pois)} POIs, Score: {result.score}")

        return

    if args.dashboard:
        from dashboard import dashboard
        # Start orchestrator in background
        orchestrator.setup()
        orchestrator.start()
        # Run dashboard
        dashboard.run()
    else:
        orchestrator.run_forever()


if __name__ == "__main__":
    main()
