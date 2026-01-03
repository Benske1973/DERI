#!/usr/bin/env python3
# debug.py - Debugging and Diagnostic Tool
"""
Diagnostic tool for the KuCoin Multiscanner Papertrader.
Use this to check system state, identify issues, and run tests.
"""

import sys
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger("debug")

def test_imports():
    """Test all module imports."""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)

    modules = [
        ("config", "config, TimeFrame, TrendDirection, PositionStatus"),
        ("kucoin_client", "kucoin_client, KuCoinClient"),
        ("indicators", "Indicators"),
        ("scanner", "scanner, MultiScanner, ScanResult"),
        ("paper_trader", "paper_trader, PaperTrader"),
        ("strategies", "strategy, TradingStrategy"),
        ("websocket_feed", "ws_feed, KuCoinWebSocket"),
        ("database", "db, Database"),
        ("dashboard", "dashboard, Dashboard"),
    ]

    errors = []
    for module, imports in modules:
        try:
            exec(f"from {module} import {imports}")
            print(f"  [OK] {module}")
        except Exception as e:
            print(f"  [FAIL] {module}: {e}")
            errors.append((module, str(e)))

    return errors


def test_kucoin_api():
    """Test KuCoin API connectivity."""
    print("\n" + "="*60)
    print("TESTING KUCOIN API")
    print("="*60)

    from kucoin_client import kucoin_client

    # Test server time
    try:
        server_time = kucoin_client.get_server_time()
        print(f"  [OK] Server time: {datetime.fromtimestamp(server_time/1000)}")
    except Exception as e:
        print(f"  [FAIL] Server time: {e}")
        return False

    # Test symbols
    try:
        symbols = kucoin_client.get_usdt_pairs()
        print(f"  [OK] USDT pairs: {len(symbols)} found")
    except Exception as e:
        print(f"  [FAIL] Get symbols: {e}")
        return False

    # Test candles
    try:
        from config import TimeFrame
        df = kucoin_client.get_candles("BTC-USDT", TimeFrame.H4, limit=10)
        print(f"  [OK] Candles: {len(df)} BTC-USDT 4H candles")
    except Exception as e:
        print(f"  [FAIL] Get candles: {e}")
        return False

    # Test ticker
    try:
        ticker = kucoin_client.get_ticker("BTC-USDT")
        print(f"  [OK] Ticker: BTC-USDT @ ${ticker.price:,.2f}")
    except Exception as e:
        print(f"  [FAIL] Get ticker: {e}")
        return False

    return True


def test_database():
    """Test database connectivity."""
    print("\n" + "="*60)
    print("TESTING DATABASE")
    print("="*60)

    from database import db

    try:
        db.init_database()
        print("  [OK] Database initialized")
    except Exception as e:
        print(f"  [FAIL] Database init: {e}")
        return False

    # Test write/read
    try:
        db.save_signal(
            symbol="TEST-USDT",
            trend="BULLISH",
            fvg_top=100.0,
            fvg_bottom=95.0,
            ob_top=100.0,
            ob_bottom=95.0,
            status="TESTING"
        )
        signal = db.get_signal("TEST-USDT")
        if signal:
            print("  [OK] Write/Read test passed")
            # Clean up
            db.update_signal_status("TEST-USDT", "DELETED")
        else:
            print("  [FAIL] Could not read test signal")
    except Exception as e:
        print(f"  [FAIL] Write/Read test: {e}")
        return False

    return True


def test_indicators():
    """Test indicator calculations."""
    print("\n" + "="*60)
    print("TESTING INDICATORS")
    print("="*60)

    from kucoin_client import kucoin_client
    from config import TimeFrame
    from indicators import Indicators

    try:
        df = kucoin_client.get_candles("BTC-USDT", TimeFrame.H4, limit=200)
        if df.empty:
            print("  [FAIL] No candle data")
            return False

        ind = Indicators(df)

        # Test EMAs
        ema = ind.ema(50)
        print(f"  [OK] EMA(50): {ema.iloc[-1]:.2f}")

        # Test RSI
        rsi = ind.rsi()
        print(f"  [OK] RSI: {rsi.iloc[-1]:.2f}")

        # Test ATR
        atr = ind.atr()
        print(f"  [OK] ATR: {atr.iloc[-1]:.2f}")

        # Test trend
        trend = ind.get_trend()
        print(f"  [OK] Trend: {trend.value}")

        # Test FVG
        fvgs = ind.find_fvg()
        print(f"  [OK] FVGs found: {len(fvgs)}")

        # Test OB
        obs = ind.find_order_blocks()
        print(f"  [OK] Order Blocks found: {len(obs)}")

        # Test score
        score = ind.calculate_signal_score("LONG")
        print(f"  [OK] Signal score: {score.total_score:.1f}")

        return True

    except Exception as e:
        print(f"  [FAIL] Indicator test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scanner():
    """Test scanner functionality."""
    print("\n" + "="*60)
    print("TESTING SCANNER")
    print("="*60)

    from scanner import scanner

    try:
        symbols = scanner.initialize()
        print(f"  [OK] Scanner initialized with {len(symbols)} symbols")
    except Exception as e:
        print(f"  [FAIL] Scanner init: {e}")
        return False

    # Test single symbol scan
    try:
        result = scanner.scan_symbol_htf("BTC-USDT")
        if result:
            print(f"  [OK] BTC-USDT scan: {len(result.pois)} POIs, Score: {result.score}")
        else:
            print("  [WARN] BTC-USDT scan returned no result (might be no setup)")
    except Exception as e:
        print(f"  [FAIL] Symbol scan: {e}")
        return False

    return True


def test_paper_trader():
    """Test paper trader functionality."""
    print("\n" + "="*60)
    print("TESTING PAPER TRADER")
    print("="*60)

    from paper_trader import paper_trader

    print(f"  Initial Capital: ${paper_trader.portfolio.initial_capital:,.2f}")
    print(f"  Current Balance: ${paper_trader.portfolio.balance:,.2f}")
    print(f"  Current Equity: ${paper_trader.portfolio.equity:,.2f}")
    print(f"  Open Positions: {len(paper_trader.portfolio.positions)}")

    # List open positions
    if paper_trader.portfolio.positions:
        print("\n  Open Positions:")
        for symbol, pos in paper_trader.portfolio.positions.items():
            print(f"    - {symbol}: {pos.side} @ {pos.entry_price:.4f}, PnL: ${pos.unrealized_pnl:.2f}")

    # Test position check
    can_open = paper_trader.can_open_position("TEST-USDT")
    print(f"\n  Can open new position: {can_open}")

    return True


def test_strategy():
    """Test strategy signal generation."""
    print("\n" + "="*60)
    print("TESTING STRATEGY")
    print("="*60)

    from strategies import strategy
    from scanner import scanner, ScanResult, POI
    from indicators import Indicators
    from kucoin_client import kucoin_client
    from config import TimeFrame, TrendDirection, PositionStatus
    from datetime import datetime

    try:
        # Get data
        df = kucoin_client.get_candles("BTC-USDT", TimeFrame.M15, limit=100)
        if df.empty:
            print("  [FAIL] No candle data")
            return False

        ind = Indicators(df)
        current_price = df["close"].iloc[-1]

        # Create mock scan result
        mock_poi = POI(
            symbol="BTC-USDT",
            poi_type="FVG",
            direction="BULLISH",
            top=current_price * 1.01,
            bottom=current_price * 0.99,
            midpoint=current_price,
            timeframe="4hour"
        )

        mock_result = ScanResult(
            symbol="BTC-USDT",
            timestamp=datetime.now(),
            current_price=current_price,
            trend=ind.get_trend(),
            htf_bias="BULLISH",
            ltf_confirmation=True,
            pois=[mock_poi],
            fvgs=[],
            order_blocks=[],
            structure=ind.analyze_market_structure(),
            score=50,
            analysis={}
        )

        # Test signal generation
        result = strategy.generate_signal(mock_result, mock_poi, ind, current_price)

        print(f"  Signal generated: {result.has_signal}")
        print(f"  Reason: {result.reason}")
        print(f"  Checks passed: {result.checks_passed}")
        print(f"  Checks failed: {result.checks_failed}")

        return True

    except Exception as e:
        print(f"  [FAIL] Strategy test: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_system_state():
    """Show current system state."""
    print("\n" + "="*60)
    print("CURRENT SYSTEM STATE")
    print("="*60)

    from paper_trader import paper_trader
    from scanner import scanner
    from database import db

    # Portfolio
    print("\n  PORTFOLIO:")
    print(f"    Balance: ${paper_trader.portfolio.balance:,.2f}")
    print(f"    Equity: ${paper_trader.portfolio.equity:,.2f}")
    print(f"    Total P&L: ${paper_trader.portfolio.total_pnl:,.2f}")
    print(f"    Open Positions: {len(paper_trader.portfolio.positions)}")
    print(f"    Trade History: {len(paper_trader.portfolio.trade_history)} trades")

    # Open positions details
    if paper_trader.portfolio.positions:
        print("\n  OPEN POSITIONS:")
        for symbol, pos in paper_trader.portfolio.positions.items():
            print(f"    {symbol}: {pos.side} @ {pos.entry_price:.6f}")
            print(f"      Current: {pos.current_price:.6f}")
            print(f"      PnL: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_percent:.2f}%)")
            print(f"      SL: {pos.stop_loss:.6f}, TP: {pos.take_profit:.6f}")

    # Scanner
    print("\n  SCANNER:")
    print(f"    Symbols: {len(scanner.symbols)}")
    print(f"    Active POIs: {sum(len(pois) for pois in scanner.active_pois.values())}")
    print(f"    Scan results: {len(scanner.scan_results)}")

    # Database stats
    print("\n  DATABASE:")
    stats = db.get_trade_statistics()
    print(f"    Total trades: {stats['total_trades']}")
    print(f"    Win rate: {stats['win_rate']:.1f}%")
    print(f"    Total P&L: ${stats['total_pnl']:.2f}")


def run_full_diagnostics():
    """Run all diagnostic tests."""
    print("\n" + "="*60)
    print("KUCOIN PAPERTRADER DIAGNOSTICS")
    print(f"Time: {datetime.now()}")
    print("="*60)

    results = []

    # Test imports
    errors = test_imports()
    results.append(("Imports", len(errors) == 0))

    if errors:
        print("\n  Import errors detected. Cannot continue.")
        return False

    # Test API
    results.append(("KuCoin API", test_kucoin_api()))

    # Test Database
    results.append(("Database", test_database()))

    # Test Indicators
    results.append(("Indicators", test_indicators()))

    # Test Scanner
    results.append(("Scanner", test_scanner()))

    # Test Paper Trader
    results.append(("Paper Trader", test_paper_trader()))

    # Test Strategy
    results.append(("Strategy", test_strategy()))

    # Show state
    show_system_state()

    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  All tests passed! System is ready.")
    else:
        print("\n  Some tests failed. Please check the errors above.")

    return all_passed


def reset_portfolio():
    """Reset the paper trading portfolio."""
    print("\n" + "="*60)
    print("RESETTING PORTFOLIO")
    print("="*60)

    from paper_trader import paper_trader
    from config import config

    # Clear positions
    paper_trader.portfolio.positions.clear()
    paper_trader.portfolio.orders.clear()
    paper_trader.portfolio.trade_history.clear()

    # Reset balance
    paper_trader.portfolio.balance = config.risk.initial_capital
    paper_trader.portfolio.equity = config.risk.initial_capital
    paper_trader.portfolio.margin_used = 0

    print(f"  Portfolio reset to ${config.risk.initial_capital:,.2f}")
    print("  All positions and history cleared.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Papertrader Diagnostics")
    parser.add_argument("--full", action="store_true", help="Run full diagnostics")
    parser.add_argument("--state", action="store_true", help="Show current state")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio")
    parser.add_argument("--api", action="store_true", help="Test API only")
    parser.add_argument("--indicators", action="store_true", help="Test indicators only")

    args = parser.parse_args()

    if args.full:
        run_full_diagnostics()
    elif args.state:
        test_imports()
        show_system_state()
    elif args.reset:
        test_imports()
        reset_portfolio()
    elif args.api:
        test_imports()
        test_kucoin_api()
    elif args.indicators:
        test_imports()
        test_indicators()
    else:
        run_full_diagnostics()
