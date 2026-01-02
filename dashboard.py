# dashboard.py - Dashboard and Statistics Viewer
"""
Terminal-based dashboard for monitoring the paper trader.
Displays real-time statistics, positions, and signals.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional

from config import config
from database import db
from paper_trader import paper_trader
from scanner import scanner
from websocket_feed import ws_feed

logger = logging.getLogger(__name__)

class Dashboard:
    """Terminal dashboard for paper trading."""

    def __init__(self):
        self.refresh_rate = 2  # seconds

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def format_currency(self, value: float) -> str:
        """Format currency value."""
        return f"${value:,.2f}"

    def format_percent(self, value: float) -> str:
        """Format percentage value."""
        sign = "+" if value > 0 else ""
        return f"{sign}{value:.2f}%"

    def format_price(self, value: float) -> str:
        """Format price based on magnitude."""
        if value >= 1000:
            return f"{value:,.2f}"
        elif value >= 1:
            return f"{value:.4f}"
        else:
            return f"{value:.6f}"

    def get_color(self, value: float) -> str:
        """Get ANSI color code based on value."""
        if value > 0:
            return "\033[92m"  # Green
        elif value < 0:
            return "\033[91m"  # Red
        return "\033[0m"  # Reset

    def reset_color(self) -> str:
        """Reset ANSI color."""
        return "\033[0m"

    def draw_header(self) -> str:
        """Draw dashboard header."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "KUCOIN MULTISCANNER PAPERTRADER"

        lines = [
            "=" * 80,
            f"{title:^80}",
            f"{'Last Update: ' + now:^80}",
            "=" * 80,
            ""
        ]
        return "\n".join(lines)

    def draw_portfolio_summary(self) -> str:
        """Draw portfolio summary section."""
        summary = paper_trader.get_portfolio_summary()

        balance = summary['balance']
        equity = summary['equity']
        total_pnl = summary['total_pnl']
        unrealized = summary['unrealized_pnl']
        pnl_color = self.get_color(total_pnl)
        unrealized_color = self.get_color(unrealized)

        fees_paid = summary.get('total_fees_paid', 0)

        lines = [
            "PORTFOLIO SUMMARY",
            "-" * 40,
            f"Initial Capital:   {self.format_currency(summary['initial_capital'])}",
            f"Balance:           {self.format_currency(balance)}",
            f"Equity:            {self.format_currency(equity)}",
            f"Total P&L:         {pnl_color}{self.format_currency(total_pnl)} ({self.format_percent(summary['total_pnl_percent'])}){self.reset_color()}",
            f"Unrealized P&L:    {unrealized_color}{self.format_currency(unrealized)}{self.reset_color()}",
            f"Fees Paid:         {self.format_currency(fees_paid)}",
            "",
            f"Open Positions:    {summary['open_positions']}",
            f"Total Trades:      {summary['total_trades']}",
            f"Win Rate:          {summary['win_rate']:.1f}%",
            f"Profit Factor:     {summary['profit_factor']:.2f}",
            ""
        ]
        return "\n".join(lines)

    def draw_open_positions(self) -> str:
        """Draw open positions section."""
        positions = paper_trader.get_open_positions()

        lines = [
            "OPEN POSITIONS",
            "-" * 80,
            f"{'Symbol':<12} {'Side':<6} {'Entry':<12} {'Current':<12} {'Size':<10} {'P&L':<12} {'P&L %':<10}",
            "-" * 80
        ]

        if not positions:
            lines.append("No open positions")
        else:
            for pos in positions:
                pnl = pos['unrealized_pnl']
                pnl_pct = pos['unrealized_pnl_percent']
                color = self.get_color(pnl)

                lines.append(
                    f"{pos['symbol']:<12} "
                    f"{pos['side']:<6} "
                    f"{self.format_price(pos['entry_price']):<12} "
                    f"{self.format_price(pos['current_price']):<12} "
                    f"{pos['quantity']:<10.4f} "
                    f"{color}{self.format_currency(pnl):<12}{self.reset_color()} "
                    f"{color}{self.format_percent(pnl_pct):<10}{self.reset_color()}"
                )

        lines.append("")
        return "\n".join(lines)

    def draw_active_signals(self) -> str:
        """Draw active signals section."""
        setups = scanner.get_active_setups()

        lines = [
            "ACTIVE SIGNALS / POIs",
            "-" * 80,
            f"{'Symbol':<12} {'Direction':<10} {'Type':<6} {'Zone Top':<12} {'Zone Bottom':<12} {'Status':<10} {'Score':<6}",
            "-" * 80
        ]

        if not setups:
            lines.append("No active signals")
        else:
            for setup in setups[:10]:  # Show top 10
                lines.append(
                    f"{setup['symbol']:<12} "
                    f"{setup['direction']:<10} "
                    f"{setup['poi_type']:<6} "
                    f"{self.format_price(setup['zone_top']):<12} "
                    f"{self.format_price(setup['zone_bottom']):<12} "
                    f"{setup['status']:<10} "
                    f"{setup['score']:<6.0f}"
                )

        lines.append("")
        return "\n".join(lines)

    def draw_recent_trades(self) -> str:
        """Draw recent trades section."""
        trades = paper_trader.get_trade_history(limit=5)

        lines = [
            "RECENT TRADES",
            "-" * 80,
            f"{'Symbol':<12} {'Side':<6} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'P&L %':<8} {'Reason':<12}",
            "-" * 80
        ]

        if not trades:
            lines.append("No trade history")
        else:
            for trade in trades:
                pnl = trade['pnl']
                color = self.get_color(pnl)

                lines.append(
                    f"{trade['symbol']:<12} "
                    f"{trade['side']:<6} "
                    f"{self.format_price(trade['entry_price']):<10} "
                    f"{self.format_price(trade['close_price'] or 0):<10} "
                    f"{color}{self.format_currency(pnl):<12}{self.reset_color()} "
                    f"{color}{self.format_percent(trade['pnl_percent']):<8}{self.reset_color()} "
                    f"{trade['close_reason'] or 'N/A':<12}"
                )

        lines.append("")
        return "\n".join(lines)

    def draw_scanner_stats(self) -> str:
        """Draw scanner statistics."""
        stats = scanner.get_statistics()

        lines = [
            "SCANNER STATUS",
            "-" * 40,
            f"Total Symbols:        {stats['total_symbols']}",
            f"Symbols with Setups:  {stats['symbols_with_setups']}",
            f"Total POIs:           {stats['total_pois']}",
            f"  - Bullish:          {stats['bullish_pois']}",
            f"  - Bearish:          {stats['bearish_pois']}",
            f"Last HTF Scan:        {stats['last_htf_scan'] or 'Never'}",
            ""
        ]
        return "\n".join(lines)

    def draw_websocket_stats(self) -> str:
        """Draw WebSocket statistics."""
        stats = ws_feed.get_stats()

        status = "CONNECTED" if stats['connected'] else "DISCONNECTED"
        status_color = "\033[92m" if stats['connected'] else "\033[91m"

        lines = [
            "WEBSOCKET STATUS",
            "-" * 40,
            f"Status:              {status_color}{status}{self.reset_color()}",
            f"Messages Received:   {stats['messages_received']:,}",
            f"Symbols Tracking:    {stats['symbols_tracked']}",
            f"Reconnects:          {stats['reconnect_count']}",
            f"Last Message:        {stats['last_message'] or 'Never'}",
            ""
        ]
        return "\n".join(lines)

    def draw_footer(self) -> str:
        """Draw dashboard footer."""
        lines = [
            "=" * 80,
            "Press Ctrl+C to exit | Commands: [S]can | [R]efresh | [Q]uit",
            "=" * 80
        ]
        return "\n".join(lines)

    def render(self) -> str:
        """Render complete dashboard."""
        sections = [
            self.draw_header(),
            self.draw_portfolio_summary(),
            self.draw_open_positions(),
            self.draw_active_signals(),
            self.draw_recent_trades(),
            self.draw_scanner_stats(),
            self.draw_websocket_stats(),
            self.draw_footer()
        ]
        return "\n".join(sections)

    def run(self):
        """Run dashboard loop."""
        print("Starting dashboard...")

        try:
            while True:
                self.clear_screen()
                print(self.render())
                time.sleep(self.refresh_rate)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")


class StatisticsViewer:
    """Detailed statistics viewer."""

    def __init__(self):
        pass

    def show_portfolio_details(self):
        """Show detailed portfolio statistics."""
        summary = paper_trader.get_portfolio_summary()

        print("\n" + "=" * 60)
        print("DETAILED PORTFOLIO STATISTICS")
        print("=" * 60)

        print(f"\nCapital:")
        print(f"  Initial:           ${summary['initial_capital']:,.2f}")
        print(f"  Current Balance:   ${summary['balance']:,.2f}")
        print(f"  Equity:            ${summary['equity']:,.2f}")

        pnl_pct = summary['total_pnl_percent']
        print(f"\nPerformance:")
        print(f"  Total P&L:         ${summary['total_pnl']:,.2f} ({pnl_pct:+.2f}%)")
        print(f"  Unrealized P&L:    ${summary['unrealized_pnl']:,.2f}")

        print(f"\nTrading Statistics:")
        print(f"  Total Trades:      {summary['total_trades']}")
        print(f"  Winning Trades:    {summary['winning_trades']}")
        print(f"  Losing Trades:     {summary['losing_trades']}")
        print(f"  Win Rate:          {summary['win_rate']:.1f}%")

        print(f"\nRisk Metrics:")
        print(f"  Profit Factor:     {summary['profit_factor']:.2f}")
        print(f"  Avg Win:           ${summary['avg_win']:,.2f}")
        print(f"  Avg Loss:          ${summary['avg_loss']:,.2f}")
        print(f"  Largest Win:       ${summary['largest_win']:,.2f}")
        print(f"  Largest Loss:      ${summary['largest_loss']:,.2f}")

        print("\n" + "=" * 60)

    def show_trade_history(self, limit: int = 20):
        """Show trade history."""
        trades = paper_trader.get_trade_history(limit=limit)

        print("\n" + "=" * 80)
        print("TRADE HISTORY")
        print("=" * 80)
        print(f"{'ID':<8} {'Symbol':<12} {'Side':<6} {'Entry':<12} {'Exit':<12} {'P&L':>12} {'Reason':<15}")
        print("-" * 80)

        for trade in trades:
            pnl = trade['pnl']
            pnl_str = f"${pnl:+,.2f}"

            print(
                f"{trade['id']:<8} "
                f"{trade['symbol']:<12} "
                f"{trade['side']:<6} "
                f"{trade['entry_price']:<12.4f} "
                f"{(trade['close_price'] or 0):<12.4f} "
                f"{pnl_str:>12} "
                f"{trade['close_reason'] or 'N/A':<15}"
            )

        print("=" * 80)

    def show_database_stats(self):
        """Show database statistics."""
        stats = db.get_trade_statistics()

        print("\n" + "=" * 60)
        print("DATABASE STATISTICS")
        print("=" * 60)

        print(f"\nTrades:")
        print(f"  Total:         {stats['total_trades']}")
        print(f"  Winners:       {stats['winning_trades']}")
        print(f"  Losers:        {stats['losing_trades']}")
        print(f"  Win Rate:      {stats['win_rate']:.1f}%")

        print(f"\nPerformance:")
        print(f"  Total P&L:     ${stats['total_pnl']:,.2f}")
        print(f"  Avg Win:       ${stats['avg_win']:,.2f}")
        print(f"  Avg Loss:      ${stats['avg_loss']:,.2f}")
        print(f"  Expectancy:    ${stats['expectancy']:,.2f}")

        print(f"\nBest/Worst:")
        print(f"  Best Trade:    ${stats['best_trade']:,.2f}")
        print(f"  Worst Trade:   ${stats['worst_trade']:,.2f}")

        print("=" * 60)


# Singleton instances
dashboard = Dashboard()
stats_viewer = StatisticsViewer()


def show_status():
    """Show current status (legacy compatibility)."""
    print("\n--- ACTIVE POIs (Waiting for Tap) ---")
    signals = db.get_signals_by_status('SCANNING')
    for s in signals:
        print(f"Coin: {s['symbol']} | Trend: {s['trend']} | Zone: {s['fvg_bottom']:.4f} - {s['fvg_top']:.4f}")

    print("\n--- TAPPED COINS (Waiting for Confirmation) ---")
    tapped = db.get_signals_by_status('TAPPED')
    for s in tapped:
        print(f" {s['symbol']} entered the zone!")


def show_stats():
    """Show statistics (legacy compatibility)."""
    trades = db.get_trade_history()

    print(f"{'ID':<4} | {'Symbol':<10} | {'Entry':<10} | {'SL':<10} | {'TP':<10} | {'Date'}")
    print("-" * 70)

    for t in trades:
        print(
            f"{t['id']:<4} | "
            f"{t['symbol']:<10} | "
            f"{t['entry_price']:<10.4f} | "
            f"{t.get('stop_loss', 0):<10.4f} | "
            f"{t.get('take_profit', 0):<10.4f} | "
            f"{t['opened_at']}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trader Dashboard")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--history", action="store_true", help="Show trade history")
    parser.add_argument("--status", action="store_true", help="Show current status")

    args = parser.parse_args()

    if args.stats:
        stats_viewer.show_portfolio_details()
        stats_viewer.show_database_stats()
    elif args.history:
        stats_viewer.show_trade_history()
    elif args.status:
        show_status()
    else:
        dashboard.run()
