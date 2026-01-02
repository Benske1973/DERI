# database.py - Database Module
"""
SQLite database for persistent storage of:
- Signals and POIs
- Trade logs
- Portfolio snapshots
- Performance metrics
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from config import config

logger = logging.getLogger(__name__)

class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_database(self):
        """Initialize database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Signals table (POIs)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trend TEXT,
                    poi_type TEXT,
                    fvg_top REAL,
                    fvg_bottom REAL,
                    ob_top REAL,
                    ob_bottom REAL,
                    status TEXT DEFAULT 'SCANNING',
                    score REAL DEFAULT 0,
                    timeframe TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tapped_at DATETIME,
                    confirmed_at DATETIME,
                    metadata TEXT,
                    UNIQUE(symbol, fvg_top, fvg_bottom)
                )
            ''')

            # Trade log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL DEFAULT 0,
                    pnl_percent REAL DEFAULT 0,
                    status TEXT DEFAULT 'OPEN',
                    close_reason TEXT,
                    signal_id INTEGER,
                    opened_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    closed_at DATETIME,
                    metadata TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals(id)
                )
            ''')

            # Portfolio snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    balance REAL,
                    equity REAL,
                    unrealized_pnl REAL,
                    open_positions INTEGER,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL
                )
            ''')

            # Daily statistics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE UNIQUE,
                    starting_balance REAL,
                    ending_balance REAL,
                    daily_pnl REAL,
                    trades_count INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    largest_win REAL,
                    largest_loss REAL
                )
            ''')

            # Price history (for backtesting)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT,
                    timestamp INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')

            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)')

            logger.info("Database initialized successfully")

    # ==================== SIGNALS ====================

    def save_signal(self, symbol: str, trend: str, fvg_top: float, fvg_bottom: float,
                    ob_top: float, ob_bottom: float, status: str = "SCANNING",
                    poi_type: str = "FVG", score: float = 0, timeframe: str = None,
                    metadata: Dict = None) -> int:
        """Save or update a signal."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO signals
                (symbol, trend, poi_type, fvg_top, fvg_bottom, ob_top, ob_bottom,
                 status, score, timeframe, metadata, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, trend, poi_type, fvg_top, fvg_bottom, ob_top, ob_bottom,
                  status, score, timeframe, json.dumps(metadata) if metadata else None))

            return cursor.lastrowid

    def get_signals_by_status(self, status: str) -> List[Dict]:
        """Get signals by status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals WHERE status = ? ORDER BY score DESC
            ''', (status,))

            return [dict(row) for row in cursor.fetchall()]

    def get_signal(self, symbol: str) -> Optional[Dict]:
        """Get signal for a symbol."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM signals WHERE symbol = ?', (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_signal_status(self, symbol: str, status: str):
        """Update signal status."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            update_field = ""
            if status == "TAPPED":
                update_field = ", tapped_at = CURRENT_TIMESTAMP"
            elif status == "CONFIRMED":
                update_field = ", confirmed_at = CURRENT_TIMESTAMP"

            cursor.execute(f'''
                UPDATE signals
                SET status = ?, updated_at = CURRENT_TIMESTAMP {update_field}
                WHERE symbol = ?
            ''', (status, symbol))

    def get_active_signals(self) -> List[Dict]:
        """Get all active signals (SCANNING or TAPPED)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals
                WHERE status IN ('SCANNING', 'TAPPED')
                ORDER BY score DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def clear_old_signals(self, hours: int = 24):
        """Clear signals older than specified hours."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM signals
                WHERE created_at < datetime('now', ? || ' hours')
                AND status IN ('SCANNING', 'TAPPED')
            ''', (f'-{hours}',))

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleared {deleted} old signals")

    # ==================== TRADES ====================

    def save_trade(self, trade_id: str, symbol: str, side: str, entry_price: float,
                   quantity: float, stop_loss: float = None, take_profit: float = None,
                   signal_id: int = None, metadata: Dict = None) -> int:
        """Save a new trade."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trades
                (trade_id, symbol, side, entry_price, quantity, stop_loss, take_profit,
                 signal_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (trade_id, symbol, side, entry_price, quantity, stop_loss, take_profit,
                  signal_id, json.dumps(metadata) if metadata else None))

            return cursor.lastrowid

    def close_trade(self, trade_id: str, exit_price: float, pnl: float,
                    pnl_percent: float, close_reason: str = "MANUAL"):
        """Close a trade."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE trades
                SET exit_price = ?, pnl = ?, pnl_percent = ?,
                    status = 'CLOSED', close_reason = ?, closed_at = CURRENT_TIMESTAMP
                WHERE trade_id = ?
            ''', (exit_price, pnl, pnl_percent, close_reason, trade_id))

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades WHERE status = 'OPEN' ORDER BY opened_at DESC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM trades
                WHERE status = 'CLOSED'
                ORDER BY closed_at DESC
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_trade_statistics(self) -> Dict:
        """Get trade statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total trades
            cursor.execute('SELECT COUNT(*) FROM trades WHERE status = "CLOSED"')
            total = cursor.fetchone()[0]

            # Wins
            cursor.execute('SELECT COUNT(*) FROM trades WHERE status = "CLOSED" AND pnl > 0')
            wins = cursor.fetchone()[0]

            # Total P&L
            cursor.execute('SELECT SUM(pnl) FROM trades WHERE status = "CLOSED"')
            total_pnl = cursor.fetchone()[0] or 0

            # Average win/loss
            cursor.execute('SELECT AVG(pnl) FROM trades WHERE status = "CLOSED" AND pnl > 0')
            avg_win = cursor.fetchone()[0] or 0

            cursor.execute('SELECT AVG(pnl) FROM trades WHERE status = "CLOSED" AND pnl < 0')
            avg_loss = cursor.fetchone()[0] or 0

            # Best/Worst trades
            cursor.execute('SELECT MAX(pnl) FROM trades WHERE status = "CLOSED"')
            best = cursor.fetchone()[0] or 0

            cursor.execute('SELECT MIN(pnl) FROM trades WHERE status = "CLOSED"')
            worst = cursor.fetchone()[0] or 0

            return {
                "total_trades": total,
                "winning_trades": wins,
                "losing_trades": total - wins,
                "win_rate": (wins / total * 100) if total > 0 else 0,
                "total_pnl": total_pnl,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "best_trade": best,
                "worst_trade": worst,
                "expectancy": ((wins / total) * avg_win + ((total - wins) / total) * avg_loss)
                if total > 0 else 0
            }

    # ==================== PORTFOLIO ====================

    def save_portfolio_snapshot(self, balance: float, equity: float,
                                unrealized_pnl: float, open_positions: int,
                                total_trades: int, win_rate: float,
                                profit_factor: float):
        """Save portfolio snapshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_snapshots
                (balance, equity, unrealized_pnl, open_positions,
                 total_trades, win_rate, profit_factor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (balance, equity, unrealized_pnl, open_positions,
                  total_trades, win_rate, profit_factor))

    def get_portfolio_history(self, days: int = 30) -> List[Dict]:
        """Get portfolio history."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM portfolio_snapshots
                WHERE timestamp > datetime('now', ? || ' days')
                ORDER BY timestamp
            ''', (f'-{days}',))
            return [dict(row) for row in cursor.fetchall()]

    def update_daily_stats(self, date: str, starting_balance: float,
                           ending_balance: float, trades: List[Dict]):
        """Update daily statistics."""
        daily_pnl = ending_balance - starting_balance
        wins = len([t for t in trades if t.get('pnl', 0) > 0])
        losses = len(trades) - wins
        largest_win = max([t.get('pnl', 0) for t in trades], default=0)
        largest_loss = min([t.get('pnl', 0) for t in trades], default=0)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO daily_stats
                (date, starting_balance, ending_balance, daily_pnl,
                 trades_count, wins, losses, largest_win, largest_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (date, starting_balance, ending_balance, daily_pnl,
                  len(trades), wins, losses, largest_win, largest_loss))

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """Get daily statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM daily_stats
                WHERE date > date('now', ? || ' days')
                ORDER BY date
            ''', (f'-{days}',))
            return [dict(row) for row in cursor.fetchall()]


# Singleton instance
db = Database()


# Legacy compatibility
def setup():
    """Legacy setup function."""
    db.init_database()
    print("Database initialized.")


if __name__ == "__main__":
    setup()
