# trade_monitor.py - TP/SL Monitor voor open trades
import sqlite3
import requests
import time
from datetime import datetime
from config import (
    KUCOIN_BASE_URL, MONITOR_INTERVAL, DATABASE_PATH,
    API_RATE_LIMIT, MAX_RETRIES, RETRY_DELAY
)
from logger import get_monitor_logger

log = get_monitor_logger()


def get_current_price(symbol: str) -> float:
    """Haal huidige prijs op voor een symbol."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_RATE_LIMIT)
            url = f"{KUCOIN_BASE_URL}/api/v1/market/orderbook/level1?symbol={symbol}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == '200000' and data.get('data'):
                return float(data['data'].get('price', 0))

        except Exception as e:
            log.warning(f"Price fetch failed for {symbol} (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    return 0


def check_trade_exit(trade: dict, current_price: float) -> tuple:
    """
    Check of trade TP of SL heeft geraakt.
    Returns: (hit_type, exit_price) of (None, None)
    """
    direction = trade['direction']
    entry = trade['entry_price']
    sl = trade['sl']
    tp = trade['tp']

    if direction == 'BULLISH':
        # Long trade
        if current_price >= tp:
            return 'TP', current_price
        elif current_price <= sl:
            return 'SL', current_price

    elif direction == 'BEARISH':
        # Short trade
        if current_price <= tp:
            return 'TP', current_price
        elif current_price >= sl:
            return 'SL', current_price

    return None, None


def calculate_pnl(trade: dict, exit_price: float) -> tuple:
    """Bereken PnL voor een trade."""
    entry = trade['entry_price']
    direction = trade['direction']

    if direction == 'BULLISH':
        pnl_percent = ((exit_price - entry) / entry) * 100
    else:  # BEARISH
        pnl_percent = ((entry - exit_price) / entry) * 100

    return pnl_percent


def close_trade(conn: sqlite3.Connection, trade_id: int, exit_price: float,
                result: str, pnl_percent: float):
    """Sluit een trade in de database."""
    c = conn.cursor()

    c.execute("""
        UPDATE trade_log
        SET status = 'CLOSED',
            exit_price = ?,
            exit_timestamp = datetime('now'),
            result = ?,
            pnl_percent = ?
        WHERE id = ?
    """, (exit_price, result, pnl_percent, trade_id))

    # Update ook de signals tabel
    c.execute("""
        UPDATE signals
        SET status = 'COMPLETED', updated_at = datetime('now')
        WHERE symbol = (SELECT symbol FROM trade_log WHERE id = ?)
    """, (trade_id,))

    conn.commit()


def monitor_trades():
    """Monitor alle open trades voor TP/SL hits."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()
        c.execute("""
            SELECT id, symbol, direction, entry_price, sl, tp, timestamp
            FROM trade_log
            WHERE status = 'OPEN'
        """)
        open_trades = c.fetchall()

        if not open_trades:
            log.debug("Geen open trades")
            return

        log.info(f"Monitoren van {len(open_trades)} open trades...")

        for trade in open_trades:
            trade_dict = dict(trade)
            symbol = trade_dict['symbol']

            current_price = get_current_price(symbol)
            if current_price <= 0:
                log.warning(f"{symbol}: Kon prijs niet ophalen")
                continue

            hit_type, exit_price = check_trade_exit(trade_dict, current_price)

            if hit_type:
                pnl_percent = calculate_pnl(trade_dict, exit_price)

                # Bepaal result
                if hit_type == 'TP':
                    result = 'WIN'
                    emoji = '🎯'
                else:
                    result = 'LOSS'
                    emoji = '❌'

                # Sluit de trade
                close_trade(conn, trade_dict['id'], exit_price, result, pnl_percent)

                log.info(f"{emoji} TRADE CLOSED: {symbol} {trade_dict['direction']}")
                log.info(f"  Result: {result}")
                log.info(f"  Entry: {trade_dict['entry_price']:.6f}")
                log.info(f"  Exit: {exit_price:.6f}")
                log.info(f"  PnL: {pnl_percent:+.2f}%")

            else:
                # Log current status
                entry = trade_dict['entry_price']
                if trade_dict['direction'] == 'BULLISH':
                    unrealized_pnl = ((current_price - entry) / entry) * 100
                else:
                    unrealized_pnl = ((entry - current_price) / entry) * 100

                log.debug(f"{symbol}: Price={current_price:.6f}, Unrealized PnL={unrealized_pnl:+.2f}%")

    except Exception as e:
        log.error(f"Monitor error: {e}")
    finally:
        conn.close()


def get_trade_stats():
    """Bereken en toon trade statistieken."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()

        # Totaal trades
        c.execute("SELECT COUNT(*) as total FROM trade_log")
        total = c.fetchone()['total']

        # Closed trades stats
        c.execute("""
            SELECT
                COUNT(*) as closed,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_percent) as avg_pnl,
                SUM(pnl_percent) as total_pnl
            FROM trade_log
            WHERE status = 'CLOSED'
        """)
        stats = c.fetchone()

        # Open trades
        c.execute("SELECT COUNT(*) as open FROM trade_log WHERE status = 'OPEN'")
        open_count = c.fetchone()['open']

        log.info("=" * 40)
        log.info("TRADE STATISTIEKEN")
        log.info("=" * 40)
        log.info(f"Totaal trades: {total}")
        log.info(f"Open trades: {open_count}")
        log.info(f"Gesloten trades: {stats['closed'] or 0}")

        if stats['closed'] and stats['closed'] > 0:
            win_rate = (stats['wins'] / stats['closed']) * 100 if stats['closed'] else 0
            log.info(f"Wins: {stats['wins'] or 0}")
            log.info(f"Losses: {stats['losses'] or 0}")
            log.info(f"Win Rate: {win_rate:.1f}%")
            log.info(f"Gemiddelde PnL: {stats['avg_pnl'] or 0:.2f}%")
            log.info(f"Totale PnL: {stats['total_pnl'] or 0:.2f}%")

        log.info("=" * 40)

    except Exception as e:
        log.error(f"Stats error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Trade Monitor gestart - Ctrl+C om te stoppen")

    # Toon initiële stats
    get_trade_stats()

    iteration = 0
    while True:
        try:
            monitor_trades()
            iteration += 1

            # Toon stats elke 30 iteraties (5 minuten bij 10s interval)
            if iteration % 30 == 0:
                get_trade_stats()

            time.sleep(MONITOR_INTERVAL)

        except KeyboardInterrupt:
            log.info("Monitor gestopt door gebruiker")
            get_trade_stats()  # Toon finale stats
            break
        except Exception as e:
            log.error(f"Onverwachte error: {e}")
            time.sleep(60)
