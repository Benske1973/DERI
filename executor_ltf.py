# executor_ltf.py - LTF Executor met bullish/bearish support
import sqlite3
import requests
import pandas as pd
import time
from config import (
    KUCOIN_BASE_URL, LTF_TIMEFRAME, RISK_REWARD_RATIO,
    EXECUTOR_INTERVAL, API_RATE_LIMIT, DATABASE_PATH,
    MAX_RETRIES, RETRY_DELAY
)
from logger import get_executor_logger

log = get_executor_logger()


def api_request(url: str) -> dict:
    """API request met retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(API_RATE_LIMIT)
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('code') == '200000':
                return data
        except Exception as e:
            log.warning(f"API request failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None


def check_bullish_confirmation(df: pd.DataFrame) -> bool:
    """Check voor bullish ChoCH (Change of Character)."""
    if len(df) < 15:
        return False

    # Swing high van de laatste 14 candles (exclusief huidige)
    swing_high = df['h'].iloc[-15:-1].max()
    current_close = df['c'].iloc[-1]

    # Bullish confirmatie: body close boven swing high
    return current_close > swing_high


def check_bearish_confirmation(df: pd.DataFrame) -> bool:
    """Check voor bearish ChoCH (Change of Character)."""
    if len(df) < 15:
        return False

    # Swing low van de laatste 14 candles (exclusief huidige)
    swing_low = df['l'].iloc[-15:-1].min()
    current_close = df['c'].iloc[-1]

    # Bearish confirmatie: body close onder swing low
    return current_close < swing_low


def process_signal(conn: sqlite3.Connection, signal: tuple) -> bool:
    """Verwerk een TAPPED signal."""
    symbol, trend, ob_top, ob_bottom = signal

    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles?symbol={symbol}&type={LTF_TIMEFRAME}"
    data = api_request(url)

    if not data or 'data' not in data or not data['data']:
        log.warning(f"{symbol}: Geen LTF candle data")
        return False

    try:
        df = pd.DataFrame(
            data['data'],
            columns=['ts', 'o', 'c', 'h', 'l', 'v', 'a']
        ).astype(float).iloc[::-1]

        c = conn.cursor()
        confirmed = False

        if trend == 'BULLISH':
            if check_bullish_confirmation(df):
                # Bullish trade setup
                entry = ob_top
                sl = ob_bottom
                risk = entry - sl
                tp = entry + (risk * RISK_REWARD_RATIO)
                confirmed = True
                log.info(f"BULLISH CONFIRMED: {symbol}")

        elif trend == 'BEARISH':
            if check_bearish_confirmation(df):
                # Bearish trade setup
                entry = ob_bottom
                sl = ob_top
                risk = sl - entry
                tp = entry - (risk * RISK_REWARD_RATIO)
                confirmed = True
                log.info(f"BEARISH CONFIRMED: {symbol}")

        if confirmed:
            # Update signal status
            c.execute("""
                UPDATE signals
                SET status = 'IN_TRADE', updated_at = datetime('now')
                WHERE symbol = ?
            """, (symbol,))

            # Log trade
            c.execute("""
                INSERT INTO trade_log
                (symbol, direction, entry_price, sl, tp, status)
                VALUES (?, ?, ?, ?, ?, 'OPEN')
            """, (symbol, trend, entry, sl, tp))

            conn.commit()

            log.info(f"TRADE LOGGED: {symbol} {trend}")
            log.info(f"  Entry: {entry:.6f}")
            log.info(f"  SL: {sl:.6f}")
            log.info(f"  TP: {tp:.6f}")
            log.info(f"  R:R = 1:{RISK_REWARD_RATIO}")

            return True

    except Exception as e:
        log.error(f"{symbol}: Processing error - {e}")

    return False


def check_ltf_confirmation():
    """Hoofdfunctie voor LTF confirmatie check."""
    conn = sqlite3.connect(DATABASE_PATH)

    try:
        c = conn.cursor()
        c.execute("""
            SELECT symbol, trend, ob_top, ob_bottom
            FROM signals
            WHERE status = 'TAPPED'
        """)
        tapped_signals = c.fetchall()

        if not tapped_signals:
            log.debug("Geen TAPPED signals om te verwerken")
            return

        log.info(f"Verwerken van {len(tapped_signals)} TAPPED signals...")

        for signal in tapped_signals:
            process_signal(conn, signal)

    except Exception as e:
        log.error(f"Executor error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Executor gestart - Ctrl+C om te stoppen")

    while True:
        try:
            check_ltf_confirmation()
            time.sleep(EXECUTOR_INTERVAL)
        except KeyboardInterrupt:
            log.info("Executor gestopt door gebruiker")
            break
        except Exception as e:
            log.error(f"Onverwachte error: {e}")
            time.sleep(60)
