# watcher_ws.py - WebSocket Watcher met persistente DB connectie
import asyncio
import websockets
import json
import sqlite3
import requests
import time
from config import (
    KUCOIN_BASE_URL, KUCOIN_WS_PUBLIC, DATABASE_PATH,
    MAX_RETRIES, RETRY_DELAY
)
from logger import get_watcher_logger

log = get_watcher_logger()


class DatabaseConnection:
    """Persistente database connectie manager."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self._connect()

    def _connect(self):
        """Maak nieuwe database connectie."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            log.info("Database connectie gemaakt")
        except Exception as e:
            log.error(f"Database connectie error: {e}")
            raise

    def execute(self, query: str, params: tuple = ()):
        """Voer query uit met auto-reconnect."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor
        except sqlite3.Error as e:
            log.warning(f"Database error, reconnecting: {e}")
            self._connect()
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor

    def commit(self):
        """Commit transactie."""
        self.conn.commit()

    def close(self):
        """Sluit connectie."""
        if self.conn:
            self.conn.close()
            log.info("Database connectie gesloten")


def get_ws_token() -> tuple:
    """Haal WebSocket token en URL op."""
    for attempt in range(MAX_RETRIES):
        try:
            url = f"{KUCOIN_BASE_URL}{KUCOIN_WS_PUBLIC}"
            response = requests.post(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == '200000':
                server = data['data']['instanceServers'][0]
                ws_url = f"{server['endpoint']}?token={data['data']['token']}"
                ping_interval = server.get('pingInterval', 20000) // 1000
                return ws_url, ping_interval

        except Exception as e:
            log.warning(f"Token request failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))

    return None, None


def check_zone_tap(db: DatabaseConnection, symbol: str, price: float) -> bool:
    """Check of prijs in een FVG zone valt."""
    cursor = db.execute(
        "SELECT fvg_top, fvg_bottom, trend, status FROM signals WHERE symbol = ?",
        (symbol,)
    )
    row = cursor.fetchone()

    if not row or row['status'] != 'SCANNING':
        return False

    fvg_top = row['fvg_top']
    fvg_bottom = row['fvg_bottom']
    trend = row['trend']

    # Check tap gebaseerd op trend richting
    if trend == 'BULLISH':
        # Bullish: prijs moet in de FVG zone komen (van boven naar beneden)
        if fvg_bottom <= price <= fvg_top:
            db.execute(
                "UPDATE signals SET status = 'TAPPED', updated_at = datetime('now') WHERE symbol = ?",
                (symbol,)
            )
            db.commit()
            log.info(f"BULLISH TAP: {symbol} @ {price:.4f} (Zone: {fvg_bottom:.4f} - {fvg_top:.4f})")
            return True

    elif trend == 'BEARISH':
        # Bearish: prijs moet in de FVG zone komen (van onder naar boven)
        if fvg_bottom <= price <= fvg_top:
            db.execute(
                "UPDATE signals SET status = 'TAPPED', updated_at = datetime('now') WHERE symbol = ?",
                (symbol,)
            )
            db.commit()
            log.info(f"BEARISH TAP: {symbol} @ {price:.4f} (Zone: {fvg_bottom:.4f} - {fvg_top:.4f})")
            return True

    return False


async def handle_ws():
    """Hoofd WebSocket handler."""
    db = DatabaseConnection(DATABASE_PATH)
    reconnect_delay = 5

    while True:
        try:
            ws_url, ping_interval = get_ws_token()

            if not ws_url:
                log.error("Kon geen WebSocket token krijgen")
                await asyncio.sleep(reconnect_delay)
                continue

            log.info("Verbinden met KuCoin WebSocket...")

            async with websockets.connect(
                ws_url,
                ping_interval=ping_interval,
                ping_timeout=10,
                close_timeout=5
            ) as ws:

                # Subscribe op alle tickers
                subscribe_msg = {
                    "id": str(int(time.time() * 1000)),
                    "type": "subscribe",
                    "topic": "/market/ticker:all",
                    "privateChannel": False,
                    "response": True
                }
                await ws.send(json.dumps(subscribe_msg))

                log.info("Verbonden en subscribed op alle tickers")
                reconnect_delay = 5  # Reset delay na succesvolle connectie
                taps_count = 0
                msg_count = 0

                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg)

                        if data.get("type") == "message" and "data" in data:
                            topic = data.get('topic', '')
                            symbol = topic.split(":")[-1] if ":" in topic else None

                            if symbol and symbol.endswith('-USDT'):
                                try:
                                    price = float(data['data'].get('price', 0))
                                    if price > 0:
                                        msg_count += 1
                                        if check_zone_tap(db, symbol, price):
                                            taps_count += 1

                                        # Log stats elke 10000 messages
                                        if msg_count % 10000 == 0:
                                            log.info(f"Processed: {msg_count} msgs, Taps: {taps_count}")

                                except (ValueError, KeyError) as e:
                                    log.debug(f"Parse error voor {symbol}: {e}")

                        elif data.get("type") == "welcome":
                            log.debug("Welcome message ontvangen")

                        elif data.get("type") == "ack":
                            log.debug("Subscription bevestigd")

                    except asyncio.TimeoutError:
                        # Stuur ping om connectie levend te houden
                        try:
                            await ws.send(json.dumps({"type": "ping"}))
                        except:
                            break

                    except websockets.exceptions.ConnectionClosed:
                        log.warning("WebSocket connectie gesloten")
                        break

        except Exception as e:
            log.error(f"WebSocket error: {e}")

        log.info(f"Opnieuw verbinden in {reconnect_delay} seconden...")
        await asyncio.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff, max 60s


async def main():
    """Main entry point."""
    log.info("=" * 50)
    log.info("Watcher gestart - Ctrl+C om te stoppen")

    try:
        await handle_ws()
    except KeyboardInterrupt:
        log.info("Watcher gestopt door gebruiker")
    except Exception as e:
        log.error(f"Fatal error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
