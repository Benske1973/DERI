# websocket_feed.py - Real-time Price Feed via WebSocket
"""
WebSocket connection to KuCoin for real-time price updates.
Monitors prices and triggers callbacks on price updates.
"""

import asyncio
import websockets
import json
import time
import logging
from typing import Dict, Callable, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

from config import config
from kucoin_client import KuCoinWebSocketToken

logger = logging.getLogger(__name__)

@dataclass
class PriceUpdate:
    """Price update structure."""
    symbol: str
    price: float
    best_bid: float
    best_ask: float
    volume_24h: float
    timestamp: int

@dataclass
class WebSocketStats:
    """WebSocket connection statistics."""
    connected: bool = False
    connect_time: Optional[datetime] = None
    disconnect_time: Optional[datetime] = None
    messages_received: int = 0
    reconnect_count: int = 0
    last_message_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

class KuCoinWebSocket:
    """
    KuCoin WebSocket client for real-time price feeds.

    Features:
    - Automatic reconnection with exponential backoff
    - Multiple symbol subscription
    - Price callbacks for external handlers
    - Connection health monitoring
    """

    def __init__(self):
        self.ws_token = KuCoinWebSocketToken()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_running = False
        self.subscribed_symbols: Set[str] = set()
        self.prices: Dict[str, PriceUpdate] = {}
        self.callbacks: List[Callable] = []
        self.stats = WebSocketStats()

        # Configuration
        self.cfg = config.websocket
        self.ping_interval = self.cfg.ping_interval
        self.reconnect_delay = self.cfg.reconnect_delay
        self.max_reconnects = self.cfg.max_reconnect_attempts

        # Tasks
        self._message_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None

    def add_callback(self, callback: Callable[[str, float], None]):
        """
        Add price update callback.

        Args:
            callback: Function(symbol, price) to call on price updates
        """
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            endpoint, token = self.ws_token.get_public_token()

            if not endpoint or not token:
                logger.error("Failed to get WebSocket token")
                return False

            ws_url = f"{endpoint}?token={token}"

            self.ws = await websockets.connect(
                ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.cfg.ping_timeout
            )

            self.stats.connected = True
            self.stats.connect_time = datetime.now()

            logger.info(f"WebSocket connected to KuCoin")
            return True

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.stats.errors.append(f"{datetime.now()}: {str(e)}")
            return False

    async def disconnect(self):
        """Disconnect WebSocket."""
        self.is_running = False

        if self._message_task:
            self._message_task.cancel()
        if self._ping_task:
            self._ping_task.cancel()

        if self.ws:
            await self.ws.close()
            self.ws = None

        self.stats.connected = False
        self.stats.disconnect_time = datetime.now()

        logger.info("WebSocket disconnected")

    async def subscribe_all_tickers(self):
        """Subscribe to all ticker updates."""
        if not self.ws:
            return

        message = {
            "id": str(int(time.time() * 1000)),
            "type": "subscribe",
            "topic": "/market/ticker:all",
            "privateChannel": False,
            "response": True
        }

        await self.ws.send(json.dumps(message))
        logger.info("Subscribed to all tickers")

    async def subscribe_symbols(self, symbols: List[str]):
        """Subscribe to specific symbols."""
        if not self.ws or not symbols:
            return

        self.subscribed_symbols.update(symbols)

        # Subscribe in batches of 100
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i+100]
            topic = ",".join([f"/market/ticker:{s}" for s in batch])

            message = {
                "id": str(int(time.time() * 1000)),
                "type": "subscribe",
                "topic": topic,
                "privateChannel": False,
                "response": True
            }

            await self.ws.send(json.dumps(message))
            await asyncio.sleep(0.1)

        logger.info(f"Subscribed to {len(symbols)} symbols")

    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            self.stats.messages_received += 1
            self.stats.last_message_time = datetime.now()

            msg_type = data.get("type")

            if msg_type == "message":
                topic = data.get("topic", "")
                ticker_data = data.get("data", {})

                if "/market/ticker" in topic:
                    symbol = topic.split(":")[-1]

                    # Handle "all" ticker topic
                    if symbol == "all" and "subject" in data:
                        symbol = data.get("subject", "")

                    if symbol and ticker_data:
                        price = float(ticker_data.get("price", 0))

                        if price > 0:
                            update = PriceUpdate(
                                symbol=symbol,
                                price=price,
                                best_bid=float(ticker_data.get("bestBid", 0)),
                                best_ask=float(ticker_data.get("bestAsk", 0)),
                                volume_24h=float(ticker_data.get("volValue", 0)),
                                timestamp=int(time.time() * 1000)
                            )

                            self.prices[symbol] = update

                            # Trigger callbacks
                            for callback in self.callbacks:
                                try:
                                    callback(symbol, price)
                                except Exception as e:
                                    logger.error(f"Callback error: {e}")

            elif msg_type == "welcome":
                logger.debug("WebSocket welcome received")

            elif msg_type == "pong":
                logger.debug("Pong received")

            elif msg_type == "ack":
                logger.debug(f"Subscription acknowledged: {data.get('id')}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message[:100]}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _message_loop(self):
        """Main message receiving loop."""
        while self.is_running and self.ws:
            try:
                message = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=30
                )
                await self._handle_message(message)

            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await self._send_ping()

            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
                break

            except Exception as e:
                logger.error(f"Message loop error: {e}")
                break

    async def _send_ping(self):
        """Send ping to keep connection alive."""
        if self.ws:
            try:
                message = {
                    "id": str(int(time.time() * 1000)),
                    "type": "ping"
                }
                await self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Ping failed: {e}")

    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        delay = self.reconnect_delay

        for attempt in range(self.max_reconnects):
            logger.info(f"Reconnecting (attempt {attempt + 1}/{self.max_reconnects})...")

            if await self.connect():
                await self.subscribe_all_tickers()
                self.stats.reconnect_count += 1
                return True

            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)  # Max 60 seconds

        logger.error("Max reconnection attempts reached")
        return False

    async def run(self):
        """Main run loop with automatic reconnection."""
        self.is_running = True

        while self.is_running:
            if not await self.connect():
                if not await self._reconnect():
                    break
                continue

            await self.subscribe_all_tickers()

            # Start message loop
            await self._message_loop()

            # If we get here, connection was lost
            if self.is_running:
                logger.info("Connection lost, attempting reconnect...")
                await asyncio.sleep(self.reconnect_delay)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        update = self.prices.get(symbol)
        return update.price if update else None

    def get_all_prices(self) -> Dict[str, float]:
        """Get all current prices."""
        return {symbol: update.price for symbol, update in self.prices.items()}

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "connected": self.stats.connected,
            "connect_time": self.stats.connect_time.isoformat() if self.stats.connect_time else None,
            "messages_received": self.stats.messages_received,
            "reconnect_count": self.stats.reconnect_count,
            "last_message": self.stats.last_message_time.isoformat()
            if self.stats.last_message_time else None,
            "symbols_tracked": len(self.prices),
            "recent_errors": self.stats.errors[-5:]
        }


# Singleton instance
ws_feed = KuCoinWebSocket()


# Legacy compatibility handler
async def handle_ws():
    """Legacy WebSocket handler."""
    from database import db
    from paper_trader import paper_trader

    def price_callback(symbol: str, price: float):
        """Handle price updates."""
        # Update paper trader
        paper_trader.update_price(symbol, price)

        # Check signals
        signal = db.get_signal(symbol)
        if signal and signal.get('status') == 'SCANNING':
            fvg_top = signal.get('fvg_top', 0)
            fvg_bottom = signal.get('fvg_bottom', 0)

            if fvg_bottom <= price <= fvg_top:
                db.update_signal_status(symbol, 'TAPPED')
                logger.info(f"TAPPED: {symbol} @ {price:.4f} (Zone: {fvg_bottom:.4f}-{fvg_top:.4f})")

    ws_feed.add_callback(price_callback)
    await ws_feed.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    async def test():
        def print_price(symbol, price):
            if symbol in ["BTC-USDT", "ETH-USDT"]:
                print(f"{symbol}: {price:.2f}")

        ws_feed.add_callback(print_price)
        await ws_feed.run()

    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nStopped by user")
