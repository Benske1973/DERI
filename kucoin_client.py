# kucoin_client.py - KuCoin API Client for Paper Trading
"""
Comprehensive KuCoin API client for fetching market data.
Uses only public endpoints - no API keys required for paper trading.
"""

import requests
import pandas as pd
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from config import TimeFrame, config

logger = logging.getLogger(__name__)

@dataclass
class Ticker:
    """Ticker data structure."""
    symbol: str
    price: float
    change_24h: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: int

@dataclass
class OHLCV:
    """OHLCV candle data structure."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float

class KuCoinClient:
    """
    KuCoin API Client for market data.

    Features:
    - Rate limiting with automatic backoff
    - Retry logic for failed requests
    - Caching for frequently accessed data
    - All public endpoints for paper trading
    """

    def __init__(self):
        self.base_url = config.kucoin_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "KuCoin-Papertrader/1.0"
        })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

        # Cache
        self._symbols_cache: Dict = {}
        self._symbols_cache_time: float = 0
        self._cache_duration: int = 300  # 5 minutes

    def _request(self, method: str, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and retry logic."""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

        url = f"{self.base_url}{endpoint}"

        for attempt in range(3):
            try:
                self.last_request_time = time.time()

                if method.upper() == "GET":
                    response = self.session.get(url, params=params, timeout=10)
                elif method.upper() == "POST":
                    response = self.session.post(url, json=params, timeout=10)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                data = response.json()

                if data.get("code") == "200000":
                    return data.get("data", {})
                else:
                    logger.warning(f"API error: {data.get('msg', 'Unknown error')}")
                    return {}

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff

        return {}

    def get_symbols(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get all trading symbols.

        Returns:
            List of symbol information dictionaries
        """
        # Check cache
        if not force_refresh and self._symbols_cache:
            if time.time() - self._symbols_cache_time < self._cache_duration:
                return self._symbols_cache.get("symbols", [])

        data = self._request("GET", "/api/v2/symbols")

        if data:
            self._symbols_cache = {"symbols": data}
            self._symbols_cache_time = time.time()

        return data if data else []

    def get_usdt_pairs(self,
                       exclude_stablecoins: bool = True,
                       exclude_leveraged: bool = True,
                       min_volume: float = 0) -> List[str]:
        """
        Get all USDT trading pairs.

        Args:
            exclude_stablecoins: Exclude stablecoin pairs
            exclude_leveraged: Exclude leveraged tokens (3L, 3S, etc.)
            min_volume: Minimum 24h volume filter

        Returns:
            List of USDT pair symbols
        """
        symbols = self.get_symbols()
        pairs = []

        stablecoins = config.scanner.stablecoins

        for sym in symbols:
            symbol = sym.get("symbol", "")

            # Only USDT pairs
            if not symbol.endswith("-USDT"):
                continue

            # Check if trading is enabled
            if not sym.get("enableTrading", True):
                continue

            base = sym.get("baseCurrency", "")

            # Exclude stablecoins
            if exclude_stablecoins and base in stablecoins:
                continue

            # Exclude leveraged tokens
            if exclude_leveraged:
                if any(x in base for x in ["3L", "3S", "2L", "2S", "5L", "5S"]):
                    continue

            pairs.append(symbol)

        return pairs

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        Get ticker for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC-USDT")

        Returns:
            Ticker data or None
        """
        data = self._request("GET", f"/api/v1/market/stats", {"symbol": symbol})

        if data:
            return Ticker(
                symbol=symbol,
                price=float(data.get("last", 0)),
                change_24h=float(data.get("changeRate", 0)) * 100,
                volume_24h=float(data.get("volValue", 0)),
                high_24h=float(data.get("high", 0)),
                low_24h=float(data.get("low", 0)),
                timestamp=int(time.time() * 1000)
            )
        return None

    def get_all_tickers(self) -> List[Ticker]:
        """
        Get tickers for all symbols.

        Returns:
            List of Ticker objects
        """
        data = self._request("GET", "/api/v1/market/allTickers")

        tickers = []
        if data and "ticker" in data:
            for t in data["ticker"]:
                try:
                    tickers.append(Ticker(
                        symbol=t.get("symbol", ""),
                        price=float(t.get("last", 0)),
                        change_24h=float(t.get("changeRate", 0)) * 100,
                        volume_24h=float(t.get("volValue", 0)),
                        high_24h=float(t.get("high", 0)),
                        low_24h=float(t.get("low", 0)),
                        timestamp=int(time.time() * 1000)
                    ))
                except (ValueError, TypeError):
                    continue

        return tickers

    def get_top_volume_pairs(self, count: int = 50, quote: str = "USDT") -> List[str]:
        """
        Get top trading pairs by 24h volume.

        Args:
            count: Number of pairs to return
            quote: Quote currency filter

        Returns:
            List of symbols sorted by volume
        """
        tickers = self.get_all_tickers()

        # Filter by quote currency
        filtered = [t for t in tickers if t.symbol.endswith(f"-{quote}")]

        # Exclude stablecoins and leveraged
        stablecoins = config.scanner.stablecoins
        filtered = [
            t for t in filtered
            if not any(s in t.symbol.split("-")[0] for s in stablecoins)
            and not any(x in t.symbol for x in ["3L", "3S", "2L", "2S"])
        ]

        # Sort by volume
        sorted_pairs = sorted(filtered, key=lambda x: x.volume_24h, reverse=True)

        return [t.symbol for t in sorted_pairs[:count]]

    def get_candles(self,
                    symbol: str,
                    timeframe: TimeFrame,
                    start_time: int = None,
                    end_time: int = None,
                    limit: int = 500) -> pd.DataFrame:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start timestamp (seconds)
            end_time: End timestamp (seconds)
            limit: Number of candles (max 1500)

        Returns:
            DataFrame with OHLCV data
        """
        params = {
            "symbol": symbol,
            "type": timeframe.value,
        }

        if start_time:
            params["startAt"] = start_time
        if end_time:
            params["endAt"] = end_time

        data = self._request("GET", "/api/v1/market/candles", params)

        if not data:
            return pd.DataFrame()

        # KuCoin returns: [timestamp, open, close, high, low, volume, turnover]
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"]
        )

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sort by timestamp (oldest first)
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Convert timestamp to datetime
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

        return df

    def get_orderbook(self, symbol: str, depth: int = 20) -> Dict:
        """
        Get orderbook for a symbol.

        Args:
            symbol: Trading pair symbol
            depth: Orderbook depth (20 or 100)

        Returns:
            Orderbook data with bids and asks
        """
        endpoint = f"/api/v1/market/orderbook/level2_{depth}"
        data = self._request("GET", endpoint, {"symbol": symbol})

        if data:
            return {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
                "timestamp": data.get("time", int(time.time() * 1000))
            }
        return {"bids": [], "asks": [], "timestamp": 0}

    def get_trade_history(self, symbol: str) -> List[Dict]:
        """
        Get recent trade history.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of recent trades
        """
        data = self._request("GET", "/api/v1/market/histories", {"symbol": symbol})

        if data:
            return [
                {
                    "time": int(t.get("time", 0)) // 1000000000,
                    "price": float(t.get("price", 0)),
                    "size": float(t.get("size", 0)),
                    "side": t.get("side", "")
                }
                for t in data
            ]
        return []

    def get_server_time(self) -> int:
        """Get KuCoin server time in milliseconds."""
        data = self._request("GET", "/api/v1/timestamp")
        return data if isinstance(data, int) else int(time.time() * 1000)

    def get_currencies(self) -> List[Dict]:
        """Get list of all currencies."""
        return self._request("GET", "/api/v1/currencies") or []

    def get_fiat_prices(self, currencies: List[str] = None) -> Dict[str, float]:
        """
        Get fiat prices for currencies.

        Args:
            currencies: List of currency codes (default: BTC, ETH)

        Returns:
            Dictionary of currency to USD price
        """
        params = {}
        if currencies:
            params["currencies"] = ",".join(currencies)

        data = self._request("GET", "/api/v1/prices", params)

        if data:
            return {k: float(v) for k, v in data.items()}
        return {}


class KuCoinWebSocketToken:
    """Get WebSocket connection token."""

    def __init__(self):
        self.base_url = config.kucoin_base_url

    def get_public_token(self) -> Tuple[str, str]:
        """
        Get public WebSocket token.

        Returns:
            Tuple of (websocket_url, token)
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/bullet-public",
                timeout=10
            )
            data = response.json()

            if data.get("code") == "200000":
                instance = data["data"]["instanceServers"][0]
                endpoint = instance["endpoint"]
                token = data["data"]["token"]
                return endpoint, token

        except Exception as e:
            logger.error(f"Failed to get WebSocket token: {e}")

        return "", ""


# Singleton instance
kucoin_client = KuCoinClient()
