"""
Data Fetcher - Historical OHLCV data
"""
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import time
import os
import json


class DataFetcher:
    """Fetch and cache historical OHLCV data"""
    
    def __init__(self, exchange_id: str = 'kucoin', cache_dir: str = 'data'):
        self.exchange = getattr(ccxt, exchange_id)()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch(self, symbol: str, timeframe: str = '4h', 
              days: int = 365, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for backtesting.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe
            days: Number of days of history
            use_cache: Use cached data if available
        """
        cache_file = os.path.join(
            self.cache_dir, 
            f"{symbol.replace('/', '_')}_{timeframe}_{days}d.csv"
        )
        
        # Try cache first
        if use_cache and os.path.exists(cache_file):
            df = pd.read_csv(cache_file, parse_dates=['timestamp'])
            # Check if cache is fresh (less than 4 hours old)
            cache_age = time.time() - os.path.getmtime(cache_file)
            if cache_age < 4 * 3600:
                return df
        
        # Fetch from exchange
        print(f"  Fetching {symbol} {timeframe} ({days} days)...")
        
        all_ohlcv = []
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                if len(ohlcv) < 1000:
                    break
                    
                time.sleep(0.1)  # Rate limit
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
                break
        
        if not all_ohlcv:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # Cache
        if use_cache:
            df.to_csv(cache_file, index=False)
        
        return df
    
    def get_top_coins(self, quote: str = 'USDT', min_volume: float = 10_000_000, 
                      limit: int = 50) -> List[str]:
        """Get top coins by volume"""
        try:
            tickers = self.exchange.fetch_tickers()
            pairs = []
            
            for symbol, ticker in tickers.items():
                if not symbol.endswith(f'/{quote}'):
                    continue
                vol = ticker.get('quoteVolume', 0) or 0
                if vol >= min_volume:
                    pairs.append((symbol, vol))
            
            pairs.sort(key=lambda x: x[1], reverse=True)
            return [p[0] for p in pairs[:limit]]
            
        except Exception as e:
            print(f"Error: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
