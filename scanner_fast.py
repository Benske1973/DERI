# scanner_fast.py - Snelle parallelle scanner met asyncio
import asyncio
import aiohttp
import pandas as pd
import sqlite3
import time
from config import (
    KUCOIN_BASE_URL, MIN_VOLUME_USDT, HTF_TIMEFRAME,
    EMA_LENGTH, SCANNER_INTERVAL, DATABASE_PATH
)
from logger import get_scanner_logger

log = get_scanner_logger()

# Parallel settings
MAX_CONCURRENT = 20  # Max gelijktijdige requests
RATE_LIMIT_DELAY = 0.05  # 50ms tussen requests


def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """Bereken EMA."""
    return series.ewm(span=length, adjust=False).mean()


def detect_bullish_fvg(df: pd.DataFrame) -> tuple:
    """Detecteer Bullish FVG."""
    for i in range(len(df) - 1, 5, -1):
        if df['l'].iloc[i] > df['h'].iloc[i - 2]:
            fvg_top = df['l'].iloc[i]
            fvg_bottom = df['h'].iloc[i - 2]
            ob_top = df['h'].iloc[i - 2]
            ob_bottom = df['l'].iloc[i - 2]
            return fvg_top, fvg_bottom, ob_top, ob_bottom
    return None, None, None, None


def detect_bearish_fvg(df: pd.DataFrame) -> tuple:
    """Detecteer Bearish FVG."""
    for i in range(len(df) - 1, 5, -1):
        if df['h'].iloc[i] < df['l'].iloc[i - 2]:
            fvg_top = df['l'].iloc[i - 2]
            fvg_bottom = df['h'].iloc[i]
            ob_top = df['h'].iloc[i - 2]
            ob_bottom = df['l'].iloc[i - 2]
            return fvg_top, fvg_bottom, ob_top, ob_bottom
    return None, None, None, None


async def fetch_symbols(session: aiohttp.ClientSession) -> list:
    """Haal alle high-volume symbols op."""
    url = f"{KUCOIN_BASE_URL}/api/v1/market/allTickers"

    try:
        async with session.get(url) as resp:
            data = await resp.json()

        if data.get('code') != '200000':
            return []

        symbols = []
        for ticker in data['data'].get('ticker', []):
            symbol = ticker.get('symbol', '')
            if not symbol.endswith('-USDT'):
                continue
            try:
                vol = float(ticker.get('volValue', 0))
                if vol >= MIN_VOLUME_USDT:
                    symbols.append({
                        'symbol': symbol,
                        'volume': vol,
                        'price': float(ticker.get('last', 0))
                    })
            except:
                continue

        symbols.sort(key=lambda x: x['volume'], reverse=True)
        return symbols

    except Exception as e:
        log.error(f"Fetch symbols error: {e}")
        return []


async def analyze_symbol(session: aiohttp.ClientSession, sym: str, semaphore: asyncio.Semaphore) -> dict:
    """Analyseer één symbol voor FVG setups."""
    async with semaphore:
        await asyncio.sleep(RATE_LIMIT_DELAY)

        url = f"{KUCOIN_BASE_URL}/api/v1/market/candles?symbol={sym}&type={HTF_TIMEFRAME}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()

            if data.get('code') != '200000' or not data.get('data'):
                return None

            df = pd.DataFrame(
                data['data'],
                columns=['ts', 'o', 'c', 'h', 'l', 'v', 'a']
            ).astype(float).iloc[::-1]

            if len(df) < EMA_LENGTH + 10:
                return None

            df['ema'] = calculate_ema(df['c'], EMA_LENGTH)
            current_price = df['c'].iloc[-1]
            current_ema = df['ema'].iloc[-1]

            # Bullish check
            if current_price > current_ema:
                ft, fb, ot, ob = detect_bullish_fvg(df)
                if ft is not None:
                    return {
                        'symbol': sym,
                        'trend': 'BULLISH',
                        'fvg_top': ft,
                        'fvg_bottom': fb,
                        'ob_top': ot,
                        'ob_bottom': ob
                    }

            # Bearish check
            elif current_price < current_ema:
                ft, fb, ot, ob = detect_bearish_fvg(df)
                if ft is not None:
                    return {
                        'symbol': sym,
                        'trend': 'BEARISH',
                        'fvg_top': ft,
                        'fvg_bottom': fb,
                        'ob_top': ot,
                        'ob_bottom': ob
                    }

            return None

        except Exception as e:
            log.debug(f"{sym}: Error - {e}")
            return None


def save_results(symbols: list, setups: list):
    """Sla resultaten op in database."""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Save active symbols
    c.execute("DELETE FROM active_symbols")
    for s in symbols:
        c.execute("""
            INSERT OR REPLACE INTO active_symbols (symbol, volume_24h, last_price, updated_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (s['symbol'], s['volume'], s['price']))

    # Save setups
    for setup in setups:
        c.execute("""
            INSERT OR REPLACE INTO signals
            (symbol, trend, fvg_top, fvg_bottom, ob_top, ob_bottom, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'SCANNING', datetime('now'))
        """, (
            setup['symbol'], setup['trend'], setup['fvg_top'],
            setup['fvg_bottom'], setup['ob_top'], setup['ob_bottom']
        ))

    conn.commit()
    conn.close()


async def run_scanner_async():
    """Hoofdfunctie - parallel scannen."""
    log.info("=" * 50)
    log.info("FAST Scanner gestart (parallel)")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        # Haal symbols op
        log.info(f"Ophalen van symbols met volume > {MIN_VOLUME_USDT} USDT...")
        symbols = await fetch_symbols(session)

        if not symbols:
            log.error("Geen symbols gevonden!")
            return

        log.info(f"Gevonden: {len(symbols)} symbols")

        # Parallel analyseren
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        tasks = [
            analyze_symbol(session, s['symbol'], semaphore)
            for s in symbols
        ]

        log.info(f"Analyseren van {len(tasks)} symbols (max {MAX_CONCURRENT} parallel)...")

        results = await asyncio.gather(*tasks)

        # Filter None results
        setups = [r for r in results if r is not None]

        # Save to database
        save_results(symbols, setups)

        elapsed = time.time() - start_time

        log.info(f"Scan voltooid in {elapsed:.1f}s: {len(setups)} setups uit {len(symbols)} symbols")

        # Log setups
        for setup in setups[:10]:  # Max 10 tonen
            emoji = "🟢" if setup['trend'] == 'BULLISH' else "🔴"
            log.info(f"  {emoji} {setup['symbol']:12} {setup['trend']:8} FVG: {setup['fvg_bottom']:.6f} - {setup['fvg_top']:.6f}")

        if len(setups) > 10:
            log.info(f"  ... en {len(setups) - 10} meer")


def run_scanner():
    """Wrapper voor async scanner."""
    asyncio.run(run_scanner_async())


if __name__ == "__main__":
    log.info("Fast Scanner - Ctrl+C om te stoppen")

    while True:
        try:
            run_scanner()
            log.info(f"Volgende scan over {SCANNER_INTERVAL // 60} minuten...")
            time.sleep(SCANNER_INTERVAL)
        except KeyboardInterrupt:
            log.info("Scanner gestopt")
            break
        except Exception as e:
            log.error(f"Error: {e}")
            time.sleep(60)
