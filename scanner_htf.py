# scanner_htf.py - HTF Scanner met dynamische symbols en bullish/bearish setups
import requests
import pandas as pd
import sqlite3
import time
from config import (
    KUCOIN_BASE_URL, MIN_VOLUME_USDT, HTF_TIMEFRAME,
    EMA_LENGTH, SCANNER_INTERVAL, API_RATE_LIMIT,
    MAX_RETRIES, RETRY_DELAY, DATABASE_PATH
)
from logger import get_scanner_logger

log = get_scanner_logger()


def calculate_ema(series: pd.Series, length: int) -> pd.Series:
    """Bereken EMA handmatig (zonder pandas_ta)."""
    return series.ewm(span=length, adjust=False).mean()


def api_request(url: str, retries: int = MAX_RETRIES) -> dict:
    """API request met retry logic en rate limiting."""
    for attempt in range(retries):
        try:
            time.sleep(API_RATE_LIMIT)  # Rate limiting
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('code') == '200000':
                return data
            else:
                log.warning(f"API error: {data.get('msg', 'Unknown error')}")
        except requests.exceptions.RequestException as e:
            log.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    return None


def get_high_volume_symbols() -> list:
    """Haal alle USDT pairs op met volume > MIN_VOLUME_USDT."""
    log.info(f"Ophalen van symbols met volume > {MIN_VOLUME_USDT} USDT...")

    url = f"{KUCOIN_BASE_URL}/api/v1/market/allTickers"
    data = api_request(url)

    if not data or 'data' not in data:
        log.error("Kon ticker data niet ophalen")
        return []

    symbols = []
    for ticker in data['data'].get('ticker', []):
        symbol = ticker.get('symbol', '')
        if not symbol.endswith('-USDT'):
            continue

        try:
            vol_value = float(ticker.get('volValue', 0))
            if vol_value >= MIN_VOLUME_USDT:
                symbols.append({
                    'symbol': symbol,
                    'volume': vol_value,
                    'price': float(ticker.get('last', 0))
                })
        except (ValueError, TypeError):
            continue

    # Sorteer op volume (hoogste eerst)
    symbols.sort(key=lambda x: x['volume'], reverse=True)

    log.info(f"Gevonden: {len(symbols)} symbols met volume > {MIN_VOLUME_USDT} USDT")
    return symbols


def save_active_symbols(symbols: list, conn: sqlite3.Connection):
    """Sla actieve symbols op in database."""
    c = conn.cursor()
    c.execute("DELETE FROM active_symbols")

    for s in symbols:
        c.execute("""
            INSERT OR REPLACE INTO active_symbols (symbol, volume_24h, last_price, updated_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (s['symbol'], s['volume'], s['price']))

    conn.commit()
    log.info(f"{len(symbols)} active symbols opgeslagen")


def detect_bullish_fvg(df: pd.DataFrame) -> tuple:
    """Detecteer Bullish Fair Value Gap."""
    for i in range(len(df) - 1, 5, -1):
        # Bullish FVG: Low[i] > High[i-2] (gap omhoog)
        if df['l'].iloc[i] > df['h'].iloc[i - 2]:
            fvg_top = df['l'].iloc[i]
            fvg_bottom = df['h'].iloc[i - 2]
            # Order Block is de bearish kaars voor de beweging
            ob_top = df['h'].iloc[i - 2]
            ob_bottom = df['l'].iloc[i - 2]
            return fvg_top, fvg_bottom, ob_top, ob_bottom
    return None, None, None, None


def detect_bearish_fvg(df: pd.DataFrame) -> tuple:
    """Detecteer Bearish Fair Value Gap."""
    for i in range(len(df) - 1, 5, -1):
        # Bearish FVG: High[i] < Low[i-2] (gap omlaag)
        if df['h'].iloc[i] < df['l'].iloc[i - 2]:
            fvg_top = df['l'].iloc[i - 2]
            fvg_bottom = df['h'].iloc[i]
            # Order Block is de bullish kaars voor de beweging
            ob_top = df['h'].iloc[i - 2]
            ob_bottom = df['l'].iloc[i - 2]
            return fvg_top, fvg_bottom, ob_top, ob_bottom
    return None, None, None, None


def analyze_symbol(sym: str, conn: sqlite3.Connection) -> bool:
    """Analyseer een symbol voor SMC setups."""
    url = f"{KUCOIN_BASE_URL}/api/v1/market/candles?symbol={sym}&type={HTF_TIMEFRAME}"
    data = api_request(url)

    if not data or 'data' not in data or not data['data']:
        log.warning(f"{sym}: Geen candle data beschikbaar")
        return False

    try:
        df = pd.DataFrame(
            data['data'],
            columns=['ts', 'o', 'c', 'h', 'l', 'v', 'a']
        ).astype(float).iloc[::-1]

        if len(df) < EMA_LENGTH + 10:
            log.debug(f"{sym}: Onvoldoende data ({len(df)} candles)")
            return False

        # Bereken EMA voor trend
        df['ema'] = calculate_ema(df['c'], EMA_LENGTH)
        current_price = df['c'].iloc[-1]
        current_ema = df['ema'].iloc[-1]

        c = conn.cursor()

        # Check voor BULLISH setup (prijs boven EMA = uptrend)
        if current_price > current_ema:
            ft, fb, ot, ob = detect_bullish_fvg(df)
            if ft is not None:
                c.execute("""
                    INSERT OR REPLACE INTO signals
                    (symbol, trend, fvg_top, fvg_bottom, ob_top, ob_bottom, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (sym, 'BULLISH', ft, fb, ot, ob, 'SCANNING'))
                conn.commit()
                log.info(f"BULLISH setup: {sym} | FVG: {fb:.4f} - {ft:.4f}")
                return True
            else:
                log.debug(f"{sym}: Uptrend maar geen Bullish FVG")

        # Check voor BEARISH setup (prijs onder EMA = downtrend)
        elif current_price < current_ema:
            ft, fb, ot, ob = detect_bearish_fvg(df)
            if ft is not None:
                c.execute("""
                    INSERT OR REPLACE INTO signals
                    (symbol, trend, fvg_top, fvg_bottom, ob_top, ob_bottom, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (sym, 'BEARISH', ft, fb, ot, ob, 'SCANNING'))
                conn.commit()
                log.info(f"BEARISH setup: {sym} | FVG: {fb:.4f} - {ft:.4f}")
                return True
            else:
                log.debug(f"{sym}: Downtrend maar geen Bearish FVG")

        return False

    except Exception as e:
        log.error(f"{sym}: Analyse error - {e}")
        return False


def run_scanner():
    """Hoofdfunctie voor de scanner."""
    log.info("=" * 50)
    log.info("Scanner gestart")

    conn = sqlite3.connect(DATABASE_PATH)

    try:
        # Haal high volume symbols op
        symbols = get_high_volume_symbols()

        if not symbols:
            log.error("Geen symbols gevonden!")
            return

        # Sla op in database
        save_active_symbols(symbols, conn)

        # Analyseer elke symbol
        setups_found = 0
        for i, s in enumerate(symbols):
            sym = s['symbol']
            log.debug(f"[{i+1}/{len(symbols)}] Analyseren: {sym} (Vol: ${s['volume']:,.0f})")

            if analyze_symbol(sym, conn):
                setups_found += 1

        log.info(f"Scan voltooid: {setups_found} setups gevonden uit {len(symbols)} symbols")

    except Exception as e:
        log.error(f"Scanner error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    log.info("HTF Scanner opgestart - Ctrl+C om te stoppen")

    while True:
        try:
            run_scanner()
            log.info(f"Volgende scan over {SCANNER_INTERVAL // 60} minuten...")
            time.sleep(SCANNER_INTERVAL)
        except KeyboardInterrupt:
            log.info("Scanner gestopt door gebruiker")
            break
        except Exception as e:
            log.error(f"Onverwachte error: {e}")
            time.sleep(60)  # Wacht 1 minuut bij error
