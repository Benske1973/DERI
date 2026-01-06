import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import requests

DB_PATH = Path(__file__).resolve().parent / "trading_bot.db"

DEFAULT_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "AVAX-USDT", "DOT-USDT", "LINK-USDT", "MATIC-USDT"]
HTF_CANDLE_TYPE = os.getenv("HTF_CANDLE_TYPE", "4h")  # KuCoin candle type, e.g. 4h, 1day
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", ",".join(DEFAULT_SYMBOLS)).split(",") if s.strip()]
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", "900"))

def detect_smc_setup(df):
    # Loop van nieuw naar oud
    for i in range(len(df)-1, 5, -1):
        # Bullish FVG: Low[i] > High[i-2]
        if df['l'].iloc[i] > df['h'].iloc[i-2]:
            fvg_top = df['l'].iloc[i]
            fvg_bottom = df['h'].iloc[i-2]
            # Order Block is de bearish kaars vòòr de beweging
            ob_top = df['h'].iloc[i-2]
            ob_bottom = df['l'].iloc[i-2]
            return "BULLISH", fvg_top, fvg_bottom, ob_top, ob_bottom
    return None, 0, 0, 0, 0

def run_scanner():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Selecteer een lijst met actieve paren
    for sym in SYMBOLS:
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={sym}&type={HTF_CANDLE_TYPE}"
            resp = requests.get(url, timeout=15)
            payload = resp.json()
            data = payload["data"]
            df = pd.DataFrame(data, columns=['ts','o','c','h','l','v','a']).astype(float).iloc[::-1]
            
            # Trend filter
            df['ema50'] = ta.ema(df['c'], length=50)
            if df['c'].iloc[-1] < df['ema50'].iloc[-1]: continue # Alleen bullish in uptrend

            setup_type, ft, fb, ot, ob = detect_smc_setup(df)
            
            if setup_type:
                c.execute("INSERT OR REPLACE INTO signals (symbol, trend, fvg_top, fvg_bottom, ob_top, ob_bottom, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (sym, setup_type, ft, fb, ot, ob, 'SCANNING'))
                print(f"🔎 POI opgeslagen voor {sym} (FVG: {fb}-{ft})")
        except Exception as e:
            print(f"Scanner error bij {sym}: {e}")
            continue
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    while True:
        print("Scannen naar nieuwe HTF POI's...")
        run_scanner()
        time.sleep(SCAN_INTERVAL_SECONDS)  # default: elke 15 min