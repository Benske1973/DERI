import sqlite3
import os
import requests
import pandas as pd
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trading_bot.db"

LTF_CANDLE_TYPE = os.getenv("LTF_CANDLE_TYPE", "15min")  # KuCoin candle type, e.g. 15min
LTF_POLL_INTERVAL_SECONDS = int(os.getenv("LTF_POLL_INTERVAL_SECONDS", "30"))
SWING_LOOKBACK_CANDLES = int(os.getenv("SWING_LOOKBACK_CANDLES", "15"))
SWING_SOURCE = os.getenv("SWING_SOURCE", "high").lower()  # high | close

def check_ltf_confirmation():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Haal munten op die de zone hebben geraakt
    c.execute("SELECT symbol, ob_top, ob_bottom FROM signals WHERE status='TAPPED'")
    active_targets = c.fetchall()

    for sym, ob_top, ob_bottom in active_targets:
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={sym}&type={LTF_CANDLE_TYPE}"
            resp = requests.get(url, timeout=15)
            payload = resp.json()
            data = payload["data"]
            df = pd.DataFrame(data, columns=['ts','o','c','h','l','v','a']).astype(float).iloc[::-1]
            
            # 1. Swing High bepalen (bron instelbaar: high of close)
            if SWING_SOURCE == "close":
                swing_series = df["c"]
            else:
                swing_series = df["h"]

            m15_high_to_break = swing_series.iloc[-SWING_LOOKBACK_CANDLES:-1].max()
            current_close = df['c'].iloc[-1]

            # 2. De Body Close ChoCH check
            if current_close > m15_high_to_break:
                entry = ob_top
                sl = ob_bottom
                risk = entry - sl
                tp = entry + (risk * 3) # Mark Douglas 1:3 Ratio

                print(f"🚀 ChoCH BEVESTIGD op {sym}. Logging trade...")
                
                # Update de signalen tabel naar IN_TRADE
                c.execute("UPDATE signals SET status='IN_TRADE' WHERE symbol=?", (sym,))
                
                # SCHRIJF NAAR LOGBOEK (SQLite)
                c.execute('''INSERT INTO trade_log (symbol, entry_price, sl, tp) 
                             VALUES (?, ?, ?, ?)''', (sym, entry, sl, tp))
                
                print(f"📝 GECONTROLEERD & GELOGD: {sym} | Entry: {round(entry,4)} | SL: {round(sl,4)} | TP: {round(tp,4)}")
                
        except Exception as e:
            print(f"Error bij {sym}: {e}")
            continue
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    while True:
        check_ltf_confirmation()
        time.sleep(LTF_POLL_INTERVAL_SECONDS)