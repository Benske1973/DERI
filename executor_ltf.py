import sqlite3
import requests
import pandas as pd
import time

def check_ltf_confirmation():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    # Haal munten op die de zone hebben geraakt
    c.execute("SELECT symbol, ob_top, ob_bottom FROM signals WHERE status='TAPPED'")
    active_targets = c.fetchall()

    for sym, ob_top, ob_bottom in active_targets:
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={sym}&type=15min"
            data = requests.get(url).json()['data']
            df = pd.DataFrame(data, columns=['ts','o','c','h','l','v','a']).astype(float).iloc[::-1]
            
            # 1. Swing High bepalen (Body Close Filter)
            m15_high_to_break = df['h'].iloc[-15:-1].max()
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
        time.sleep(30)