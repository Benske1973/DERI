import sqlite3
import time
from datetime import datetime

def run_executor():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    
    # Check for NEW signals
    c.execute("SELECT symbol, signal_type, entry_price FROM signals_v6 WHERE status='NEW'")
    new_signals = c.fetchall()
    
    for symbol, sig_type, price in new_signals:
        print(f"🚀 EXECUTING {sig_type} on {symbol} @ {price}")
        
        # Here you would place the real order via ccxt
        # exchange.create_market_order(symbol, side, amount)
        
        # Calculate SL/TP (Simple 1:3 RR or fixed percentage for now)
        # Assuming 2% SL and 6% TP
        sl = price * 0.98 if sig_type == 'BUY' else price * 1.02
        tp = price * 1.06 if sig_type == 'BUY' else price * 0.94
        
        # Log trade
        c.execute('''INSERT INTO trade_log (symbol, entry_price, sl, tp, status, result) 
                     VALUES (?, ?, ?, ?, 'OPEN', ?)''', 
                  (symbol, price, sl, tp, sig_type))
        
        # Update signal status
        c.execute("UPDATE signals_v6 SET status='IN_TRADE', sl=?, tp=? WHERE symbol=?", (sl, tp, symbol))
        
        print(f"✅ Trade logged: {symbol} {sig_type}")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print("Starting V6 Executor...")
    while True:
        run_executor()
        time.sleep(60) # Check every minute
