# view_stats.py
import sqlite3

conn = sqlite3.connect('trading_bot.db')
c = conn.cursor()
c.execute("SELECT * FROM trade_log")
trades = c.fetchall()

print(f"{'ID':<4} | {'Symbool':<10} | {'Entry':<10} | {'SL':<10} | {'TP':<10} | {'Datum'}")
print("-" * 70)
for t in trades:
    print(f"{t[0]:<4} | {t[1]:<10} | {t[2]:<10.4f} | {t[3]:<10.4f} | {t[4]:<10.4f} | {t[5]}")

conn.close()