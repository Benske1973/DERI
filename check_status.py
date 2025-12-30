# check_status.py
import sqlite3

conn = sqlite3.connect('trading_bot.db')
c = conn.cursor()

print("\n--- ACTIEVE POI's (Wachten op Tap) ---")
c.execute("SELECT symbol, trend, fvg_top, fvg_bottom FROM signals WHERE status='SCANNING'")
for row in c.fetchall():
    print(f"Coin: {row[0]} | Trend: {row[1]} | Zone: {row[3]} - {row[2]}")

print("\n--- GETAPTE COINS (Wachten op M15 ChoCH) ---")
c.execute("SELECT symbol FROM signals WHERE status='TAPPED'")
for row in c.fetchall():
    print(f"🔥 {row[0]} is de zone binnengegaan!")

conn.close()