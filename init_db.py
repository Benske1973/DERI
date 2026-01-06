import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trading_bot.db"

def setup():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Tabel voor de huidige signalen/scans
    c.execute("DROP TABLE IF EXISTS signals")
    c.execute('''CREATE TABLE signals 
                 (symbol TEXT PRIMARY KEY, trend TEXT, 
                  fvg_top REAL, fvg_bottom REAL, 
                  ob_top REAL, ob_bottom REAL, status TEXT)''')

    # Tabel voor het logboek (Jouw statistieken)
    c.execute('''CREATE TABLE IF NOT EXISTS trade_log 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  entry_price REAL,
                  sl REAL,
                  tp REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  status TEXT DEFAULT 'OPEN',
                  exit_price REAL,
                  result TEXT)''')
                  
    conn.commit()
    conn.close()
    print("Database & Logboek geïnitialiseerd.")

if __name__ == "__main__":
    setup()