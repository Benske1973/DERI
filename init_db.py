# init_db.py - Database initialisatie met verbeterd schema
import sqlite3
import os
from config import DATABASE_PATH

def setup():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # Tabel voor actieve signalen/scans
    c.execute("DROP TABLE IF EXISTS signals")
    c.execute('''CREATE TABLE signals (
        symbol TEXT PRIMARY KEY,
        trend TEXT,
        fvg_top REAL,
        fvg_bottom REAL,
        ob_top REAL,
        ob_bottom REAL,
        status TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Tabel voor trade logboek
    c.execute('''CREATE TABLE IF NOT EXISTS trade_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        sl REAL,
        tp REAL,
        position_size REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'OPEN',
        exit_price REAL,
        exit_timestamp DATETIME,
        result TEXT,
        pnl_percent REAL,
        pnl_usdt REAL
    )''')

    # Tabel voor symbol cache (coins met volume >50K)
    c.execute("DROP TABLE IF EXISTS active_symbols")
    c.execute('''CREATE TABLE active_symbols (
        symbol TEXT PRIMARY KEY,
        volume_24h REAL,
        last_price REAL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # Tabel voor backtest resultaten
    c.execute('''CREATE TABLE IF NOT EXISTS backtest_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT,
        direction TEXT,
        entry_price REAL,
        sl REAL,
        tp REAL,
        result TEXT,
        pnl_percent REAL,
        timestamp DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    conn.commit()
    conn.close()
    print("Database geinitialiseerd met nieuwe schema's.")

if __name__ == "__main__":
    setup()
