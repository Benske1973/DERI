import sqlite3

def setup():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    
    # Tabel voor de huidige signalen/scans (SMC)
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
    
    # ==========================================
    # MACD MONEY MAP TABELLEN
    # ==========================================
    
    # MACD Signalen tabel
    c.execute('''CREATE TABLE IF NOT EXISTS macd_signals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  signal TEXT,
                  signal_type TEXT,
                  confidence INTEGER,
                  entry_price REAL,
                  stop_loss REAL,
                  take_profit REAL,
                  trend_bias TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  status TEXT DEFAULT 'ACTIVE')''')
    
    # MACD Alerts historie
    c.execute('''CREATE TABLE IF NOT EXISTS macd_alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  alert_type TEXT,
                  message TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # MACD Trade log (specifiek voor MACD Money Map trades)
    c.execute('''CREATE TABLE IF NOT EXISTS macd_trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  signal_type TEXT,
                  entry_price REAL,
                  stop_loss REAL,
                  take_profit REAL,
                  entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                  exit_price REAL,
                  exit_time DATETIME,
                  pnl_percent REAL,
                  result TEXT,
                  notes TEXT)''')
                  
    conn.commit()
    conn.close()
    print("✅ Database geïnitialiseerd met alle tabellen:")
    print("   - signals (SMC)")
    print("   - trade_log")
    print("   - macd_signals")
    print("   - macd_alerts")
    print("   - macd_trades")

if __name__ == "__main__":
    setup()