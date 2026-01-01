import requests, pandas as pd, pandas_ta as ta, sqlite3, time

def calculate_atr(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Bereken Average True Range"""
    high = df['h']
    low = df['l']
    close = df['c']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    
    return atr

def detect_smc_setup(df):
    # Bereken ATR voor Order Block dikte
    atr = calculate_atr(df, period=200)
    
    # Loop van nieuw naar oud
    for i in range(len(df)-1, 5, -1):
        current_atr = atr.iloc[i-2]
        
        # Bullish FVG: Low[i] > High[i-2]
        if df['l'].iloc[i] > df['h'].iloc[i-2]:
            fvg_top = df['l'].iloc[i]
            fvg_bottom = df['h'].iloc[i-2]
            # Order Block volgens BigBeluga: top = high[2], bottom = high[2] - atr
            ob_top = df['h'].iloc[i-2]
            ob_bottom = df['h'].iloc[i-2] - current_atr
            return "BULLISH", fvg_top, fvg_bottom, ob_top, ob_bottom
        
        # Bearish FVG: High[i] < Low[i-2]
        if df['h'].iloc[i] < df['l'].iloc[i-2]:
            fvg_top = df['l'].iloc[i-2]
            fvg_bottom = df['h'].iloc[i]
            # Order Block volgens BigBeluga: top = low[2] + atr, bottom = low[2]
            ob_top = df['l'].iloc[i-2] + current_atr
            ob_bottom = df['l'].iloc[i-2]
            return "BEARISH", fvg_top, fvg_bottom, ob_top, ob_bottom
    
    return None, 0, 0, 0, 0

def run_scanner():
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    # Selecteer een lijst met actieve paren
    symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'AVAX-USDT', 'DOT-USDT', 'LINK-USDT', 'MATIC-USDT']

    for sym in symbols:
        try:
            url = f"https://api.kucoin.com/api/v1/market/candles?symbol={sym}&type=4h"
            data = requests.get(url).json()['data']
            df = pd.DataFrame(data, columns=['ts','o','c','h','l','v','a']).astype(float).iloc[::-1]
            
            # Trend filter
            df['ema50'] = ta.ema(df['c'], length=50)
            if df['c'].iloc[-1] < df['ema50'].iloc[-1]: continue # Alleen bullish in uptrend

            setup_type, ft, fb, ot, ob = detect_smc_setup(df)
            
            if setup_type:
                c.execute("INSERT OR REPLACE INTO signals (symbol, trend, fvg_top, fvg_bottom, ob_top, ob_bottom, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (sym, setup_type, ft, fb, ot, ob, 'SCANNING'))
                print(f"🔎 POI opgeslagen voor {sym} (FVG: {fb}-{ft})")
        except: continue
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    while True:
        print("Scannen naar nieuwe HTF POI's...")
        run_scanner()
        time.sleep(900) # Elke 15 min