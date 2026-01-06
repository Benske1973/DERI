import ccxt
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime

# ==========================================
# 1. INSTELLINGEN (V6: The Final Sniper)
# ==========================================
# Symbols to scan
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT', 'MATIC/USDT']
TIMEFRAME = '4h'

# --- MACD Instellingen ---
FAST_LEN = 12
SLOW_LEN = 26
SIG_LEN = 9

# --- V5 Lagged Breakout Instellingen ---
LOOKBACK_PERIOD = 20       # Periode voor basislijn ruis
SENSITIVITY = 3.0          # Signaal moet 3x sterker zijn dan gemiddelde ruis

# --- NIEUW: V6 RSI FILTER ---
USE_RSI_FILTER = True
RSI_THRESHOLD = 50

def calc_ema(source, length):
    return source.ewm(span=length, adjust=False).mean()

def calculate_indicators(df):
    # A. MACD
    df['ma_fast'] = calc_ema(df['close'], FAST_LEN)
    df['ma_slow'] = calc_ema(df['close'], SLOW_LEN)
    df['macd'] = df['ma_fast'] - df['ma_slow']
    df['signal'] = calc_ema(df['macd'], SIG_LEN)
    df['hist'] = df['macd'] - df['signal']

    # B. Lagged Volatility Baseline
    df['abs_hist'] = df['hist'].abs()
    df['baseline_noise'] = df['abs_hist'].rolling(window=LOOKBACK_PERIOD).mean().shift(1)
    df['baseline_noise'] = df['baseline_noise'].replace(0, 0.00000001)

    # C. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    return df

def run_scanner():
    print(f"[{datetime.now()}] Starting V6 Scan...")
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    
    exchange = ccxt.kucoin()
    
    for symbol in SYMBOLS:
        try:
            # Kucoin uses dash sometimes, but ccxt normalizes to slash. 
            # Check if we need to convert format. 
            # CCXT usually expects 'BTC/USDT'.
            
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=100)
            if not ohlcv:
                print(f"No data for {symbol}")
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df = calculate_indicators(df)
            
            # Check last completed candle (index -2, as -1 is current forming candle)
            # Actually, fetch_ohlcv usually returns the latest candle which might be incomplete.
            # To be safe, we check index -2.
            
            i = len(df) - 2
            if i < LOOKBACK_PERIOD:
                continue
                
            hist_curr = df['hist'].iloc[i]
            hist_prev = df['hist'].iloc[i-1]
            baseline = df['baseline_noise'].iloc[i]
            rsi_curr = df['rsi'].iloc[i]
            close_price = df['close'].iloc[i]
            
            is_explosion = hist_curr > (baseline * SENSITIVITY)
            pass_rsi_buy = (rsi_curr > RSI_THRESHOLD) if USE_RSI_FILTER else True
            
            is_dump = hist_curr < -(baseline * SENSITIVITY)
            pass_rsi_sell = (rsi_curr < 50) if USE_RSI_FILTER else True
            
            signal_type = None
            
            # BUY SIGNAL
            if hist_curr > 0 and hist_curr > hist_prev and is_explosion and pass_rsi_buy:
                signal_type = 'BUY'
            
            # SELL SIGNAL
            elif hist_curr < 0 and hist_curr < hist_prev and is_dump and pass_rsi_sell:
                signal_type = 'SELL'
                
            if signal_type:
                print(f"✅ SIGNAL FOUND: {symbol} - {signal_type} @ {close_price}")
                # Upsert into DB
                c.execute('''INSERT OR REPLACE INTO signals_v6 
                             (symbol, signal_type, entry_price, status) 
                             VALUES (?, ?, ?, 'NEW')''', 
                          (symbol, signal_type, close_price))
            else:
                # print(f"No signal for {symbol}")
                pass

        except Exception as e:
            print(f"Error scanning {symbol}: {e}")
            
    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] Scan complete.")

if __name__ == "__main__":
    while True:
        run_scanner()
        time.sleep(900) # Scan every 15 minutes (since it's 4h timeframe, we don't need to be super frequent, but 15m is good to catch the close)
