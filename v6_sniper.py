import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. INSTELLINGEN (V6: The Final Sniper)
# ==========================================
symbol = 'PRCL/USDT'
timeframe = '4h'

# --- MACD Instellingen ---
fast_len = 12
slow_len = 26
sig_len = 9

# --- V5 Lagged Breakout Instellingen ---
lookback_period = 20       # Periode voor basislijn ruis
sensitivity = 3.0          # Signaal moet 3x sterker zijn dan gemiddelde ruis

# --- NIEUW: V6 RSI FILTER ---
use_rsi_filter = True      # Zet op False om alle breakouts te zien
rsi_threshold = 50         # RSI moet boven 50 zijn om te bevestigen dat kopers de baas zijn

# ==========================================
# 2. DATA OPHALEN
# ==========================================
print(f"Scannen met V6 (RSI + Lagged Breakout) op {symbol} ({timeframe})...")
exchange = ccxt.kucoin()

try:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=500)
    if not ohlcv:
        print("Geen data.")
        exit()
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
except Exception as e:
    print(f"Error: {e}")
    exit()

# ==========================================
# 3. INDICATOREN BEREKENEN
# ==========================================
def calc_ema(source, length):
    return source.ewm(span=length, adjust=False).mean()

# A. MACD
df['ma_fast'] = calc_ema(df['close'], fast_len)
df['ma_slow'] = calc_ema(df['close'], slow_len)
df['macd'] = df['ma_fast'] - df['ma_slow']
df['signal'] = calc_ema(df['macd'], sig_len)
df['hist'] = df['macd'] - df['signal']

# B. Lagged Volatility Baseline (De "Smart" Threshold)
df['abs_hist'] = df['hist'].abs()
# Shift(1) is cruciaal: Vergelijk knal van NU met stilte van GISTEREN
df['baseline_noise'] = df['abs_hist'].rolling(window=lookback_period).mean().shift(1)
df['baseline_noise'] = df['baseline_noise'].replace(0, 0.00000001)

# C. RSI (Relative Strength Index) - Handmatig berekend voor snelheid
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ==========================================
# 4. SIGNALEN GENEREREN
# ==========================================
buy_signals = [np.nan] * len(df)
sell_signals = [np.nan] * len(df)

start_index = lookback_period + 2

for i in range(start_index, len(df)):
    
    # Waarden ophalen
    hist_curr = df['hist'].iloc[i]
    hist_prev = df['hist'].iloc[i-1]
    baseline = df['baseline_noise'].iloc[i]
    rsi_curr = df['rsi'].iloc[i]
    
    # --- FILTERS ---
    # 1. Breakout Check: Is de staaf 3x groter dan normaal?
    is_explosion = hist_curr > (baseline * sensitivity)
    
    # 2. RSI Check: Hebben we momentum? (Boven 50 = Bullish territory)
    pass_rsi = (rsi_curr > rsi_threshold) if use_rsi_filter else True
    
    # --- BUY SIGNAL ---
    # Logica: Groene staaf + Groter dan vorige + Breakout + RSI OK
    if hist_curr > 0 and hist_curr > hist_prev and is_explosion and pass_rsi:
        buy_signals[i] = df['low'].iloc[i] * 0.95

    # --- SELL SIGNAL (Optioneel: RSI < 50) ---
    is_dump = hist_curr < -(baseline * sensitivity)
    pass_rsi_sell = (rsi_curr < 50) if use_rsi_filter else True
    
    if hist_curr < 0 and hist_curr < hist_prev and is_dump and pass_rsi_sell:
        sell_signals[i] = df['high'].iloc[i] * 1.05

df['buy_signal'] = buy_signals
df['sell_signal'] = sell_signals

# ==========================================
# 5. VISUALISATIE
# ==========================================
plt.figure(figsize=(14, 12))

# -- PLOT 1: PRIJS --
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df['timestamp'], df['close'], label='Prijs', color='#787b86', linewidth=1)

valid_buy = df.dropna(subset=['buy_signal'])
valid_sell = df.dropna(subset=['sell_signal'])

ax1.scatter(valid_buy['timestamp'], valid_buy['buy_signal'], 
            color='#00E676', marker='^', s=180, label='Sniper Buy', zorder=5, edgecolors='black')
ax1.scatter(valid_sell['timestamp'], valid_sell['sell_signal'], 
            color='#FF5252', marker='v', s=180, label='Sniper Sell', zorder=5, edgecolors='black')

ax1.set_title(f'{symbol} ({timeframe}) - V6: The Final Sniper (MACD + Volatility + RSI)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.1)

# -- PLOT 2: MACD + THRESHOLD --
ax2 = plt.subplot(3, 1, 2, sharex=ax1)

# Kleuren
conditions = [
    (df['hist'] >= 0) & (df['hist'] > df['hist'].shift(1)),
    (df['hist'] >= 0) & (df['hist'] <= df['hist'].shift(1)),
    (df['hist'] < 0) & (df['hist'] > df['hist'].shift(1)), 
    (df['hist'] < 0) & (df['hist'] <= df['hist'].shift(1))
]
colors = ['#26a69a', '#b2dfdb', '#ffcdd2', '#ff5252']
bar_colors = np.select(conditions, colors, default='#b2dfdb')

ax2.bar(df['timestamp'], df['hist'], color=bar_colors, width=0.06)
threshold_line = df['baseline_noise'] * sensitivity
ax2.plot(df['timestamp'], threshold_line, color='#e91e63', linestyle='-', linewidth=1, label='Breakout Line')
ax2.fill_between(df['timestamp'], -threshold_line, threshold_line, color='#e91e63', alpha=0.05)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.1)

# -- PLOT 3: RSI MONITOR --
ax3 = plt.subplot(3, 1, 3, sharex=ax1)
ax3.plot(df['timestamp'], df['rsi'], color='#9c27b0', label='RSI')
ax3.axhline(50, color='gray', linestyle='--')
ax3.axhline(30, color='green', linestyle=':', alpha=0.5)
ax3.axhline(70, color='red', linestyle=':', alpha=0.5)
ax3.fill_between(df['timestamp'], 50, df['rsi'], where=(df['rsi'] >= 50), color='#9c27b0', alpha=0.1)
ax3.set_ylabel('RSI Momentum')
ax3.legend()
ax3.grid(True, alpha=0.1)

plt.tight_layout()
print("Saving plot to v6_analysis.png...")
plt.savefig('v6_analysis.png')
print("Done.")
