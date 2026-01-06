# 🎯 Crypto Breakout Scanner System

Geautomatiseerd systeem voor het detecteren van breakout opportunities in crypto.

## 🚀 Quick Start

```bash
cd /workspace/sniper_bot

# Full market scan (aanbevolen)
python3 scanner_dashboard.py all

# Specifieke modes
python3 scanner_dashboard.py hunt     # Oversold coins met bounce potential
python3 scanner_dashboard.py active   # Actieve breakouts NU
python3 scanner_dashboard.py volume   # Volume explosies (15min)
```

## 📊 Beschikbare Scanners

### 1. Scanner Dashboard (Aanbevolen)
```bash
python3 scanner_dashboard.py all
```
Combineert alle scans in één overzicht:
- Active breakouts
- Hunt signals (oversold + momentum)
- Volume explosions

### 2. Optimized Hunter
```bash
python3 optimized_hunter.py
```
Gebaseerd op analyse van 142 mega movers (50%+ gains).
Focust op:
- Oversold RSI (< 40) - 51% van mega movers had dit!
- Bottom reversals
- MACD momentum flips

### 3. Continuous Scanner
```bash
# Single scan
python3 continuous_scanner.py

# Continuous mode (elke 5 min)
python3 continuous_scanner.py 5
```
Draait continu en alert alleen op NIEUWE signalen.

### 4. Breakout Hunter
```bash
python3 breakout_hunter.py
```
Klassieke breakout detectie:
- Volume spikes
- Resistance breaks
- Momentum scores

### 5. Mega Mover Analyzer
```bash
python3 mega_mover_analyzer.py
```
Analyseert coins die 50%+ moves hebben gemaakt.
Identificeert patronen voor early detection.

## 📈 Signal Types

### HIGH URGENCY 🚨
- Score 70+
- Active breakout met volume
- RSI < 35 met recovery

### MEDIUM URGENCY 📢
- Score 50-70
- Momentum building
- Near breakout level

### WATCHLIST 👀
- Score 25-50
- Setup forming
- Monitor closely

## 🔑 Key Indicators

| Indicator | Bullish Signal |
|-----------|---------------|
| RSI | < 40 (oversold), recovery starting |
| Volume | > 2x average = significant |
| MACD | Histogram flip positive |
| Price | Within 10% of 7d low |

## 📊 Based on Data Analysis

Van 142 geanalyseerde mega movers (50%+ gains):
- **51%** had oversold RSI (<40) VOOR de move
- **Average days to peak**: 6.8 dagen
- **35%** waren bottom reversals
- **35%** waren resistance breaks

## 💡 Best Practices

1. **Timing**: Run scans meerdere keren per dag
2. **Confirmatie**: Wacht op volume spike bij entry
3. **Stop Loss**: Gebruik 2x ATR onder entry
4. **Take Profit**: Trail na 50%+ gain

## 🛠️ Requirements

```bash
pip install ccxt pandas numpy
```

## 📁 Project Structure

```
sniper_bot/
├── scanner_dashboard.py    # All-in-one scanner
├── optimized_hunter.py     # Data-driven hunter
├── continuous_scanner.py   # 24/7 monitoring
├── breakout_hunter.py      # Classic breakout detection
├── mega_mover_analyzer.py  # Pattern analysis
├── early_detector.py       # Pre-breakout setups
├── live_scanner.py         # SwingTrader signals
└── core/
    ├── backtest.py         # Backtesting engine
    ├── data.py             # Data fetching
    ├── indicators.py       # Technical indicators
    └── strategy.py         # Strategy base class
```

## ⚠️ Disclaimer

Dit is alleen voor educatieve doeleinden. Trade op eigen risico.
