# 🎯 MACD Money Map - Trading Bot

A complete Python trading system implementing the **MACD Money Map** strategy with 3 interconnected systems for high-probability trade setups.

## 📊 The 3 Systems

### System 1: Trend System
Catches the big moves that run for days or weeks.

- **Zero Line Foundation**: MACD above 0 = ONLY buys, below 0 = ONLY sells
- **Distance Rule**: Only take crossovers far from zero (>0.5 or <-0.5)
- **Wait Rule**: Wait 2-3 candles after crossover for confirmation

### System 2: Reversal System
Catches major turning points before they happen.

- **Divergence Detection**: 
  - Bearish: Price higher high + MACD lower high
  - Bullish: Price lower low + MACD higher low
- **Histogram Patterns**:
  - The Flip: First opposite color bar
  - Shrinking Tower: Bars getting smaller
  - Zero Bounce: Histogram bounces off zero

### System 3: Confirmation System
Filters bad trades and confirms winners.

- **Triple Timeframe Stack**: Daily → 4H → 1H (all must agree)
- **Price Action Confirmation**: Signals at support/resistance levels

## 🚀 Quick Start

```bash
# 1. Initialize the database
python init_db.py

# 2. Run the morning scan
python macd_money_map.py

# 3. Continuous monitoring
python macd_monitor.py

# 4. Visualize a specific symbol
python macd_visualizer.py BTC/USDT
```

## 📁 Files

| File | Description |
|------|-------------|
| `macd_money_map.py` | Main scanner with all 3 systems |
| `macd_monitor.py` | Real-time monitoring with alerts |
| `macd_visualizer.py` | Chart visualization of analysis |
| `init_db.py` | Database initialization |
| `check_status.py` | View current signal status |
| `view_stats.py` | View trading statistics |

## ⚙️ Configuration

Edit `CONFIG` in `macd_money_map.py`:

```python
CONFIG = {
    'symbols': ['BTC/USDT', 'ETH/USDT', ...],
    'timeframes': {
        'trend': '1d',    # Daily for trend bias
        'setup': '4h',    # 4H for setups
        'entry': '1h'     # 1H for entry
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'trend_system': {
        'distance_threshold': 0.5,  # Distance from zero line
        'wait_candles': 2           # Candles to wait after crossover
    },
    'risk': {
        'risk_reward': 2.0          # 2R target
    }
}
```

## 📈 The 5-Minute Morning Routine

1. **Check the Trend**: Look at Daily MACD (above/below zero = your bias)
2. **Find the Setup**: Scan for crossovers or divergence on 4H
3. **Confirm Everything**: Verify all 3 timeframes align
4. **Execute**: Entry at candle close, SL at swing point, TP at 2R

## 🎯 A+ Setup Criteria

A trade is **A+ quality** when:
- ✅ Daily MACD confirms trend direction
- ✅ 4H shows valid crossover OR divergence
- ✅ 1H histogram confirms with a flip
- ✅ Price is at key support/resistance level
- ✅ Confidence score ≥ 90%

## 📊 Database Tables

| Table | Purpose |
|-------|---------|
| `macd_signals` | Active MACD signals |
| `macd_alerts` | Alert history |
| `macd_trades` | Trade log with P&L |
| `signals` | SMC signals (legacy) |
| `trade_log` | General trade log |

## 🔔 Monitor Commands

```bash
# Single scan
python macd_monitor.py --once

# Continuous monitoring (default 5 min interval)
python macd_monitor.py

# Custom interval (in seconds)
python macd_monitor.py 60  # Every minute
```

## 📉 Risk Management

- **Entry**: At candle close, never mid-candle
- **Stop Loss**: At recent swing high/low
- **Take Profit**: Always 2R (2x your risk)
- **Position Sizing**: Risk 1-2% per trade
- **Partial Close**: 50% at target, trail the rest

## ⚠️ Disclaimer

This bot is for educational purposes only. Trading involves significant risk. Always do your own research and never trade with money you can't afford to lose.

---

Built with ❤️ using the MACD Money Map strategy
