# DERI

## Scripts

### V6: The Final Sniper (MACD + Lagged Breakout + RSI)

Install deps:

```bash
pip install -r requirements.txt
```

Run (saves a PNG to `output/` by default):

```bash
python3 scanner_v6_sniper.py --symbol PRCL/USDT --timeframe 4h
```

Disable RSI filter (show all breakouts):

```bash
python3 scanner_v6_sniper.py --symbol PRCL/USDT --timeframe 4h --no-rsi-filter
```