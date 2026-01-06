# DERI

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## V6 "Final Sniper" scanner (ccxt + MACD breakout + RSI)

Runs an on-demand scan and shows/saves a 3-panel chart (price, MACD histogram + breakout threshold, RSI).

```bash
python3 sniper_v6.py --symbol PRCL/USDT --timeframe 4h --exchange kucoin --limit 500
```

Headless / server mode (save chart to file):

```bash
python3 sniper_v6.py --symbol BTC/USDT --timeframe 4h --exchange kucoin --limit 500 --no-plot --save-plot sniper.png
```
