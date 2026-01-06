# DERI – KuCoin SMC Scanner/Watcher/Executor (Python)

Dit project draait een eenvoudige pipeline:

- **HTF scanner** (`scanner_htf.py`): zoekt een bullish SMC setup (FVG + OB) op higher timeframe (default `4h`) en schrijft zones naar SQLite.
- **Realtime watcher** (`watcher_ws.py`): monitort KuCoin via WebSocket en markeert een coin als **TAPPED** zodra prijs de FVG-zone raakt.
- **LTF executor** (`executor_ltf.py`): checkt lower timeframe (default `15min`) op bevestiging (ChoCH) en logt een trade (entry/sl/tp) in SQLite.

## Install (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Start

Database initialiseren:

```bash
python3 init_db.py
```

Alles starten (scanner + watcher + executor):

```bash
python3 run_bot.py --init-db
```

Status bekijken:

```bash
python3 check_status.py
```

Trade log bekijken:

```bash
python3 view_stats.py
```

## Config (optioneel via environment variables)

- **`SYMBOLS`**: comma-separated lijst (default: BTC-USDT, ETH-USDT, …)
- **`HTF_CANDLE_TYPE`**: KuCoin candle type (default: `4h`)
- **`SCAN_INTERVAL_SECONDS`**: scanner interval (default: `900`)
- **`LTF_CANDLE_TYPE`**: KuCoin candle type (default: `15min`)
- **`LTF_POLL_INTERVAL_SECONDS`**: executor interval (default: `30`)
- **`WATCHLIST_REFRESH_SECONDS`**: hoe vaak watcher de DB opnieuw inleest (default: `30`)
- **`SWING_SOURCE`**: `high` of `close` (default: `high`)
- **`SWING_LOOKBACK_CANDLES`**: lookback (default: `15`)