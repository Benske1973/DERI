# KuCoin Multiscanner Papertrader

Een geavanceerde paper trading bot voor KuCoin met multi-symbol scanning, SMC/ICT analyse en real-time price monitoring.

## Features

- **Multi-Symbol Scanning**: Scant automatisch de top 50 KuCoin pairs op volume
- **SMC/ICT Analyse**: Fair Value Gaps, Order Blocks, Break of Structure, Change of Character
- **Paper Trading**: Volledige trading simulatie zonder risico
- **Real-time Monitoring**: WebSocket verbinding voor live prijsupdates
- **Portfolio Management**: Positie sizing, risk management, trailing stops
- **Dashboard**: Terminal-based monitoring interface

## Architectuur

```
├── config.py           # Configuratie & instellingen
├── kucoin_client.py    # KuCoin API client
├── indicators.py       # Technische analyse indicatoren
├── scanner.py          # Multi-symbol scanner engine
├── paper_trader.py     # Paper trading engine
├── strategies.py       # Trading strategieën
├── websocket_feed.py   # Real-time price feed
├── database.py         # SQLite database
├── dashboard.py        # Terminal dashboard
├── main.py             # Main orchestrator
└── requirements.txt    # Python dependencies
```

## Installatie

```bash
# Clone repository
git clone <repo-url>
cd DERI

# Maak virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# of: venv\Scripts\activate  # Windows

# Installeer dependencies
pip install -r requirements.txt
```

## Gebruik

### Start Paper Trader

```bash
# Standaard mode
python main.py

# Met dashboard
python main.py --dashboard

# Alleen scannen
python main.py --scan-only

# Bekijk statistieken
python main.py --stats
```

### Configuratie

Pas `config.py` aan voor je eigen instellingen:

```python
# Risico instellingen
risk_per_trade = 0.01      # 1% risico per trade
max_position_size = 0.10   # Max 10% per positie
max_open_positions = 5     # Max 5 posities tegelijk

# Strategie instellingen
min_score = 70.0           # Minimum signaal score
require_choch = True       # Vereis CHoCH confirmatie
use_session_filter = True  # Filter op trading sessions
```

### Dashboard Commands

```
[S] - Force HTF scan
[R] - Refresh display
[Q] - Quit
```

## Trading Logic

### 1. HTF Scanner (4H)
- Detecteert trend richting via EMAs
- Vindt Fair Value Gaps (FVG)
- Identificeert Order Blocks (OB)
- Analyseert market structure

### 2. Price Monitor (WebSocket)
- Real-time prijsupdates
- Detecteert zone taps
- Triggert LTF confirmatie check

### 3. LTF Confirmatie (15M)
- Change of Character (CHoCH)
- Break of Structure (BOS)
- Momentum & volume confirmatie
- Signal scoring (0-100)

### 4. Trade Execution
- Automatische position sizing
- Stop loss op basis van ATR
- Take profit met R:R ratio
- Partial take profits
- Trailing stops

## Indicatoren

- **Trend**: EMA 9/21/50/200, ADX
- **Momentum**: RSI, MACD, Stochastic
- **Volatility**: ATR, Bollinger Bands
- **Volume**: OBV, Volume SMA, VWAP
- **SMC**: FVG, Order Blocks, Swing Points

## Risk Management

- **Position Sizing**: Gebaseerd op risico percentage
- **Stop Loss**: ATR-based of fixed percentage
- **Take Profit**: R:R ratio (default 1:3)
- **Partial TP**: 33% @ 1R, 33% @ 2R, 34% @ 3R
- **Trailing Stop**: Activatie @ 1.5R, trail @ 0.5R

## Database

SQLite database (`kucoin_papertrader.db`) bevat:
- `signals` - Actieve POI's en signalen
- `trades` - Trade log met P&L
- `portfolio_snapshots` - Historische portfolio waarden
- `daily_stats` - Dagelijkse statistieken

## API Endpoints (Geen API key nodig)

De bot gebruikt alleen publieke KuCoin endpoints:
- `/api/v1/market/candles` - OHLCV data
- `/api/v1/market/allTickers` - Alle tickers
- `/api/v1/bullet-public` - WebSocket token

## Disclaimer

Deze software is uitsluitend bedoeld voor educatieve doeleinden en paper trading.
Gebruik op eigen risico. De ontwikkelaars zijn niet verantwoordelijk voor
eventuele verliezen bij het gebruik van deze software voor live trading.

## Licentie

MIT License
