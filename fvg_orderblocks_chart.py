# fvg_orderblocks_chart.py - FVG Order Blocks Chart met Heikin Ashi
# Gebaseerd op BigBeluga's TradingView indicator
# Web interface met KuCoin coin selector en timeframe keuze

from flask import Flask, render_template_string, request, jsonify
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
#                      CONFIGURATIE
# ═══════════════════════════════════════════════════════════════

KUCOIN_API = "https://api.kucoin.com"

# FVG Settings (zelfde als Pine Script)
DEFAULT_FILTER_PCT = 0.5    # Filter gaps by %
BOX_AMOUNT = 6              # Max aantal blocks
SHOW_BROKEN = False         # Toon broken blocks

# Colors
COL_BULL = "rgba(20, 190, 148, 0.4)"   # #14be94
COL_BEAR = "rgba(194, 25, 25, 0.4)"    # #c21919
COL_BULL_BORDER = "#14be94"
COL_BEAR_BORDER = "#c21919"


# ═══════════════════════════════════════════════════════════════
#                      DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class OrderBlock:
    start_idx: int
    top: float
    bottom: float
    is_bull: bool
    gap_pct: float
    broken: bool = False
    break_idx: Optional[int] = None


# ═══════════════════════════════════════════════════════════════
#                      KUCOIN API
# ═══════════════════════════════════════════════════════════════

def get_all_symbols() -> List[str]:
    """Haal alle USDT trading pairs op"""
    try:
        r = requests.get(f"{KUCOIN_API}/api/v1/symbols", timeout=10)
        data = r.json()['data']

        symbols = []
        for m in data:
            if (m['quoteCurrency'] == 'USDT' and
                m['enableTrading'] and
                not any(x in m['symbol'] for x in ['3L-', '3S-', 'UP-', 'DOWN-', 'BULL-', 'BEAR-'])):
                symbols.append(m['symbol'])

        return sorted(symbols)
    except Exception as e:
        print(f"Error getting symbols: {e}")
        return []


def get_candles(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Haal candle data op van KuCoin"""
    try:
        url = f"{KUCOIN_API}/api/v1/market/candles?symbol={symbol}&type={timeframe}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if data.get('code') != '200000' or not data.get('data'):
            return pd.DataFrame()

        # KuCoin format: [timestamp, open, close, high, low, volume, turnover]
        df = pd.DataFrame(data['data'], columns=['ts', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        df = df.astype({'ts': int, 'open': float, 'close': float, 'high': float,
                        'low': float, 'volume': float, 'turnover': float})

        # Reverse (oldest first)
        df = df.iloc[::-1].reset_index(drop=True)

        # Convert timestamp
        df['datetime'] = pd.to_datetime(df['ts'], unit='s')

        return df.tail(limit)
    except Exception as e:
        print(f"Error getting candles: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
#                      HEIKIN ASHI
# ═══════════════════════════════════════════════════════════════

def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken Heikin Ashi candles"""
    ha = df.copy()

    # HA Close = (Open + High + Low + Close) / 4
    ha['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

    # HA Open = (Previous HA Open + Previous HA Close) / 2
    ha['ha_open'] = 0.0
    ha.loc[0, 'ha_open'] = (df.loc[0, 'open'] + df.loc[0, 'close']) / 2

    for i in range(1, len(ha)):
        ha.loc[i, 'ha_open'] = (ha.loc[i-1, 'ha_open'] + ha.loc[i-1, 'ha_close']) / 2

    # HA High = max(High, HA Open, HA Close)
    ha['ha_high'] = ha[['high', 'ha_open', 'ha_close']].max(axis=1)

    # HA Low = min(Low, HA Open, HA Close)
    ha['ha_low'] = ha[['low', 'ha_open', 'ha_close']].min(axis=1)

    return ha


# ═══════════════════════════════════════════════════════════════
#                      FVG DETECTION (BigBeluga Logic)
# ═══════════════════════════════════════════════════════════════

def detect_fvg_orderblocks(df: pd.DataFrame, filter_pct: float = 0.5) -> List[OrderBlock]:
    """
    Detecteer Fair Value Gaps (FVG) Order Blocks
    Gebaseerd op BigBeluga's Pine Script logica
    """
    blocks = []

    if len(df) < 3:
        return blocks

    # Use regular OHLC for FVG detection
    for i in range(2, len(df)):
        high_2 = df.loc[i-2, 'high']
        low_2 = df.loc[i-2, 'low']
        high_1 = df.loc[i-1, 'high']
        low_1 = df.loc[i-1, 'low']
        high_0 = df.loc[i, 'high']
        low_0 = df.loc[i, 'low']

        # Bullish FVG: high[2] < low AND high[2] < high[1] AND low[2] < low
        # Gap tussen candle -2 high en candle 0 low
        filt_up = (low_0 - high_2) / low_0 * 100 if low_0 > 0 else 0

        is_bull_gap = (high_2 < low_0 and
                       high_2 < high_1 and
                       low_2 < low_0 and
                       filt_up > filter_pct)

        if is_bull_gap:
            blocks.append(OrderBlock(
                start_idx=i-1,
                top=low_0,          # Top of gap
                bottom=high_2,      # Bottom of gap (order block zone)
                is_bull=True,
                gap_pct=filt_up,
                broken=False
            ))

        # Bearish FVG: low[2] > high AND low[2] > low[1] AND high[2] > high
        filt_dn = (low_2 - high_0) / low_2 * 100 if low_2 > 0 else 0

        is_bear_gap = (low_2 > high_0 and
                       low_2 > low_1 and
                       high_2 > high_0 and
                       filt_dn > filter_pct)

        if is_bear_gap:
            blocks.append(OrderBlock(
                start_idx=i-1,
                top=low_2,          # Top of order block zone
                bottom=high_0,      # Bottom of gap
                is_bull=False,
                gap_pct=filt_dn,
                broken=False
            ))

    # Check for broken blocks
    for block in blocks:
        for i in range(block.start_idx + 2, len(df)):
            if block.is_bull:
                # Bullish block broken when price goes below bottom
                if df.loc[i, 'high'] < block.bottom:
                    block.broken = True
                    block.break_idx = i
                    break
            else:
                # Bearish block broken when price goes above top
                if df.loc[i, 'low'] > block.top:
                    block.broken = True
                    block.break_idx = i
                    break

    # Filter: keep only unbroken or limit amount
    if not SHOW_BROKEN:
        blocks = [b for b in blocks if not b.broken]

    # Keep last N blocks per type
    bull_blocks = [b for b in blocks if b.is_bull][-BOX_AMOUNT:]
    bear_blocks = [b for b in blocks if not b.is_bull][-BOX_AMOUNT:]

    return bull_blocks + bear_blocks


# ═══════════════════════════════════════════════════════════════
#                      CHART CREATION
# ═══════════════════════════════════════════════════════════════

def create_chart(symbol: str, timeframe: str, filter_pct: float = 0.5, use_heikin_ashi: bool = True) -> str:
    """Maak interactieve Plotly chart met FVG Order Blocks"""

    # Get data
    df = get_candles(symbol, timeframe, limit=300)
    if df.empty:
        return "<p>Error loading data</p>"

    # Calculate Heikin Ashi
    df = calculate_heikin_ashi(df)

    # Detect FVG Order Blocks (always use regular candles for detection)
    blocks = detect_fvg_orderblocks(df, filter_pct)

    # Choose which candles to display
    if use_heikin_ashi:
        o_col, h_col, l_col, c_col = 'ha_open', 'ha_high', 'ha_low', 'ha_close'
    else:
        o_col, h_col, l_col, c_col = 'open', 'high', 'low', 'close'

    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.8, 0.2])

    # Candlestick colors
    colors = ['#14be94' if df.loc[i, c_col] >= df.loc[i, o_col] else '#c21919'
              for i in df.index]

    # Add candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df['datetime'],
            open=df[o_col],
            high=df[h_col],
            low=df[l_col],
            close=df[c_col],
            name='Price',
            increasing_line_color='#14be94',
            decreasing_line_color='#c21919',
            increasing_fillcolor='#14be94',
            decreasing_fillcolor='#c21919',
        ),
        row=1, col=1
    )

    # Add Order Blocks as rectangles
    for block in blocks:
        if block.start_idx >= len(df):
            continue

        x0 = df.loc[block.start_idx, 'datetime']
        x1 = df.loc[len(df)-1, 'datetime']  # Extend to current

        color = COL_BULL if block.is_bull else COL_BEAR
        border = COL_BULL_BORDER if block.is_bull else COL_BEAR_BORDER

        # Adjust opacity if broken
        if block.broken:
            color = "rgba(128, 128, 128, 0.3)"
            border = "#808080"

        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=block.bottom, y1=block.top,
            fillcolor=color,
            line=dict(color=border, width=1),
            row=1, col=1
        )

        # Add percentage label
        label_text = f"{block.gap_pct:.2f}%"
        fig.add_annotation(
            x=x1,
            y=(block.top + block.bottom) / 2,
            text=label_text,
            showarrow=False,
            font=dict(size=10, color='white'),
            bgcolor=border,
            row=1, col=1
        )

    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=df['datetime'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )

    # Update layout
    candle_type = "Heikin Ashi" if use_heikin_ashi else "Regular"
    fig.update_layout(
        title=f'{symbol} - {timeframe} - {candle_type} - FVG Order Blocks',
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        xaxis_rangeslider_visible=False,
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    # Update axes
    fig.update_xaxes(gridcolor='#1e222d', showgrid=True)
    fig.update_yaxes(gridcolor='#1e222d', showgrid=True)

    return fig.to_html(full_html=False, include_plotlyjs='cdn')


# ═══════════════════════════════════════════════════════════════
#                      WEB INTERFACE
# ═══════════════════════════════════════════════════════════════

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>FVG Order Blocks Scanner</title>
    <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #131722;
            color: #d1d4dc;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #14be94;
            text-align: center;
            margin-bottom: 20px;
        }
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: flex-end;
            background: #1e222d;
            padding: 20px;
            border-radius: 8px;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            color: #787b86;
            font-size: 12px;
            text-transform: uppercase;
        }
        select, input {
            background: #2a2e39;
            border: 1px solid #363a45;
            color: #d1d4dc;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 14px;
            min-width: 150px;
        }
        select:focus, input:focus {
            outline: none;
            border-color: #14be94;
        }
        .select2-container--default .select2-selection--single {
            background: #2a2e39;
            border: 1px solid #363a45;
            border-radius: 4px;
            height: 40px;
        }
        .select2-container--default .select2-selection--single .select2-selection__rendered {
            color: #d1d4dc;
            line-height: 40px;
            padding-left: 15px;
        }
        .select2-container--default .select2-selection--single .select2-selection__arrow {
            height: 38px;
        }
        .select2-dropdown {
            background: #2a2e39;
            border: 1px solid #363a45;
        }
        .select2-container--default .select2-search--dropdown .select2-search__field {
            background: #1e222d;
            border: 1px solid #363a45;
            color: #d1d4dc;
        }
        .select2-container--default .select2-results__option--highlighted[aria-selected] {
            background: #14be94;
        }
        .select2-results__option {
            color: #d1d4dc;
        }
        button {
            background: #14be94;
            color: #131722;
            border: none;
            padding: 10px 30px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover {
            background: #0fa07e;
        }
        button:disabled {
            background: #363a45;
            cursor: not-allowed;
        }
        .chart-container {
            background: #131722;
            border-radius: 8px;
            overflow: hidden;
        }
        .loading {
            text-align: center;
            padding: 100px;
            color: #787b86;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .checkbox-group input[type="checkbox"] {
            width: 18px;
            height: 18px;
            min-width: 18px;
        }
        .legend {
            display: flex;
            gap: 30px;
            justify-content: center;
            margin-top: 15px;
            font-size: 13px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .legend-box {
            width: 20px;
            height: 14px;
            border-radius: 2px;
        }
        .bull-box { background: rgba(20, 190, 148, 0.6); border: 1px solid #14be94; }
        .bear-box { background: rgba(194, 25, 25, 0.6); border: 1px solid #c21919; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 FVG Order Blocks [BigBeluga Style]</h1>

        <div class="controls">
            <div class="control-group">
                <label>Symbol (KuCoin)</label>
                <select id="symbol" style="width: 200px;">
                    {% for s in symbols %}
                    <option value="{{ s }}" {% if s == selected_symbol %}selected{% endif %}>{{ s }}</option>
                    {% endfor %}
                </select>
            </div>

            <div class="control-group">
                <label>Timeframe</label>
                <select id="timeframe">
                    <option value="1min" {% if timeframe == '1min' %}selected{% endif %}>1 Minute</option>
                    <option value="5min" {% if timeframe == '5min' %}selected{% endif %}>5 Minutes</option>
                    <option value="15min" {% if timeframe == '15min' %}selected{% endif %}>15 Minutes</option>
                    <option value="30min" {% if timeframe == '30min' %}selected{% endif %}>30 Minutes</option>
                    <option value="1hour" {% if timeframe == '1hour' %}selected{% endif %}>1 Hour</option>
                    <option value="4hour" {% if timeframe == '4hour' %}selected{% endif %}>4 Hours</option>
                    <option value="1day" {% if timeframe == '1day' %}selected{% endif %}>1 Day</option>
                </select>
            </div>

            <div class="control-group">
                <label>Filter Gap %</label>
                <input type="number" id="filter" value="{{ filter_pct }}" step="0.1" min="0" max="10" style="width: 100px;">
            </div>

            <div class="control-group">
                <label>&nbsp;</label>
                <div class="checkbox-group">
                    <input type="checkbox" id="heikin" {% if use_ha %}checked{% endif %}>
                    <label for="heikin" style="text-transform: none; font-size: 14px;">Heikin Ashi</label>
                </div>
            </div>

            <div class="control-group">
                <label>&nbsp;</label>
                <button id="loadBtn" onclick="loadChart()">Load Chart</button>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-box bull-box"></div>
                <span>Bullish FVG (Support Zone)</span>
            </div>
            <div class="legend-item">
                <div class="legend-box bear-box"></div>
                <span>Bearish FVG (Resistance Zone)</span>
            </div>
        </div>

        <div class="chart-container" id="chart">
            {{ chart_html|safe }}
        </div>
    </div>

    <script>
        $(document).ready(function() {
            $('#symbol').select2({
                placeholder: 'Search coin...',
                allowClear: true
            });
        });

        function loadChart() {
            const btn = document.getElementById('loadBtn');
            const chart = document.getElementById('chart');

            btn.disabled = true;
            btn.textContent = 'Loading...';
            chart.innerHTML = '<div class="loading">Loading chart...</div>';

            const symbol = document.getElementById('symbol').value;
            const timeframe = document.getElementById('timeframe').value;
            const filter = document.getElementById('filter').value;
            const heikin = document.getElementById('heikin').checked;

            fetch(`/chart?symbol=${symbol}&timeframe=${timeframe}&filter=${filter}&ha=${heikin}`)
                .then(response => response.text())
                .then(html => {
                    chart.innerHTML = html;
                    btn.disabled = false;
                    btn.textContent = 'Load Chart';
                })
                .catch(err => {
                    chart.innerHTML = '<div class="loading">Error loading chart</div>';
                    btn.disabled = false;
                    btn.textContent = 'Load Chart';
                });
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    symbols = get_all_symbols()
    selected = request.args.get('symbol', 'BTC-USDT')
    timeframe = request.args.get('timeframe', '15min')
    filter_pct = float(request.args.get('filter', 0.5))
    use_ha = request.args.get('ha', 'true').lower() == 'true'

    chart_html = create_chart(selected, timeframe, filter_pct, use_ha)

    return render_template_string(
        HTML_TEMPLATE,
        symbols=symbols,
        selected_symbol=selected,
        timeframe=timeframe,
        filter_pct=filter_pct,
        use_ha=use_ha,
        chart_html=chart_html
    )


@app.route('/chart')
def get_chart():
    symbol = request.args.get('symbol', 'BTC-USDT')
    timeframe = request.args.get('timeframe', '15min')
    filter_pct = float(request.args.get('filter', 0.5))
    use_ha = request.args.get('ha', 'true').lower() == 'true'

    return create_chart(symbol, timeframe, filter_pct, use_ha)


@app.route('/symbols')
def get_symbols():
    return jsonify(get_all_symbols())


if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           📊 FVG ORDER BLOCKS CHART [BigBeluga Style]                        ║
║                                                                              ║
║  Open browser: http://localhost:5000                                         ║
║                                                                              ║
║  Features:                                                                   ║
║  • Heikin Ashi candles                                                       ║
║  • FVG Order Blocks (bullish/bearish zones)                                  ║
║  • KuCoin coin selector with search                                          ║
║  • Multiple timeframes                                                       ║
║  • Adjustable gap filter %                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    app.run(debug=True, port=5000)
