# dashboard.py - Web Dashboard voor trading bot monitoring
from flask import Flask, render_template_string, jsonify
import sqlite3
from datetime import datetime, timedelta
from config import DASHBOARD_HOST, DASHBOARD_PORT, DATABASE_PATH

app = Flask(__name__)

# HTML Template
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="nl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DERI Trading Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label { color: #888; font-size: 0.9em; text-transform: uppercase; }
        .positive { color: #00ff88; }
        .negative { color: #ff4757; }
        .neutral { color: #ffa502; }
        .section {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .section h2 {
            margin-bottom: 20px;
            color: #00d9ff;
            border-bottom: 2px solid #00d9ff;
            padding-bottom: 10px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        th { color: #00d9ff; font-weight: 600; }
        tr:hover { background: rgba(255,255,255,0.05); }
        .badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }
        .badge-bullish { background: rgba(0,255,136,0.2); color: #00ff88; }
        .badge-bearish { background: rgba(255,71,87,0.2); color: #ff4757; }
        .badge-scanning { background: rgba(0,217,255,0.2); color: #00d9ff; }
        .badge-tapped { background: rgba(255,165,2,0.2); color: #ffa502; }
        .badge-intrade { background: rgba(155,89,182,0.2); color: #9b59b6; }
        .badge-win { background: rgba(0,255,136,0.2); color: #00ff88; }
        .badge-loss { background: rgba(255,71,87,0.2); color: #ff4757; }
        .badge-open { background: rgba(255,165,2,0.2); color: #ffa502; }
        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border: none;
            padding: 15px 30px;
            border-radius: 30px;
            color: #1a1a2e;
            font-weight: bold;
            cursor: pointer;
            font-size: 1em;
        }
        .refresh-btn:hover { transform: scale(1.05); }
        .last-update { text-align: center; color: #666; margin-top: 20px; }
        @media (max-width: 768px) {
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            table { font-size: 0.85em; }
            th, td { padding: 8px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>DERI Trading Bot</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Actieve Symbols</div>
                <div class="stat-value" id="active-symbols">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Setups</div>
                <div class="stat-value neutral" id="active-setups">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Open Trades</div>
                <div class="stat-value" id="open-trades">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value" id="win-rate">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Totale PnL</div>
                <div class="stat-value" id="total-pnl">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Trades Vandaag</div>
                <div class="stat-value" id="trades-today">-</div>
            </div>
        </div>

        <div class="section">
            <h2>Actieve Signalen</h2>
            <table id="signals-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Trend</th>
                        <th>FVG Zone</th>
                        <th>Status</th>
                        <th>Updated</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <div class="section">
            <h2>Recente Trades</h2>
            <table id="trades-table">
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>SL</th>
                        <th>TP</th>
                        <th>PnL</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>

        <p class="last-update">Last update: <span id="last-update">-</span></p>
    </div>

    <button class="refresh-btn" onclick="refreshData()">Refresh</button>

    <script>
        async function fetchData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        function updateDashboard(data) {
            // Update stats
            document.getElementById('active-symbols').textContent = data.stats.active_symbols;
            document.getElementById('active-setups').textContent = data.stats.active_setups;
            document.getElementById('open-trades').textContent = data.stats.open_trades;
            document.getElementById('trades-today').textContent = data.stats.trades_today;

            const winRate = document.getElementById('win-rate');
            winRate.textContent = data.stats.win_rate.toFixed(1) + '%';
            winRate.className = 'stat-value ' + (data.stats.win_rate >= 50 ? 'positive' : 'negative');

            const totalPnl = document.getElementById('total-pnl');
            totalPnl.textContent = (data.stats.total_pnl >= 0 ? '+' : '') + data.stats.total_pnl.toFixed(2) + '%';
            totalPnl.className = 'stat-value ' + (data.stats.total_pnl >= 0 ? 'positive' : 'negative');

            // Update signals table
            const signalsBody = document.querySelector('#signals-table tbody');
            signalsBody.innerHTML = data.signals.map(s => `
                <tr>
                    <td>${s.symbol}</td>
                    <td><span class="badge badge-${s.trend.toLowerCase()}">${s.trend}</span></td>
                    <td>${s.fvg_bottom.toFixed(6)} - ${s.fvg_top.toFixed(6)}</td>
                    <td><span class="badge badge-${s.status.toLowerCase().replace('_', '')}">${s.status}</span></td>
                    <td>${s.updated_at || '-'}</td>
                </tr>
            `).join('');

            // Update trades table
            const tradesBody = document.querySelector('#trades-table tbody');
            tradesBody.innerHTML = data.trades.map(t => {
                const pnlClass = t.pnl_percent >= 0 ? 'positive' : 'negative';
                const pnlText = t.pnl_percent ? ((t.pnl_percent >= 0 ? '+' : '') + t.pnl_percent.toFixed(2) + '%') : '-';
                return `
                    <tr>
                        <td>${t.symbol}</td>
                        <td><span class="badge badge-${t.direction.toLowerCase()}">${t.direction}</span></td>
                        <td>${t.entry_price.toFixed(6)}</td>
                        <td>${t.sl.toFixed(6)}</td>
                        <td>${t.tp.toFixed(6)}</td>
                        <td class="${pnlClass}">${pnlText}</td>
                        <td><span class="badge badge-${t.status.toLowerCase()}">${t.status}</span></td>
                    </tr>
                `;
            }).join('');

            document.getElementById('last-update').textContent = new Date().toLocaleString('nl-NL');
        }

        function refreshData() {
            fetchData();
        }

        // Initial load
        fetchData();

        // Auto refresh every 30 seconds
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
"""


def get_db_connection():
    """Maak database connectie."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def dashboard():
    """Render dashboard."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/data')
def get_data():
    """API endpoint voor dashboard data."""
    conn = get_db_connection()

    try:
        c = conn.cursor()

        # Active symbols count
        c.execute("SELECT COUNT(*) as count FROM active_symbols")
        active_symbols = c.fetchone()['count']

        # Active setups
        c.execute("SELECT COUNT(*) as count FROM signals WHERE status IN ('SCANNING', 'TAPPED')")
        active_setups = c.fetchone()['count']

        # Open trades
        c.execute("SELECT COUNT(*) as count FROM trade_log WHERE status = 'OPEN'")
        open_trades = c.fetchone()['count']

        # Trades today
        today = datetime.now().strftime('%Y-%m-%d')
        c.execute("SELECT COUNT(*) as count FROM trade_log WHERE date(timestamp) = ?", (today,))
        trades_today = c.fetchone()['count']

        # Win rate & PnL
        c.execute("""
            SELECT
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl_percent), 0) as total_pnl
            FROM trade_log
            WHERE status = 'CLOSED'
        """)
        trade_stats = c.fetchone()
        wins = trade_stats['wins'] or 0
        losses = trade_stats['losses'] or 0
        total_closed = wins + losses
        win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
        total_pnl = trade_stats['total_pnl'] or 0

        # Get signals
        c.execute("""
            SELECT symbol, trend, fvg_top, fvg_bottom, status, updated_at
            FROM signals
            ORDER BY updated_at DESC
            LIMIT 20
        """)
        signals = [dict(row) for row in c.fetchall()]

        # Get trades
        c.execute("""
            SELECT symbol, direction, entry_price, sl, tp, pnl_percent, status
            FROM trade_log
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        trades = [dict(row) for row in c.fetchall()]

        return jsonify({
            'stats': {
                'active_symbols': active_symbols,
                'active_setups': active_setups,
                'open_trades': open_trades,
                'trades_today': trades_today,
                'win_rate': win_rate,
                'total_pnl': total_pnl
            },
            'signals': signals,
            'trades': trades
        })

    finally:
        conn.close()


if __name__ == '__main__':
    print(f"Dashboard starting at http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)
