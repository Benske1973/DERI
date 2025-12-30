# view_stats.py - Gedetailleerde statistieken viewer
import sqlite3
from datetime import datetime, timedelta
from config import DATABASE_PATH


def view_stats():
    """Toon gedetailleerde trade statistieken."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("\n" + "=" * 70)
    print("DERI TRADING BOT - GEDETAILLEERDE STATISTIEKEN")
    print(f"Gegenereerd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Overall stats
    print("\n📊 ALGEMENE STATISTIEKEN")
    print("-" * 40)

    c.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(pnl_percent), 0) as total_pnl,
            COALESCE(AVG(CASE WHEN status = 'CLOSED' THEN pnl_percent END), 0) as avg_pnl,
            COALESCE(MAX(pnl_percent), 0) as best_trade,
            COALESCE(MIN(pnl_percent), 0) as worst_trade
        FROM trade_log
    """)
    stats = c.fetchone()

    total = stats['total'] or 0
    open_trades = stats['open_trades'] or 0
    closed = stats['closed_trades'] or 0
    wins = stats['wins'] or 0
    losses = stats['losses'] or 0
    win_rate = (wins / closed * 100) if closed > 0 else 0

    print(f"Totaal trades:     {total}")
    print(f"Open trades:       {open_trades}")
    print(f"Gesloten trades:   {closed}")
    print(f"Wins:              {wins}")
    print(f"Losses:            {losses}")
    print(f"Win Rate:          {win_rate:.1f}%")
    print(f"Totale PnL:        {stats['total_pnl']:+.2f}%")
    print(f"Gemiddelde PnL:    {stats['avg_pnl']:+.2f}%")
    print(f"Beste trade:       {stats['best_trade']:+.2f}%")
    print(f"Slechtste trade:   {stats['worst_trade']:+.2f}%")

    # Stats per direction
    print("\n📈 STATISTIEKEN PER RICHTING")
    print("-" * 40)

    for direction in ['BULLISH', 'BEARISH']:
        c.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl_percent), 0) as total_pnl
            FROM trade_log
            WHERE direction = ? AND status = 'CLOSED'
        """, (direction,))
        dir_stats = c.fetchone()

        dir_total = dir_stats['wins'] + dir_stats['losses'] if dir_stats else 0
        dir_wr = (dir_stats['wins'] / dir_total * 100) if dir_total > 0 else 0

        emoji = "🟢" if direction == 'BULLISH' else "🔴"
        print(f"\n{emoji} {direction}:")
        print(f"   Trades: {dir_total} | W: {dir_stats['wins'] or 0} | L: {dir_stats['losses'] or 0}")
        print(f"   Win Rate: {dir_wr:.1f}%")
        print(f"   PnL: {dir_stats['total_pnl'] or 0:+.2f}%")

    # Daily breakdown
    print("\n📅 LAATSTE 7 DAGEN")
    print("-" * 40)

    c.execute("""
        SELECT
            date(timestamp) as day,
            COUNT(*) as trades,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(pnl_percent), 0) as pnl
        FROM trade_log
        WHERE status = 'CLOSED'
            AND timestamp >= datetime('now', '-7 days')
        GROUP BY date(timestamp)
        ORDER BY day DESC
    """)
    daily = c.fetchall()

    if daily:
        print(f"{'Datum':<12} {'Trades':<8} {'W':<4} {'L':<4} {'WR%':<8} {'PnL':<10}")
        print("-" * 50)
        for day in daily:
            total_day = day['wins'] + day['losses']
            wr = (day['wins'] / total_day * 100) if total_day > 0 else 0
            pnl_color = "+" if day['pnl'] >= 0 else ""
            print(f"{day['day']:<12} {day['trades']:<8} {day['wins']:<4} {day['losses']:<4} {wr:<8.1f} {pnl_color}{day['pnl']:.2f}%")
    else:
        print("Geen trades in de laatste 7 dagen")

    # Recent trades
    print("\n📝 LAATSTE 10 TRADES")
    print("-" * 70)

    c.execute("""
        SELECT symbol, direction, entry_price, exit_price, pnl_percent, result, timestamp
        FROM trade_log
        WHERE status = 'CLOSED'
        ORDER BY timestamp DESC
        LIMIT 10
    """)
    recent = c.fetchall()

    if recent:
        print(f"{'Symbol':<12} {'Dir':<8} {'Entry':<12} {'Exit':<12} {'PnL':<10} {'Result':<8}")
        print("-" * 70)
        for t in recent:
            result_emoji = "✅" if t['result'] == 'WIN' else "❌"
            pnl_str = f"{t['pnl_percent']:+.2f}%" if t['pnl_percent'] else "-"
            exit_str = f"{t['exit_price']:.6f}" if t['exit_price'] else "-"
            print(f"{t['symbol']:<12} {t['direction']:<8} {t['entry_price']:.6f}   {exit_str:<12} {pnl_str:<10} {result_emoji}")
    else:
        print("Geen gesloten trades")

    # Top performers
    print("\n🏆 TOP 5 BEST PRESTERENDE SYMBOLS")
    print("-" * 40)

    c.execute("""
        SELECT
            symbol,
            COUNT(*) as trades,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            COALESCE(SUM(pnl_percent), 0) as total_pnl
        FROM trade_log
        WHERE status = 'CLOSED'
        GROUP BY symbol
        HAVING trades >= 1
        ORDER BY total_pnl DESC
        LIMIT 5
    """)
    top = c.fetchall()

    if top:
        for i, t in enumerate(top, 1):
            wr = (t['wins'] / t['trades'] * 100) if t['trades'] > 0 else 0
            print(f"{i}. {t['symbol']:<12} | Trades: {t['trades']:<3} | WR: {wr:.0f}% | PnL: {t['total_pnl']:+.2f}%")
    else:
        print("Nog geen data beschikbaar")

    print("\n" + "=" * 70)
    conn.close()


if __name__ == "__main__":
    view_stats()
