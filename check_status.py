# check_status.py - Status check met verbeterde output
import sqlite3
from datetime import datetime
from config import DATABASE_PATH


def check_status():
    """Toon huidige status van de trading bot."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("\n" + "=" * 60)
    print("DERI TRADING BOT STATUS")
    print(f"Tijdstip: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Active symbols
    c.execute("SELECT COUNT(*) as count FROM active_symbols")
    active_count = c.fetchone()['count']
    print(f"\nActieve symbols (>50K volume): {active_count}")

    # Scanning signals
    print("\n--- ACTIEVE POI's (Wachten op Tap) ---")
    c.execute("""
        SELECT symbol, trend, fvg_top, fvg_bottom, updated_at
        FROM signals
        WHERE status = 'SCANNING'
        ORDER BY updated_at DESC
    """)
    scanning = c.fetchall()
    if scanning:
        for row in scanning:
            arrow = "🟢" if row['trend'] == 'BULLISH' else "🔴"
            print(f"{arrow} {row['symbol']:12} | {row['trend']:8} | Zone: {row['fvg_bottom']:.6f} - {row['fvg_top']:.6f}")
    else:
        print("   Geen actieve POI's")

    # Tapped signals
    print("\n--- GETAPTE COINS (Wachten op LTF Confirmatie) ---")
    c.execute("""
        SELECT symbol, trend, fvg_top, fvg_bottom
        FROM signals
        WHERE status = 'TAPPED'
    """)
    tapped = c.fetchall()
    if tapped:
        for row in tapped:
            arrow = "🟢" if row['trend'] == 'BULLISH' else "🔴"
            print(f"🎯 {arrow} {row['symbol']:12} | {row['trend']}")
    else:
        print("   Geen getapte coins")

    # In trade
    print("\n--- ACTIEVE TRADES ---")
    c.execute("""
        SELECT symbol, direction, entry_price, sl, tp
        FROM trade_log
        WHERE status = 'OPEN'
    """)
    trades = c.fetchall()
    if trades:
        for row in trades:
            arrow = "🟢" if row['direction'] == 'BULLISH' else "🔴"
            print(f"{arrow} {row['symbol']:12} | Entry: {row['entry_price']:.6f} | SL: {row['sl']:.6f} | TP: {row['tp']:.6f}")
    else:
        print("   Geen actieve trades")

    # Quick stats
    print("\n--- SNELLE STATISTIEKEN ---")
    c.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            COALESCE(SUM(pnl_percent), 0) as total_pnl
        FROM trade_log
        WHERE status = 'CLOSED'
    """)
    stats = c.fetchone()
    total = stats['total'] or 0
    wins = stats['wins'] or 0
    losses = stats['losses'] or 0
    total_pnl = stats['total_pnl'] or 0
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    print(f"   Totaal gesloten trades: {total}")
    print(f"   Wins: {wins} | Losses: {losses}")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Totale PnL: {total_pnl:+.2f}%")

    print("\n" + "=" * 60)
    conn.close()


if __name__ == "__main__":
    check_status()
