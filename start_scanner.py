#!/usr/bin/env python3
"""
MACD Money Map Scanner - Quick Start
=====================================
Choose which scanner to run:
1. Multi-Scanner (All coins >50k volume, real-time WebSocket)
2. Morning Scan (One-time scan of top coins)
3. Single Coin Analysis
"""

import sys
import os

def print_menu():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MACD MONEY MAP SCANNER                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  [1] Multi-Scanner (Real-time WebSocket)                     ║
║      → Scans ALL coins >50k volume continuously              ║
║      → Alerts when MACD signals appear                       ║
║                                                              ║
║  [2] Morning Scan (One-time)                                 ║
║      → Quick scan of top coins                               ║
║      → Shows current setups                                  ║
║                                                              ║
║  [3] Single Coin Analysis                                    ║
║      → Deep analysis of one coin                             ║
║      → Shows all 3 systems in detail                         ║
║                                                              ║
║  [4] View Active Signals                                     ║
║      → Show signals from database                            ║
║                                                              ║
║  [0] Exit                                                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

def run_multi_scanner():
    """Run the real-time multi-scanner"""
    print("\n🚀 Starting Multi-Scanner...")
    print("   Monitoring ALL coins with >50k volume")
    print("   Press Ctrl+C to stop\n")
    import sys
    os.system(f'"{sys.executable}" macd_multi_scanner.py')

def run_morning_scan():
    """Run a one-time morning scan"""
    print("\n📊 Running Morning Scan...\n")
    from macd_money_map import morning_scan
    signals = morning_scan()
    if signals:
        print(f"\n✅ Found {len(signals)} actionable signals!")
    else:
        print("\n⚠️ No A+ setups found at this time")

def run_single_analysis():
    """Analyze a single coin"""
    symbol = input("\nEnter symbol (e.g. BTC-USDT): ").strip().upper()
    if not symbol:
        symbol = "BTC-USDT"
    
    print(f"\n🔍 Analyzing {symbol}...\n")
    
    from macd_multi_scanner import full_macd_analysis
    result = full_macd_analysis(symbol)
    
    print(f"{'='*60}")
    print(f"MACD MONEY MAP ANALYSIS: {symbol}")
    print(f"{'='*60}")
    
    # Signal status
    if result['signal']:
        direction = "🟢 LONG" if result['signal'] == 'BUY' else "🔴 SHORT"
        print(f"\n🎯 SIGNAL: {direction} (Confidence: {result['confidence']}%)")
        print(f"   Entry:      {result['entry']}")
        print(f"   Stop Loss:  {result['stop_loss']}")
        print(f"   Take Profit:{result['take_profit']}")
    else:
        print(f"\n⚪ No signal at this time")
    
    # Timeframe details
    print(f"\n{'─'*60}")
    print("TIMEFRAME ANALYSIS:")
    print(f"{'─'*60}")
    
    for tf, data in result['details'].items():
        if data.get('valid'):
            bias_icon = "🟢" if data['bias'] == 'BULLISH' else "🔴" if data['bias'] == 'BEARISH' else "⚪"
            print(f"\n{tf.upper()} Timeframe:")
            print(f"  {bias_icon} Bias: {data['bias']}")
            print(f"  MACD: {data['macd']:.6f}")
            print(f"  Crossover: {data['crossover'] or 'None'}")
            print(f"  Histogram: {data['hist_pattern'] or 'No pattern'}")
            print(f"  Support/Resistance: {'At Support' if data['at_support'] else 'At Resistance' if data['at_resistance'] else 'Middle'}")

def view_signals():
    """View signals from database"""
    import sqlite3
    
    conn = sqlite3.connect('trading_bot.db')
    c = conn.cursor()
    
    try:
        c.execute("SELECT * FROM macd_signals ORDER BY timestamp DESC LIMIT 20")
        signals = c.fetchall()
        
        if not signals:
            print("\n📭 No signals in database yet")
            return
        
        print(f"\n{'='*80}")
        print("RECENT MACD SIGNALS")
        print(f"{'='*80}")
        print(f"{'ID':<4} {'Symbol':<12} {'Signal':<8} {'Conf':<6} {'Entry':<12} {'Time'}")
        print(f"{'─'*80}")
        
        for s in signals:
            print(f"{s[0]:<4} {s[1]:<12} {s[2]:<8} {s[4]:<6} {s[5] or 'N/A':<12} {s[9]}")
        
    except Exception as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def main():
    # Initialize database
    import subprocess
    import sys
    subprocess.run([sys.executable, "init_db.py"], capture_output=True)
    
    while True:
        print_menu()
        choice = input("Select option [0-4]: ").strip()
        
        if choice == '1':
            run_multi_scanner()
        elif choice == '2':
            run_morning_scan()
        elif choice == '3':
            run_single_analysis()
        elif choice == '4':
            view_signals()
        elif choice == '0':
            print("\n👋 Goodbye!\n")
            break
        else:
            print("\n❌ Invalid option, try again")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
