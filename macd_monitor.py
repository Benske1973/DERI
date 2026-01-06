"""
MACD Money Map - Real-Time Monitor
===================================
Continuous monitoring for MACD signals across all 3 systems

Features:
- Periodic scanning of all configured symbols
- Alert on new A+ setups
- Track signal status changes
- Save signals to database
"""

import time
import sqlite3
from datetime import datetime
from typing import List, Dict
from macd_money_map import MACDMoneyMap, CONFIG, morning_scan


class MACDMonitor:
    """
    Real-time MACD Money Map monitor
    """
    
    def __init__(self, scan_interval: int = 300):  # Default: 5 minutes
        self.scanner = MACDMoneyMap()
        self.scan_interval = scan_interval
        self.active_signals = {}
        self.setup_database()
    
    def setup_database(self):
        """Ensure database tables exist"""
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        
        # MACD signals table
        c.execute('''CREATE TABLE IF NOT EXISTS macd_signals 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      symbol TEXT,
                      signal TEXT,
                      signal_type TEXT,
                      confidence INTEGER,
                      entry_price REAL,
                      stop_loss REAL,
                      take_profit REAL,
                      trend_bias TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                      status TEXT DEFAULT 'ACTIVE')''')
        
        # Alert history
        c.execute('''CREATE TABLE IF NOT EXISTS macd_alerts
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      symbol TEXT,
                      alert_type TEXT,
                      message TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        
        conn.commit()
        conn.close()
    
    def log_alert(self, symbol: str, alert_type: str, message: str):
        """Log an alert to database"""
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        c.execute("INSERT INTO macd_alerts (symbol, alert_type, message) VALUES (?, ?, ?)",
                  (symbol, alert_type, message))
        conn.commit()
        conn.close()
    
    def check_for_new_signals(self, results: List[Dict]) -> List[Dict]:
        """Check if there are any new signals compared to last scan"""
        new_signals = []
        
        for result in results:
            symbol = result['symbol']
            current_signal = result['final_signal']
            
            # Check if this is a new or changed signal
            previous = self.active_signals.get(symbol, {})
            previous_signal = previous.get('signal')
            
            if current_signal in ['BUY', 'SELL']:
                if current_signal != previous_signal:
                    new_signals.append(result)
                    self.active_signals[symbol] = {
                        'signal': current_signal,
                        'timestamp': datetime.now(),
                        'confidence': result['confidence']
                    }
        
        return new_signals
    
    def print_alert(self, signal: Dict):
        """Print a prominent alert for new signals"""
        print("\n" + "🚨" * 30)
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎯 NEW SIGNAL ALERT 🎯                     ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol:      {signal['symbol']:<45} ║
║  Signal:      {signal['final_signal']:<45} ║
║  Type:        {signal['signal_type'] or 'N/A':<45} ║
║  Confidence:  {str(signal['confidence']) + '%':<45} ║
║  Entry:       {str(round(signal['entry'], 6)) if signal['entry'] else 'N/A':<45} ║
║  Stop Loss:   {str(round(signal['stop_loss'], 6)) if signal['stop_loss'] else 'N/A':<45} ║
║  Take Profit: {str(round(signal['take_profit'], 6)) if signal['take_profit'] else 'N/A':<45} ║
║  Time:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<45} ║
╚══════════════════════════════════════════════════════════════╝
        """)
        print("🚨" * 30 + "\n")
    
    def display_dashboard(self, results: List[Dict]):
        """Display a live dashboard of all symbols"""
        print("\n" + "=" * 80)
        print(f"📊 MACD MONEY MAP DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Sort by confidence (signals first)
        sorted_results = sorted(results, key=lambda x: (
            x['final_signal'] in ['BUY', 'SELL'],
            x['confidence'] or 0
        ), reverse=True)
        
        print(f"\n{'Symbol':<12} {'Bias':<10} {'Signal':<12} {'Conf':<6} {'Entry':<12} {'Status'}")
        print("-" * 80)
        
        for r in sorted_results:
            bias = r['timeframes'].get('setup', {}).get('trend', {}).get('bias', 'N/A')
            signal = r['final_signal'] or 'NONE'
            conf = f"{r['confidence']}%" if r['confidence'] else '-'
            entry = f"{r['entry']:.6f}" if r['entry'] else '-'
            
            # Color coding
            if signal == 'BUY':
                status = '🟢 LONG'
            elif signal == 'SELL':
                status = '🔴 SHORT'
            elif signal == 'PENDING':
                status = '⏳ WAIT'
            else:
                status = '⚪ -'
            
            print(f"{r['symbol']:<12} {bias:<10} {signal:<12} {conf:<6} {entry:<12} {status}")
        
        # Active signals summary
        active_buys = sum(1 for r in results if r['final_signal'] == 'BUY')
        active_sells = sum(1 for r in results if r['final_signal'] == 'SELL')
        pending = sum(1 for r in results if r['final_signal'] == 'PENDING')
        
        print("\n" + "-" * 80)
        print(f"Summary: 🟢 {active_buys} BUY signals | 🔴 {active_sells} SELL signals | ⏳ {pending} pending")
        print("=" * 80)
    
    def run_continuous(self):
        """Run continuous monitoring loop"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║           MACD MONEY MAP - REAL-TIME MONITOR                 ║
║                                                              ║
║  Scanning for:                                               ║
║  • System 1: Trend signals (Zero Line + Crossovers)          ║
║  • System 2: Reversal signals (Divergence + Histogram)       ║
║  • System 3: Confirmation (Triple Timeframe Alignment)       ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                print(f"\n🔍 Scan #{scan_count} starting...")
                
                # Run scan
                results = self.scanner.scan_all()
                
                # Check for new signals
                new_signals = self.check_for_new_signals(results)
                
                # Alert on new signals
                for signal in new_signals:
                    self.print_alert(signal)
                    self.log_alert(
                        signal['symbol'],
                        signal['final_signal'],
                        f"{signal['signal_type']} signal at {signal['entry']} (Confidence: {signal['confidence']}%)"
                    )
                
                # Save to database
                self.scanner.save_signals_to_db(results)
                
                # Display dashboard
                self.display_dashboard(results)
                
                # Wait for next scan
                print(f"\n⏰ Next scan in {self.scan_interval} seconds...")
                time.sleep(self.scan_interval)
        
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user")
            self.display_final_summary()
    
    def display_final_summary(self):
        """Display summary of the monitoring session"""
        print("\n" + "=" * 60)
        print("📊 SESSION SUMMARY")
        print("=" * 60)
        
        conn = sqlite3.connect('trading_bot.db')
        c = conn.cursor()
        
        # Recent alerts
        c.execute("SELECT * FROM macd_alerts ORDER BY timestamp DESC LIMIT 10")
        alerts = c.fetchall()
        
        if alerts:
            print("\nRecent Alerts:")
            for alert in alerts:
                print(f"  [{alert[4]}] {alert[1]}: {alert[2]} - {alert[3]}")
        
        # Active signals
        c.execute("SELECT * FROM macd_signals WHERE status='ACTIVE' ORDER BY timestamp DESC LIMIT 10")
        signals = c.fetchall()
        
        if signals:
            print("\nActive Signals:")
            for sig in signals:
                print(f"  {sig[1]}: {sig[2]} ({sig[3]}) - Entry: {sig[5]}, SL: {sig[6]}, TP: {sig[7]}")
        
        conn.close()


def quick_scan():
    """Run a single quick scan"""
    monitor = MACDMonitor()
    results = monitor.scanner.scan_all()
    monitor.display_dashboard(results)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--once':
        # Single scan
        quick_scan()
    else:
        # Continuous monitoring
        interval = int(sys.argv[1]) if len(sys.argv) > 1 else 300
        monitor = MACDMonitor(scan_interval=interval)
        monitor.run_continuous()
